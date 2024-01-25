# Copyright (c) InternLM. All rights reserved.
"""
``convert2hf_internlm2.py`` is used to convert INTERNLM2's checkpoint to huggingface format.

python transformers/convert2hf_internlm2.py --src /path/to/src --tgt /path/to/tgt \
       --max_shard 2GB --max_pos 32768 \
       --tokenizer /path/to/tokenizer.model \
       --rotary_type origin
```
"""
import argparse
import gc
import json
import os
import re
import sys
import time

import torch
from einops import rearrange
from internlm2_model import (
    InternLM2Config,
    InternLM2ForCausalLM,
    InternLM2TokenizerFast,
)
from tqdm import tqdm

from transformers.modeling_utils import no_init_weights

sys.path.insert(0, os.getcwd())

try:
    from internlm.utils.storage_manager import (
        check_folder,
        get_fns,
        init_storage_manager,
        llm_load,
    )

    init_storage_manager(False, None, None)
except ImportError:

    def get_fns(folder):
        return os.listdir(folder)

    def llm_load(fn, map_location="cpu"):
        return torch.load(fn, map_location=map_location)

    def check_folder(fp):
        assert os.path.exists(fp)


basic_config = dict(
    num_chunks=1,
    checkpoint=False,
    dtype=torch.half,
    embed_split_hidden=False,
    num_layers=40,
    hidden_size=5120,
    vocab_size=150494,
    embed_grad_scale=1,
    parallel_output=False,
    num_attention_heads=40,
    mlp_ratio=8 / 3,
    apply_post_layer_norm=False,
    no_bias=True,
    deepnorm=False,
    residual_in_fp32=False,
    norm_type="rmsnorm",
    drop_rate=0,
    attn_drop_rate=0,
    model_type="llama",
    adapt_hf=False,
    norm_head=False,
)
embedding_key_list = ["tok_embeddings.word_embeddings.weight", "tok_embeddings.weight", None]


def _find_max_tp_pp(names):
    ckpt_names = []
    for name in names:
        if name.startswith("model_t") and not name.endswith("md5"):
            # _t: avoid conflictint with model_config.pt
            ckpt_names.append(name)

    max_tp, max_pp = -1, -1
    for ckpt in ckpt_names:
        _, tp, pp = os.path.splitext(ckpt)[0].split("_")
        max_tp = max(max_tp, int(tp[2:]) + 1)
        max_pp = max(max_pp, int(pp[2:]) + 1)

    return max_tp, max_pp


def load_source(src):
    """
    load model_config.pt and model_tp{x}_pp{x}.pt from ``src``

    :return:
        - model_config: dict
        - states: 2-d array. states[i][j] stands for state_dict of tp_i pp_j
    """

    # config
    print("Config loading", flush=True)
    config_file = os.path.join(src, "model_config.pt")
    check_folder(config_file)
    update_config = llm_load(config_file)
    print(update_config)
    model_config = basic_config
    model_config.update(update_config)
    assert model_config["no_bias"], "Model with bias is not supported when model_type is llama."
    print("Config loaded.", flush=True)

    # checkpoint
    # find tp pp
    ckpt_names = get_fns(src)
    max_tp, max_pp = _find_max_tp_pp(ckpt_names)

    # 2-d array tp_rank, pp_rank
    print("Source Checkpoint Loading", flush=True)
    states = [[None for _ in range(max_pp)] for __ in range(max_tp)]
    for tp in tqdm(range(max_tp)):
        for pp in tqdm(range(max_pp)):
            ckpt_name = os.path.join(src, f"model_tp{tp}_pp{pp}.pt")
            states[tp][pp] = llm_load(ckpt_name, map_location="cpu")
    print("Source Checkpoint Loaded", flush=True)
    return model_config, states


def merge(states):
    """
    Merge state dicts of pipeline format and shift some layers.

    :return:
        - config: InternLMConfig
        - states: merged state dict
    """
    # merge pp
    merged_states = []
    print("Pipeline Merging", flush=True)
    for tp_state in tqdm(states):
        layer_shift = 0
        shifted_state = {}
        # shift key
        for tp_pp_state in tp_state:
            _layer_shift = 0
            keys = list(tp_pp_state.keys())
            for key in keys:
                if key.endswith(".inv_freq"):
                    continue
                match = re.search(r"\.\d+\.", key)
                name = key
                if match is not None:
                    # layers
                    s, e = match.span()
                    layer_idx = int(key[s + 1 : e - 1]) + layer_shift
                    _layer_shift = max(_layer_shift, int(key[s + 1 : e - 1]))
                    name = key[:s] + f".{layer_idx}." + key[e:]
                if name.startswith("model."):
                    name = name[6:]
                shifted_state[name] = tp_pp_state[key]
            layer_shift += _layer_shift + 1

        merged_states.append(shifted_state)

    print("Pipeline Merged", flush=True)

    return merged_states


def permute(qkv, num_heads, num_kv_heads, head_dim, adapt_hf=True):
    if adapt_hf:
        return qkv

    print(f"adapt_hf is {adapt_hf}, do permuting...")
    q_per_kv = num_heads // num_kv_heads
    qkv = rearrange(qkv.T, "o (g n i) -> o g n i", n=q_per_kv + 2, i=head_dim)
    q, k, v = qkv[..., :q_per_kv, :], qkv[..., -2:-1, :], qkv[..., -1:, :]
    q = torch.cat([q[..., ::2], q[..., 1::2]], dim=-1)
    k = torch.cat([k[..., ::2], k[..., 1::2]], dim=-1)
    qkv = torch.cat((q, k, v), dim=2)
    qkv = rearrange(qkv, "o g n i -> o (g n i)").T
    return qkv


def convert(src, tgt, tokenizer, dtype, max_shard_size, max_pos, rope_scaling):
    """
    Convert state_dict to hf format.

    1. Load and merge state dict
    2. Convert to huggingface
    3. Load tokneizer and save it with ``tokenizer.save_pretrained``
    4. Load state dict to the model
    5. Call ``model.save_pretrained`` to save checkpoints.
    """
    # load states
    model_config, src_states = load_source(src)
    states = merge(src_states)
    del src_states

    num_shards = len(states)
    print("Converting to huggingface format...", flush=True)

    intermediate_size = None

    print("Start converting...", flush=True)
    state_dict = {}
    for layer_i in tqdm(range(model_config["num_layers"])):
        state_dict.update(
            {
                f"model.layers.{layer_i}.attention_norm.weight": states[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"model.layers.{layer_i}.ffn_norm.weight": states[0][f"layers.{layer_i}.ffn_norm.weight"].clone(),
            }
        )
        state_dict[f"model.layers.{layer_i}.attention.wqkv.weight"] = permute(
            torch.cat([states[i][f"layers.{layer_i}.attention.wqkv.weight"] for i in range(num_shards)], dim=0),
            num_heads=model_config["num_attention_heads"],
            num_kv_heads=model_config["num_kv_attention_heads"],
            head_dim=model_config["hidden_size"] // model_config["num_attention_heads"],
            adapt_hf=model_config.get("adapt_hf", True),
        )

        state_dict[f"model.layers.{layer_i}.attention.wo.weight"] = torch.cat(
            [states[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
        )
        state_dict[f"model.layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
            [states[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
        )

        intermediate_size = states[0][f"layers.{layer_i}.feed_forward.w2.weight"].shape[1] * num_shards
        state_dict[f"model.layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
            [states[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
        )
        state_dict[f"model.layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
            [states[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
        )

    # embedding
    if model_config["embed_split_hidden"]:
        embed_concat_dim = 1
    else:
        embed_concat_dim = 0
    for embedding_key in embedding_key_list:
        if embedding_key in states[0]:
            break
    if embedding_key is None:
        raise KeyError("Cannot find embedding key!")
    state_dict.update(
        {
            "model.norm.weight": states[0]["norm.weight"],
            "model.tok_embeddings.weight": torch.cat(
                [states[i][embedding_key] for i in range(num_shards)], dim=embed_concat_dim
            ),
        },
    )
    if model_config["norm_head"]:
        state_dict["output.weight"] = torch.nn.functional.normalize(
            torch.cat([states[i]["output.weight"] for i in range(num_shards)], dim=0), dim=-1
        )
    else:
        state_dict["output.weight"] = torch.cat([states[i]["output.weight"] for i in range(num_shards)], dim=0)

    # initialize model
    # tokenizer
    tokenizer = InternLM2TokenizerFast(tokenizer)
    # config
    config = InternLM2Config(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=intermediate_size,
        num_attention_heads=model_config["num_attention_heads"],
        num_hidden_layers=model_config["num_layers"],
        rms_norm_eps=model_config["layer_norm_epsilon"],
        bias=not model_config["no_bias"],
        rope_theta=model_config.get("rope_base", 10000),
        rope_scaling=rope_scaling,
    )
    if "num_kv_attention_heads" in model_config:
        config.num_key_value_heads = model_config["num_kv_attention_heads"]
    # tokenizer
    config.max_position_embeddings = max_pos
    # set bos eos pad to avoid improper generation
    # since model.generate will create attention_mask
    # according to pad_token_id and bos_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    # model
    print("Initializing model...", flush=True)
    start = time.time()
    with no_init_weights():
        model = InternLM2ForCausalLM._from_config(config, torch_dtype=dtype)
    print(f"Initializing model takes {time.time() - start}s", flush=True)
    model.load_state_dict(state_dict)

    del states
    gc.collect()
    print(f"Saving model to {tgt}...", flush=True)
    tokenizer.save_pretrained(tgt)
    model.save_pretrained(tgt, max_shard_size=max_shard_size)

    # fix auto_map in config
    with open(os.path.join(tgt, "config.json")) as fp:
        config_dict = json.load(fp)
    config_dict["auto_map"]["AutoModel"] = "modeling_internlm2.InternLM2ForCausalLM"
    with open(os.path.join(tgt, "config.json"), "w") as fp:
        json.dump(config_dict, fp, indent=2)


def get_rope_scaling(args):
    if args.rotary_type == "origin":
        return None
    elif args.rotary_type == "dynamic":
        return {"type": args.rotary_type, "factor": args.scaling_factor}
    else:
        raise NotImplementedError(f"Unknown rope type {args.rotary_type}")


def print_args(args):
    print("-------------- Arguments --------------")
    print(f"Source Path: {args.src}")
    print(f"Target Path: {args.tgt}")
    print(f"Dtype: {args.dtype}")
    print(f"Max Shard Size: {args.max_shard}")
    print(f"Max Position Embedding: {args.max_pos}")
    print(f"Tokenizer Path: {args.tokenizer}")
    print(f"Rotary Type: {args.rotary_type}")
    print(f"Scaling Factor: {args.scaling_factor}")
    print("---------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--src", type=str, default=None, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="Data type after converting")
    parser.add_argument("--max_shard", type=str, default="10GB", help="Max size of every sharded checkpoint.")
    parser.add_argument("--max_pos", type=int, default=4096, help="Max position embedding of model.")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer model.")
    # rope
    parser.add_argument(
        "--rotary_type", type=str, default="dynamic", choices=["dynamic", "origin"], help="Type of rotary embedding"
    )
    parser.add_argument("--scaling_factor", type=float, default=2.0, help="Scaling factor of dynamic rotary embedding")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    dtype = getattr(torch, args.dtype)
    rope_scaling = get_rope_scaling(args)

    assert args.src is not None, "--src is needed!"
    assert args.tokenizer is not None, "--tokenizer is needed!"
    start = time.time()
    convert(args.src, args.tgt, args.tokenizer, dtype, args.max_shard, args.max_pos, rope_scaling)
    print(f"Converting model takes {time.time() - start}s totally", flush=True)
