# Copyright (c) InternLM. All rights reserved.
"""
python revert_internlm.py  --src /path/to/src --tgt /path/to/tgt \
    --tp_size 2 --embed_split --use_flash --version 1
"""
import argparse
import os

import torch
from tqdm import tqdm

from transformers import AutoConfig


def load_safetensors(filename):
    from safetensors import safe_open

    model = safe_open(filename, framework="pt")
    state_dict = {}
    for k in model.keys():
        state_dict[k] = model.get_tensor(k)

    return state_dict


def revert(
    src: str,
    tgt: str,
    tp_size: int,
    embed_split_hidden: bool,
    adapt_hf: bool,
    use_flash: bool,
    version: int,
    safetensors: bool,
):
    hf_state = {}
    print("Loading HF checkpoints...")
    suffix = ".bin" if not safetensors else ".safetensors"
    load_func = torch.load if not safetensors else load_safetensors
    for filename in tqdm(os.listdir(src)):
        if not filename.endswith(suffix):
            continue
        hf_state.update(load_func(os.path.join(src, filename)))
    print("Reverting HF checkpoints to InternLM...")
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)

    n_heads = config.num_attention_heads
    try:
        n_kv_heads = config.num_key_value_heads
    except AttributeError:
        n_kv_heads = n_heads
    dim = config.hidden_size

    n_heads_per_shard = n_heads // tp_size
    dims_per_head = dim // n_heads

    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        if adapt_hf:
            return w
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    # revert
    states = [{} for _ in range(tp_size)]

    # layers
    for layer_i in tqdm(range(config.num_hidden_layers)):
        for i in range(tp_size):
            states[i][f"layers.{layer_i}.attention_norm.weight"] = hf_state[
                f"model.layers.{layer_i}.input_layernorm.weight"
            ].clone()
            states[i][f"layers.{layer_i}.ffn_norm.weight"] = hf_state[
                f"model.layers.{layer_i}.post_attention_layernorm.weight"
            ].clone()

        wqs = (
            permute(hf_state[f"model.layers.{layer_i}.self_attn.q_proj.weight"])
            .view(n_heads_per_shard, -1, dim)
            .chunk(tp_size, 0)
        )
        wks = (
            permute(hf_state[f"model.layers.{layer_i}.self_attn.k_proj.weight"], n_kv_heads, -1, dim)
            .view(-1, dims_per_head, dim)
            .chunk(tp_size, 0)
        )
        wvs = hf_state[f"model.layers.{layer_i}.self_attn.v_proj.weight"].view(-1, dims_per_head, dim).chunk(tp_size, 0)
        wos = hf_state[f"model.layers.{layer_i}.self_attn.o_proj.weight"].chunk(tp_size, 1)
        w1s = hf_state[f"model.layers.{layer_i}.mlp.gate_proj.weight"].chunk(tp_size, 0)
        if version == 1:
            w3s = hf_state[f"model.layers.{layer_i}.mlp.down_proj.weight"].chunk(tp_size, 1)
            w2s = hf_state[f"model.layers.{layer_i}.mlp.up_proj.weight"].chunk(tp_size, 0)
        else:
            w2s = hf_state[f"model.layers.{layer_i}.mlp.down_proj.weight"].chunk(tp_size, 1)
            w3s = hf_state[f"model.layers.{layer_i}.mlp.up_proj.weight"].chunk(tp_size, 0)

        for i in range(tp_size):
            states[i][f"layers.{layer_i}.attention.wq.weight"] = wqs[i].reshape(-1, dim).clone()
            states[i][f"layers.{layer_i}.attention.wk.weight"] = wks[i].reshape(-1, dim).clone()
            states[i][f"layers.{layer_i}.attention.wv.weight"] = wvs[i].reshape(-1, dim).clone()
            states[i][f"layers.{layer_i}.attention.wo.weight"] = wos[i].clone()
            states[i][f"layers.{layer_i}.feed_forward.w1.weight"] = w1s[i].clone()
            states[i][f"layers.{layer_i}.feed_forward.w3.weight"] = w3s[i].clone()
            states[i][f"layers.{layer_i}.feed_forward.w2.weight"] = w2s[i].clone()

    if embed_split_hidden:
        embed_concat_dim = 1
    else:
        embed_concat_dim = 0
    embeds = hf_state["model.embed_tokens.weight"].chunk(tp_size, embed_concat_dim)
    outputs = hf_state["lm_head.weight"].chunk(tp_size, 0)
    for i in range(tp_size):
        states[i]["norm.weight"] = hf_state["model.norm.weight"].clone()
        states[i]["tok_embeddings.weight"] = embeds[i].clone()
        states[i]["output.weight"] = outputs[i].clone()

    mlp_ratio = round((config.intermediate_size - 255) / config.hidden_size + 0.01, 2)
    if "rotary" in config.to_dict():
        rope_base = config.rotary["base"]
    elif "rope_theta" in config.to_dict():
        rope_base = config.rope_theta
    else:
        rope_base = 10000
    model_config = dict(
        num_attention_heads=n_heads,
        embed_split_hidden=embed_split_hidden,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        norm_type="rmsnorm",
        layer_norm_epsilon=config.rms_norm_eps,
        no_bias=True,
        mlp_ratio=mlp_ratio,
        num_kv_attention_heads=n_kv_heads,
        dtype=config.torch_dtype,
        norm_head=False,
        adapt_hf=adapt_hf,
        use_flash_attn=use_flash,
        rope_base=rope_base,
    )
    print("Model Config:", model_config)

    # split
    os.makedirs(tgt, exist_ok=True)
    print(f"Saving to {tgt}...")
    for tp in tqdm(range(tp_size)):
        torch.save(states[tp], os.path.join(tgt, f"model_tp{tp}_pp0.pt"))
    torch.save(model_config, os.path.join(tgt, "model_config.pt"))


def print_args(args):
    print("-------------- Arguments --------------")
    print(f"Source Path: {args.src}")
    print(f"Target Path: {args.tgt}")
    print(f"Embeb Split Hidden: {args.embed_split}")
    print(f"Adapt HF: {args.adapt_hf}")
    print(f"Use Flash Attn: {args.use_flash}")
    if args.version == 1:
        print("Version: w2->up, w3->down")
    elif args.version == 2:
        print("Version: w3->up, w2->down")
    else:
        raise NotImplementedError
    print("---------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--src", type=str, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--tp_size", type=int, help="world_size of tensor parallel")
    parser.add_argument("--embed_split", action="store_true", help="embed_split_hidden of InternLM")
    parser.add_argument("--adapt_hf", action="store_true", help="adapt_hf of InternLM")
    parser.add_argument("--use_flash", action="store_true", help="use_flash_attn of InternLM")
    parser.add_argument("--version", type=int, help="Determine the relavance between w2, w3 and up_gate, down_fate.")
    parser.add_argument("--safetensors", action="store_true", help="Whether the model is saved as safetensors")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    revert(
        args.src,
        args.tgt,
        args.tp_size,
        args.embed_split,
        args.adapt_hf,
        args.use_flash,
        args.version,
        args.safetensors,
    )
