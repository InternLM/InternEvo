# Copyright (c) InternLM. All rights reserved.
"""
python revert_internlm2.py  --src /path/to/src --tgt /path/to/tgt \
    --tp_size 2 --embed_split --use_flash
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


def revert(src: str, tgt: str, tp_size: int, embed_split_hidden: bool, use_flash: bool, safetensors: bool):
    hf_state = {}
    print("Loading HF checkpoints...")
    suffix = ".bin" if not safetensors else ".safetensors"
    load_func = torch.load if not safetensors else load_safetensors
    for filename in tqdm(os.listdir(src)):
        if not filename.endswith(suffix):
            continue
        hf_state.update(load_func(os.path.join(src, filename)))
    print("Reverting HF checkpoints to InternLM2...")
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)

    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads

    # revert
    states = [{} for _ in range(tp_size)]

    # layers
    for layer_i in tqdm(range(config.num_hidden_layers)):
        for i in range(tp_size):
            states[i][f"layers.{layer_i}.attention_norm.weight"] = hf_state[
                f"model.layers.{layer_i}.attention_norm.weight"
            ].clone()
            states[i][f"layers.{layer_i}.ffn_norm.weight"] = hf_state[f"model.layers.{layer_i}.ffn_norm.weight"].clone()

        wqkvs = hf_state[f"model.layers.{layer_i}.attention.wqkv.weight"].chunk(tp_size, 0)
        wos = hf_state[f"model.layers.{layer_i}.attention.wo.weight"].chunk(tp_size, 1)
        w1s = hf_state[f"model.layers.{layer_i}.feed_forward.w1.weight"].chunk(tp_size, 0)
        w2s = hf_state[f"model.layers.{layer_i}.feed_forward.w2.weight"].chunk(tp_size, 1)
        w3s = hf_state[f"model.layers.{layer_i}.feed_forward.w3.weight"].chunk(tp_size, 0)

        for i in range(tp_size):
            states[i][f"layers.{layer_i}.attention.wqkv.weight"] = wqkvs[i]
            states[i][f"layers.{layer_i}.attention.wo.weight"] = wos[i].clone()
            states[i][f"layers.{layer_i}.feed_forward.w1.weight"] = w1s[i].clone()
            states[i][f"layers.{layer_i}.feed_forward.w3.weight"] = w3s[i].clone()
            states[i][f"layers.{layer_i}.feed_forward.w2.weight"] = w2s[i].clone()

    if embed_split_hidden:
        embed_concat_dim = 1
    else:
        embed_concat_dim = 0
    embeds = hf_state["model.tok_embeddings.weight"].chunk(tp_size, embed_concat_dim)
    outputs = hf_state["output.weight"].chunk(tp_size, 0)
    for i in range(tp_size):
        states[i]["norm.weight"] = hf_state["model.norm.weight"].clone()
        states[i]["tok_embeddings.weight"] = embeds[i].clone()
        states[i]["output.weight"] = outputs[i].clone()

    mlp_ratio = round((config.intermediate_size - 255) / config.hidden_size + 0.01, 2)
    if "rope_theta" in config.to_dict():
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
        adapt_hf=True,
        use_flash_attn=use_flash,
        rope_base=rope_base,
        num_chunks=1,
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
    print(f"Use Flash Attn: {args.use_flash}")
    print("---------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--src", type=str, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--tp_size", type=int, help="world_size of tensor parallel")
    parser.add_argument("--embed_split", action="store_true", help="embed_split_hidden of InternLM")
    parser.add_argument("--use_flash", action="store_true", help="use_flash_attn of InternLM")
    parser.add_argument("--safetensors", action="store_true", help="Whether the model is saved as safetensors")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    revert(args.src, args.tgt, args.tp_size, args.embed_split, args.use_flash, args.safetensors)
