# Copyright (c) InternLM. All rights reserved.
import os

import torch

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger
from internlm.utils.storage_manager import get_fns, llm_load

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def load_llama_pretrained_weights(folder, model):
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = []
    for fn in fns:
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(os.path.join(folder, fn))

    if len(model_fns) == 0:
        model_fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".pth") or fn.endswith(".pt")]

    if len(model_fns) == 0:
        raise FileNotFoundError(f"No checkpoint file found in {folder}")

    model_fns.sort()

    old_tp = len(model_fns)
    cur_tp = gpc.get_world_size(ParallelMode.TENSOR)
    # If the two tp are inconsistent, you need to consider the merge before splitting
    if old_tp != cur_tp:
        raise RuntimeError(
            f"Your current tp is `{cur_tp}`, but the tp in folder:`{folder}` is `{old_tp}`, use `` to convert first"
        )

    states = llm_load(model_fns[gpc.get_local_rank(ParallelMode.TENSOR)], map_location="cpu")

    current_states = {}
    for idx, i in enumerate(range(model.first_layer, model.last_layer)):
        for name in list(states.keys()):
            if f".{i}." in name:
                current_states[name.replace(f".{i}.", f".{idx}.")] = states.pop(name)

    model_state_keys = set(list(model.state_dict().keys()))

    if "tok_embeddings.weight" in model_state_keys:
        current_states["tok_embeddings.weight"] = states["tok_embeddings.weight"]
        assert model.first_layer == 0, f"Expect model.NaiveAMPModel to be 0, but got {model.first_layer}"
    if "output.weight" in model_state_keys:
        current_states["norm.weight"] = states["norm.weight"]
        current_states["output.weight"] = states["output.weight"]
    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )

    del states
    del current_states
    internlm_accelerator.empty_cache()


def load_hf_llama_pretrained_weights(folder, model):
    """NOTE: when loading huggingface's llama pretrained weights, you should set `adapt_hf=True` in your config."""
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = [os.path.join(folder, fn) for fn in fns if fn.endswith(".bin") and fn.startswith("pytorch_model")]
    model_fns.sort()

    states = {}

    for model_fn in model_fns:
        states.update(llm_load(model_fn, map_location="cpu"))

    deep_split = getattr(model, "deep_split", False)
    if deep_split:
        print("using deep split when loading pretrained weights!")

    current_states = {}
    for idx, i in enumerate(range(model.first_layer, model.last_layer)):
        if gpc.config.model_type == "LLAMA2":
            if deep_split:
                layer_ids = i // 2
            else:
                layer_ids = i

            if not deep_split or (i + 2) % 2 == 0:
                states[f"layers.{i}.attention.wq.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.q_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wk.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.k_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wv.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.v_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention.wo.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.self_attn.o_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=1,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.attention_norm.weight"] = states.pop(
                    f"model.layers.{layer_ids}.input_layernorm.weight"
                )

            if not deep_split or (i + 2) % 2 == 1:
                states[f"layers.{i}.feed_forward.w1.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.gate_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.feed_forward.w3.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.up_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=0,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]
                states[f"layers.{i}.feed_forward.w2.weight"] = torch.chunk(
                    states.pop(f"model.layers.{layer_ids}.mlp.down_proj.weight"),
                    gpc.get_world_size(ParallelMode.TENSOR),
                    dim=1,
                )[gpc.get_local_rank(ParallelMode.TENSOR)]

                states[f"layers.{i}.ffn_norm.weight"] = states.pop(
                    f"model.layers.{layer_ids}.post_attention_layernorm.weight"
                )

            if f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq" in states:
                states.pop(f"model.layers.{layer_ids}.self_attn.rotary_emb.inv_freq")

        if gpc.config.model_type in ("LLAMA2",):
            w2 = states.pop(f"layers.{i}.feed_forward.w2.weight")
            w3 = states.pop(f"layers.{i}.feed_forward.w3.weight")
            states[f"layers.{i}.feed_forward.w2.weight"] = w3
            states[f"layers.{i}.feed_forward.w3.weight"] = w2

        for name in list(states.keys()):
            if name.startswith(f"layers.{i}"):
                current_states[name.replace(f".{i}.", f".{idx}.")] = states.pop(name)

    model_state_keys = set(list(model.state_dict().keys()))

    if "tok_embeddings.weight" in model_state_keys or "tok_embeddings.word_embeddings.weight" in model_state_keys:
        if gpc.config.model.get("embed_split_hidden", True):
            current_states["tok_embeddings.weight"] = torch.chunk(
                states["model.embed_tokens.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=1
            )[gpc.get_local_rank(ParallelMode.TENSOR)]
        else:
            current_states["tok_embeddings.word_embeddings.weight"] = torch.chunk(
                states["model.embed_tokens.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=1
            )[gpc.get_local_rank(ParallelMode.TENSOR)]
        assert model.first_layer == 0, f"Expect model.first_layer to be 0, but got {model.first_layer}"

    if "output.weight" in model_state_keys:
        current_states["norm.weight"] = states["model.norm.weight"]
        current_states["output.weight"] = torch.chunk(
            states["lm_head.weight"], gpc.get_world_size(ParallelMode.TENSOR), dim=0
        )[gpc.get_local_rank(ParallelMode.TENSOR)]

    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )
    internlm_accelerator.empty_cache()


LOAD_FUNC_DICT = {
    "llama": load_llama_pretrained_weights,
    "hf_llama": load_hf_llama_pretrained_weights,
}
