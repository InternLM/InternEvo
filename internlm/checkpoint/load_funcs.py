# Copyright (c) InternLM. All rights reserved.
import os

import torch

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import partition_uniform
from internlm.utils.logger import get_logger
from internlm.utils.storage_manager import get_fns, llm_load
from transformers import AutoModelForCausalLM

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


def load_internlm_with_dynamic_parallel_size(folder, model):

    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = get_fns(folder)
    model_fns = []
    for fn in fns:
        # filter with `_t` is for avoiding conflict with model_config.py
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(fn)

    old_tp, old_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split("_")
        old_tp = max(old_tp, int(tp[2:]) + 1)
        old_pp = max(old_pp, int(pp[2:]) + 1)

    assert old_tp > 0 and old_pp > 0, f"ckpt with tp:{old_tp} and pp:{old_pp} is illegal"

    tp = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    assert old_tp % tp == 0 or tp % old_tp == 0, (
        f"Expected TP size in loaded checkpoint to be fit with TP size in current config, but got {old_tp} in "
        f"checkpoint and {tp} in current config"
    )

    correspond_tps = []

    if old_tp <= tp:
        correspond_tps.append(tp_rank // (tp // old_tp))
        ratio = tp // old_tp
        rank = tp_rank % ratio
    else:
        for i in range(old_tp // tp):
            correspond_tps.append(tp_rank * (old_tp // tp) + i)
        rank = 0
        ratio = 1

    current_states = {}

    pp = gpc.get_world_size(ParallelMode.PIPELINE)

    assert gpc.config.model.num_chunks == 1, "May cause future collisions, ignore this if necessary"

    old_pp_partition = partition_uniform(gpc.config.model.num_layers, old_pp, 1)

    for idx, parts in enumerate(old_pp_partition):
        start, end = parts[0]
        if model.last_layer <= start or model.first_layer >= end:
            continue

        tmp_states = {}

        for correspond_tp in correspond_tps:
            model_name = f"model_tp{correspond_tp}_pp{idx}.pt"
            states = llm_load(os.path.join(folder, model_name), map_location="cpu")
            for i in range(start, end):
                if i >= model.last_layer:
                    break
                if i < model.first_layer:
                    continue
                for name in list(states.keys()):
                    if f".{i-start}." in name:
                        to_name = name.replace(f".{i-start}.", f".{i-model.first_layer}.")
                        if "norm" in name:
                            tmp_states[to_name] = [states.pop(name)]
                        elif any(x in name for x in ("out_proj", "w2")):
                            if "bias" not in name:
                                tmp_states[to_name] = tmp_states.get(to_name, [])
                                tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=-1)[rank])
                            else:
                                tmp_states[to_name] = [states.pop(name)]
                        elif any(x in name for x in ("w1", "w3")):
                            tmp_states[to_name] = tmp_states.get(to_name, [])
                            tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                        elif any(x in name for x in ("Wqkv",)):
                            tmp_states[to_name] = tmp_states.get(to_name, [])
                            _wqkv = states.pop(name).chunk(3, dim=0)
                            _wq_splits = _wqkv[0].chunk(ratio, dim=0)
                            _wk_splits = _wqkv[1].chunk(ratio, dim=0)
                            _wv_splits = _wqkv[2].chunk(ratio, dim=0)
                            new_wqkv = torch.concat([_wq_splits[rank], _wk_splits[rank], _wv_splits[rank]], dim=0)
                            tmp_states[to_name].append(new_wqkv)
                        else:
                            raise KeyError(f"Unknown key {name}.")

            if "embedding.weight" in states and model.first_layer == 0:
                tmp_states["embedding.weight"] = tmp_states.get("embedding.weight", [])
                tmp_states["embedding.weight"].append(states["embedding.weight"].chunk(ratio, dim=1)[rank])
            if "head.weight" in states and model.last_layer == gpc.config.model.num_layers:
                tmp_states["norm.weight"] = [states["norm.weight"]]
                tmp_states["head.weight"] = tmp_states.get("head.weight", [])
                tmp_states["head.weight"].append(states["head.weight"].chunk(ratio, dim=0)[rank])

            states = {}

        for name in list(tmp_states.keys()):
            data = tmp_states.pop(name)
            if len(data) == 1:
                current_states[name] = data[0]
            else:
                current_states[name] = torch.concat(
                    data, dim=1 if name == "embedding.weight" or any(x in name for x in ("out_proj", "w2")) else 0
                )

    missing_keys, unexpected_keys = model.load_state_dict(current_states, strict=False)

    if gpc.get_local_rank(ParallelMode.DATA) == 0:
        pp_rank = 0 if not gpc.is_initialized(ParallelMode.PIPELINE) else gpc.get_local_rank(ParallelMode.PIPELINE)
        logger.info(
            f"Missing keys:{missing_keys}, unexpected keys:{unexpected_keys} in "
            f"tp:{gpc.get_local_rank(ParallelMode.TENSOR)}, pp:{pp_rank}"
        )


def load_hf_model_pretrained_weights(folder, model):
    """NOTE: when loading huggingface's model pretrained weights, you should set `adapt_hf=True` in your config."""
    assert folder is not None, "Please specify the folder of the pretrained model"
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    pretrained_model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True)
    model.load_state_dict(pretrained_model.state_dict(), strict=False)

    if gpc.is_rank_for_log():
        logger.info("Pretrained weights loaded successfully")


LOAD_FUNC_DICT = {
    "llama": load_llama_pretrained_weights,
    "hf_llama": load_hf_llama_pretrained_weights,
    "internlm_test": load_internlm_with_dynamic_parallel_size,
    "hf_model": load_hf_model_pretrained_weights,
}
