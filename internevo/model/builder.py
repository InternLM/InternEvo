from typing import List, Union

import torch
from torch import nn

from internevo.core.context import ParallelMode
from internevo.core.context import global_context as gpc
from internevo.core.parallel.shard import pipeline_parallel_sharding_wrapper
from internevo.model.base_model import BaseModel
from internevo.model.modules.linear import (
    ParallelLinearWithCommExt,
    ScaleColumnParallelLinear,
)
from internevo.model.registry import model_initializer
from internevo.utils.common import get_current_device
from internevo.utils.lazy import LazyObject
from internevo.utils.logger import get_logger
from internevo.utils.parallel import is_using_fsdp, is_using_hf, is_using_isp

logger = get_logger(__file__)


def create_model() -> Union[nn.Module, List[nn.Module]]:
    if is_using_hf():
        model = create_model_hf(hf=gpc.config.hf)
    else:
        model = create_model_builtin(model_type=gpc.config.model_type)
    return model


def create_model_builtin(model_type) -> Union[nn.Module, List[nn.Module]]:

    kwargs = dict(gpc.config.model)

    num_layers = kwargs.pop("num_layers")
    num_chunks = kwargs.pop("num_chunks", 1)

    # TODO: fix use_flash_attn parameter config
    kwargs.pop("use_flash_attn", False)
    kwargs.pop("apply_post_layer_norm")
    kwargs.pop("embed_split_hidden", True)

    kwargs["checkpoint"] = float(kwargs.get("checkpoint", False))
    kwargs["device"] = get_current_device()

    model_buidler = model_initializer.get_module(module_name=model_type)

    if not gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        kwargs["first"] = kwargs["last"] = True
        kwargs["start_layer_idx"] = 0
        kwargs["num_layers"] = num_layers
        model = model_buidler(**kwargs).to(kwargs["device"])
        setattr(model, "first_layer", 0)
        setattr(model, "last_layer", num_layers)
    else:
        model = pipeline_parallel_sharding_wrapper(num_layers, num_chunks, model_buidler, **kwargs)

    if not isinstance(model, BaseModel) and gpc.is_rank_for_log():
        logger.warning(f"To load/save huggingface ckpt, built-in model should inherited from {BaseModel.__name__}")

    return model


def create_model_hf(hf: dict) -> nn.Module:
    cfg = LazyObject(hf.cfg, hf.cfg_cls)
    cfg = cfg.build()
    mod = LazyObject(hf.mod, hf.mod_cls)
    mod = mod.build()

    assert is_using_fsdp(), "Curently HF models can only train with FSDP."

    fsdp_init_method = gpc.config.parallel.fsdp.get("init_method", "cuda")
    if fsdp_init_method == "meta":
        with torch.device("meta"):
            model = mod(cfg(**hf.cfg_extra_kwargs))
    elif fsdp_init_method == "cuda":
        # TODO: does HuggingFace models support directly initialized on cuda?
        model = mod(cfg(**hf.cfg_extra_kwargs)).to(get_current_device())
    elif fsdp_init_method == "cpu":
        model = mod(cfg(**hf.cfg_extra_kwargs))
    else:
        raise ValueError(f"Unsupported fsdp init_method: {fsdp_init_method}")

    def traverse(module):
        for name, child in module.named_children():
            if (
                isinstance(child, nn.Linear)
                and not isinstance(child, ParallelLinearWithCommExt)
                and child.weight.shape == (gpc.config.VOCAB_SIZE, gpc.config.HIDDEN_SIZE)
            ):
                child_new = ScaleColumnParallelLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                setattr(module, name, child_new)
            else:
                traverse(child)

    # Do hack: lm_head or output layer should be replaced with ScaleColumnParallelLinear,
    # to get ISP fwd gather / bwd split work normally.
    if is_using_isp():
        # traverse model might be slower than replacement module by name directly
        if getattr(model, "lm_head", None) is not None:
            lm_head = model.lm_head
            lm_head_new = ScaleColumnParallelLinear(
                in_features=lm_head.in_features,
                out_features=lm_head.out_features,
                bias=lm_head.bias is not None,
                device=lm_head.weight.device,
                dtype=lm_head.weight.dtype,
            )
            setattr(model, "lm_head", lm_head_new)
        elif getattr(model, "output", None) is not None:
            output = model.output
            output_new = ScaleColumnParallelLinear(
                in_features=output.in_features,
                out_features=output.out_features,
                bias=output.bias is not None,
                device=output.weight.device,
                dtype=output.weight.dtype,
            )
            setattr(model, "output", output_new)
        else:
            traverse(model)

    return model
