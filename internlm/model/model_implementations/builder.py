from typing import List, Union

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import pipeline_parallel_sharding_wrapper
from internlm.model.model_implementations.registry import model_initializer
from internlm.model.model_implementations.transformers.base_model import (
    BaseTransformerModel,
)
from internlm.model.model_ops.modules.linear import (
    ParallelLinearWithCommExt,
    ScaleColumnParallelLinear,
)
from internlm.utils.common import get_current_device
from internlm.utils.lazy import LazyObject
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_fsdp, is_using_hf, is_using_isp

try:
    import transformer_engine.pytorch as te

    HAS_TE = True
except (ModuleNotFoundError, ImportError):
    HAS_TE = False


logger = get_logger(__file__)


def simple_swap(model, device):
    for submodule_name, submodule in model.named_modules():
        if isinstance(submodule, torch.nn.Linear):
            path_in_state_dict = submodule_name.split(".")
            current_module = model

            # traverse to leaf module
            leaf_path = path_in_state_dict[:-1]
            leaf_name = path_in_state_dict[-1]
            for child_name in leaf_path:
                current_module = getattr(current_module, child_name)

            # perform a swap
            old_leaf = getattr(current_module, leaf_name)
            new_leaf = te.Linear(old_leaf.in_features, old_leaf.out_features, old_leaf.bias is not None, device=device)
            with torch.no_grad():
                new_leaf.weight.copy_(old_leaf.weight)
                assert torch.equal(new_leaf.weight, old_leaf.weight)
                if old_leaf.bias is not None:
                    new_leaf.bias.copy_(old_leaf.bias)
                    assert torch.equal(new_leaf.bias, old_leaf.bias)

            setattr(current_module, leaf_name, new_leaf)


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

    kwargs["checkpoint"] = float(kwargs.get("checkpoint", False))
    kwargs["device"] = get_current_device()

    model_buidler = model_initializer.get_module(module_name=model_type)

    if (
        is_using_fsdp()
        and hasattr(gpc.config.model, "num_experts")
        and gpc.config.model.num_experts > 1
    ):
        kwargs["ep_group"] = gpc.get_group(ParallelMode.EXPERT)

    if not gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        kwargs["first"] = kwargs["last"] = True
        kwargs["start_layer_idx"] = 0
        kwargs["num_layers"] = num_layers
        model = model_buidler(**kwargs).to(kwargs["device"])
        setattr(model, "first_layer", 0)
        setattr(model, "last_layer", num_layers)
    else:
        model = pipeline_parallel_sharding_wrapper(num_layers, num_chunks, model_buidler, **kwargs)

    if not isinstance(model, BaseTransformerModel) and gpc.is_rank_for_log():
        logger.warning(
            f"To load/save huggingface ckpt, built-in model should inherited from {BaseTransformerModel.__name__}"
        )

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

    if HAS_TE and gpc.config.get("fp8", None) is not None:
        simple_swap(model=model, device=fsdp_init_method)

    return model
