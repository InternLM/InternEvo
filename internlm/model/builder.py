from typing import List, Union

from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.parallel_context import (
    IS_TENSOR_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
)
from internlm.core.parallel.shard import pipeline_parallel_sharding_wrapper
from internlm.model.registry import hf_config_initializer, model_initializer
from internlm.utils.common import get_current_device


def create_model(model_type, *args, **kwargs) -> Union[nn.Module, List[nn.Module]]:
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
        if "_FROM_HF" in model_type:
            hf_config_builder = hf_config_initializer.get_module(module_name=model_type)
            config = hf_config_builder(return_dict=False)
            model = model_buidler(*args, config).to(kwargs["device"])
        else:
            model = model_buidler(*args, **kwargs).to(kwargs["device"])
        setattr(model, "first_layer", 0)
        setattr(model, "last_layer", num_layers)
    else:
        model = pipeline_parallel_sharding_wrapper(num_layers, num_chunks, model_buidler, *args, **kwargs)

    if "_FROM_HF" in gpc.config.model_type:
        for module in model.modules():
            for param in module.parameters():
                if gpc.config.parallel["tensor"].get("mode", "mtp") == "isp":
                    setattr(param, IS_TENSOR_DATA_PARALLEL, True)
                else:
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)

    return model
