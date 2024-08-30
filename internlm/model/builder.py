from typing import List, Union

from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import pipeline_parallel_sharding_wrapper
from internlm.model.base_model import BaseModel
from internlm.model.registry import model_initializer
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


def create_model(model_type) -> Union[nn.Module, List[nn.Module]]:

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

    assert isinstance(model, BaseModel), f"built-in model should inherited from {BaseModel.__name__}"

    return model
