from typing import List, Union

from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import pipeline_parallel_sharding_wrapper
from internlm.model.registry import model_initializer
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
        if "_FROM_HF" in model_type:  # TODO: here need to decide which model config to choose
            hf_model_conf_map = {
                "INTERNLM_FROM_HF": ("huggingface_model.internlm_model.configuration_internlm", "InternLMConfig"),
                "INTERNLM2_FROM_HF": ("huggingface_model.internlm2_model.configuration_internlm2", "InternLM2Config"),
            }
            if model_type not in hf_model_conf_map:
                raise ValueError(f"Unknown model type: {model_type}")
            config_module_name, config_class_name = hf_model_conf_map[model_type]
            config_class = import_class_from_module(config_module_name, config_class_name)
            config = config_class()
            model = model_buidler(*args, config).to(kwargs["device"])
        else:
            model = model_buidler(*args, **kwargs).to(kwargs["device"])
        setattr(model, "first_layer", 0)
        setattr(model, "last_layer", num_layers)
    else:
        model = pipeline_parallel_sharding_wrapper(num_layers, num_chunks, model_buidler, *args, **kwargs)

    return model

def import_class_from_module(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)
