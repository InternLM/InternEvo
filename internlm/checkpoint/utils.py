#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


def get_shard_state_dict(shard_model):
    """
    Only used for FSDP module saving.
    It's a warper of model.state_dict() and with the context of 'FSDP.state_dict_type', the sharded parameter
    (saved as model.flat_param_xx in sharded FSDP module) will be gathered at every gpu.
    'offload_to_cpu' means that the model states are to be offloaded to cpu chunk by chunk, avoiding OOM in gpu

    """

    # FSDP model can only save with sharded shape SHARDED_STATE_DICT when set use_orig_params=True
    with FSDP.state_dict_type(shard_model, StateDictType.SHARDED_STATE_DICT):
        shard_states = shard_model.state_dict()

    return shard_states


def get_non_moe_state_dict(full_state_dict):
    """
    Get the state dict of the non-moe layers
    """
    for key in list(full_state_dict.keys()):
        if "expert" in key and "moe_layer.gate" not in key:
            full_state_dict.pop(key)

    return full_state_dict


def load_shard_state_dict(shard_model, shard_state, **kwargs):
    """
    Only used for FSDP module loading.

    """

    with FSDP.state_dict_type(shard_model, StateDictType.SHARDED_STATE_DICT):
        missing_k, unexpected_keys = shard_model.load_state_dict(shard_state, kwargs)

    return (missing_k, unexpected_keys)


def get_model_topology(model):
    """
    Returns:
        {
            '{name}': {'dim': int}
        }
        where name is the name of the module, and all parameters under this module are
        concatenated along the dimension 'dim'.
    """

    from flash_attn.modules.embedding import VocabParallelEmbedding

    topos = {}
    for name, module in model.named_modules():
        # If it does not meet these conditions, it is shared between various tp/dp, and it is necessary to assert
        # that they are consistent.
        if isinstance(module, VocabParallelEmbedding):
            topos[name] = {"dim": 0}
    return topos


def process_load_info(load_info):
    load_content_str = ""
    load_ckpt_folder = load_info["path"]
    load_content = load_info["content"]
    if gpc.is_rank_for_log():
        logger.info(f"Try load_ckpt_folder: {load_ckpt_folder}")

    return load_content_str, load_ckpt_folder, load_content
