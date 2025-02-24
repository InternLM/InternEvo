#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import split_data_for_sequence_parallel
from internlm.data.utils import packed_data_normalizer, unpack_data
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_isp

logger = get_logger(__file__)


def get_non_moe_state_dict(full_state_dict):
    """
    Get the state dict of the non-moe layers
    """
    for key in list(full_state_dict.keys()):
        if "expert" in key and "moe_layer.gate" not in key:
            full_state_dict.pop(key)

    return full_state_dict


def get_model_topology(model):
    """
    Returns:
        {
            '{name}': {'dim': int}
        }
        where name is the name of the module, and all parameters under this module are
        concatenated along the dimension 'dim'.
    """
    topos = {}
    for name, module in model.named_modules():  # pylint: disable=W0612
        # TODO: If it does not meet these conditions, it is shared between various tp/dp, and it is necessary to assert
        # that they are consistent.
        # In order to be compatible with CI, this function will not be deleted for now.
        pass
    return topos


def process_load_info(load_info):
    load_content_str = ""
    load_ckpt_folder = load_info["path"]
    load_content = load_info["content"]
    if gpc.is_rank_for_log():
        logger.info(f"Try load_ckpt_folder: {load_ckpt_folder}")

    return load_content_str, load_ckpt_folder, load_content
