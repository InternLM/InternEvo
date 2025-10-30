#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import itertools

import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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


def init_fsdp_v1(model: FSDP, device: torch.device) -> FSDP:
    """
    Initialize Fully Sharded Data Parallel (FSDP) for the model.
    This function is needed to properly initialize FSDP when resuming from a checkpoint.
    It runs a forward pass with dummy inputs to ensure FSDP is fully initialized.

    References:
    https://github.com/pytorch/pytorch/issues/113496
    https://github.com/huggingface/transformers/pull/34032
    https://github.com/huggingface/transformers/issues/31892

    Args:
        model: The model to initialize with FSDP.
        device: The device to run the model on.

    Returns:
        The initialized FSDP model.
    """
    model.train()
    with torch.no_grad():
        # generate dummy packed sequence
        seq_len = gpc.config.data.seq_len * gpc.config.data.micro_bsz
        input_ids = [1] * seq_len
        label = input_ids[1:] + [-100]
        cu_seqlens = list(range(0, seq_len + gpc.config.data.seq_len, gpc.config.data.seq_len))

        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        label = torch.tensor(label, device=device).unsqueeze(0)
        indexes = torch.tensor(
            list(itertools.chain(*[np.arange(l2 - l1) for l1, l2 in zip(cu_seqlens[:-1], cu_seqlens[1:])])),
            device=device,
        ).unsqueeze(0)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32).unsqueeze(0)

        data = {
            "input_ids": input_ids,
            "cu_seqlens": cu_seqlens,
            "indexes": indexes,
            "max_seqlen": seq_len,
        }

        data_fns = []

        # default data process function
        if gpc.config.data.use_packed_dataset:
            data_fns.append(packed_data_normalizer)
        else:
            data_fns.append(unpack_data)

        # support sequence parallel for isp
        if is_using_isp():
            data_fns.append(split_data_for_sequence_parallel)

        # generate dummy_input
        _data, _label = data, label
        for fn in data_fns:
            _data, _label = fn(_data, _label)
        dummy_input = _data

        # run a forward pass with dummy_input to initialize FSDP
        _ = model(**dummy_input)
    return model
