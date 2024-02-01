#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.distributed as dist
from torch import nn

from internlm.core.context import (
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_DATA_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import NaiveAMPModel
from internlm.model.utils import try_import_RMSNorm
from internlm.solver.pipeline_utils import partition_uniform

RMSNorm = try_import_RMSNorm()


def is_using_isp():
    return isinstance(gpc.config.parallel["tensor"], dict) and gpc.config.parallel["tensor"].get("mode", "mtp") == "isp"


def is_replica_zero_parallel_parameter(p):
    return hasattr(p, IS_REPLICA_ZERO_PARALLEL) and getattr(p, IS_REPLICA_ZERO_PARALLEL)


def is_tensor_data_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.TENSOR)
        and is_using_isp()
        and hasattr(p, IS_TENSOR_DATA_PARALLEL)
        and getattr(p, IS_TENSOR_DATA_PARALLEL)
    )


def is_tensor_zero_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.TENSOR)
        and not is_using_isp()
        and hasattr(p, IS_TENSOR_ZERO_PARALLEL)
        and getattr(p, IS_TENSOR_ZERO_PARALLEL)
    )


def is_weight_zero_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.WEIGHT)
        and is_using_isp()
        and hasattr(p, IS_WEIGHT_ZERO_PARALLEL)
        and getattr(p, IS_WEIGHT_ZERO_PARALLEL)
    )


def is_tensor_expert_data_parallel_parameter(p):
    return (
        gpc.is_initialized(ParallelMode.TENSOR)
        and hasattr(p, IS_TENSOR_EXPERT_DATA_PARALLEL)
        and getattr(p, IS_TENSOR_EXPERT_DATA_PARALLEL)
    )


def sync_model_param(model):
    r"""Make sure data parameters are consistent during Data Parallel Mode.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """

    sync_moe_param = gpc.is_using_parallel_mode(ParallelMode.EXPERT_DATA)
    sync_parallel_mode = ParallelMode.WEIGHT_DATA if is_using_isp() else ParallelMode.DATA
    for param in model.parameters():
        if sync_moe_param and getattr(param, "is_expert", False):
            ranks = gpc.get_ranks_in_group(ParallelMode.EXPERT_DATA)
            dist.broadcast(param, src=ranks[0], group=gpc.get_group(ParallelMode.EXPERT_DATA))
        else:
            ranks = gpc.get_ranks_in_group(sync_parallel_mode)
            dist.broadcast(param, src=ranks[0], group=gpc.get_group(sync_parallel_mode))


def sync_model_replica_param_group(model):
    r"""This function is changed from colossalai, which is ``sync_model_param``.

    We modified this function to make sure it only sync IS_REPLICA_ZERO_PARALLEL parameters in tp or wp process group.
    This function is used to make sure parameters that are not splitted are the same across each rank.
    For example, parameters like RMSNorm, LayerNorm...

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """

    parallel_mode = ParallelMode.WEIGHT if is_using_isp() else ParallelMode.TENSOR
    if gpc.is_using_parallel_mode(parallel_mode):
        for param in model.parameters():
            if is_replica_zero_parallel_parameter(param):
                ranks = gpc.get_ranks_in_group(parallel_mode)
                dist.broadcast(param, src=ranks[0], group=gpc.get_group(parallel_mode))


def get_parallel_log_file_name():
    if gpc.is_rank_for_log():
        fn_prefix = "main_"  # Indicates a rank with more output information
    else:
        fn_prefix = ""

    log_file_name = (
        f"{fn_prefix}dp={gpc.get_local_rank(ParallelMode.DATA)}_"
        f"tp={gpc.get_local_rank(ParallelMode.TENSOR)}_pp={gpc.get_local_rank(ParallelMode.PIPELINE)}"
    )
    return log_file_name


def set_model_params_layer_name(model):
    r"""Set the layer name as an attribute of the model parameters.
    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    all_parts = partition_uniform(gpc.config.model.num_layers, pipeline_size, gpc.config.model.num_chunks)
    parts = all_parts[pipeline_rank]

    if not isinstance(model, nn.ModuleList):
        model = [model]

    for chunk_idx, _chunk in enumerate(model):
        if isinstance(_chunk, NaiveAMPModel):
            _chunk = _chunk.model
        chunk_start = parts[chunk_idx][0]
        # Create a unique layer name based on the block's class name and index
        for _, children in _chunk.named_children():
            if isinstance(children, nn.ModuleList):
                for idx, block in enumerate(children):
                    for param_name, param in block.named_parameters():
                        layer_name = f"{block.__class__.__name__}Block{idx + chunk_start}"
                        layer_param_name = f"{layer_name}-{param_name}"
                        param.__setattr__("layer_name", layer_name)
                        param.__setattr__("param_name", layer_param_name)
            else:
                for param_name, param in children.named_parameters():
                    layer_name = f"{children.__class__.__name__}"
                    layer_param_name = f"{layer_name}-{param_name}"
                    param.__setattr__("layer_name", layer_name)
                    param.__setattr__("param_name", f"{layer_name}-{param_name}")
