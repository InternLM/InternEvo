"""
shard strategies for parallel
"""

from typing import Callable

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


# The head layer in ISP mode is actually a special case,
# and we would prefer a unified segmentation and communication logic.
def get_tensor_split_parallel_mode(is_head: bool = False) -> ParallelMode:
    tp_mode = gpc.config.parallel.tensor.mode

    if tp_mode == "isp" and is_head is False:
        return ParallelMode.WEIGHT
    else:
        return ParallelMode.TENSOR


def get_head_parallel_mode() -> ParallelMode:
    return ParallelMode.TENSOR


def get_parallel_strategies_split_mode(linear_name: str) -> str:
    tp_mode = gpc.config.parallel.tensor.mode

    if linear_name in ("head", "output"):
        return "head"
    elif linear_name in ("wqkv", "wq", "wk", "wv", "wkv", "w1", "w3", "w13"):
        return "column"
    elif linear_name in ("wo", "out_proj", "w2") and tp_mode == "isp":
        return "column"
    elif linear_name in ("wo", "out_proj", "w2"):
        return "row"
    else:
        return "unknown"


def partition_uniform(num_items: int, pipeline_parallel_size: int, num_chunks: int):
    assert (
        num_items % num_chunks == 0
    ), "Layer length should be divided by the number of chunks, otherwise parameter method is recomended"

    parts = [[] for _ in range(pipeline_parallel_size)]
    partition_items = num_items // num_chunks
    for idx in range(num_chunks):
        base_idx = idx * partition_items
        chunk_size = partition_items // pipeline_parallel_size
        left = pipeline_parallel_size - partition_items % pipeline_parallel_size
        if chunk_size == 0:
            raise ValueError("Some nodes in Pipeline have no requests")

        for p in range(pipeline_parallel_size):
            st = base_idx
            base_idx += chunk_size + (p >= left)
            parts[p].append((st, base_idx))

    indexes = []
    for _parts in parts:
        for s, e in _parts:
            indexes.extend(list(range(s, e)))
    assert len(indexes) == len(set(indexes)), indexes  # should have no duplicates
    assert set(indexes) == set(list(range(num_items))), (indexes, num_items)  # should have the same indexes as expected
    return parts


def pipeline_parallel_sharding_wrapper(
    num_layers: int, num_chunks: int, model_builder: Callable, device: torch.device, **kwargs
):
    """
    build generic model 1d

    Args:
        num_layers (int): The number of layer.
        num_chunks (int): The number of partitions in pipeline parallel.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    all_parts = partition_uniform(num_layers, pipeline_size, num_chunks)
    parts = all_parts[pipeline_rank]

    if gpc.is_rank_for_log():
        logger.info("The layer sharding is %r.", all_parts)

    models = []

    use_checkpoint = kwargs["checkpoint"]
    for part_idx, (start, end) in enumerate(parts):
        kwargs["num_layers"] = end - start
        kwargs["first"] = start == 0
        # If there is no content in the final layer, assign the last layer.
        kwargs["last"] = end == num_layers and len(all_parts[-1]) != 0
        kwargs["device"] = device
        kwargs["start_layer_idx"] = start
        if hasattr(gpc.config.parallel.pipeline, "use_zeropp") and gpc.config.parallel.pipeline.use_zeropp:
            if part_idx == num_chunks - 1 and gpc.is_pipeline_last_stage(ignore_virtual=True):
                kwargs["checkpoint"] = False
            else:
                kwargs["checkpoint"] = use_checkpoint
        chunk = model_builder(**kwargs).to(device)
        setattr(chunk, "first_layer", start)
        setattr(chunk, "last_layer", end)

        models.append(chunk)

    torch.distributed.barrier()

    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)

    return model
