import torch
import torch.nn as nn
import torch.distributed as dist

def _gather(input_, dim=-1, process_group=None):
    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # all gather
    input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=process_group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _split(input_, dim=-1, process_group=None):
    # skip if only one rank involved
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = dist.get_rank(process_group)
    output = tensor_list[rank].clone().contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def forward(ctx, input_, dim, process_group):
        ctx.process_group = process_group
        ctx.dim = dim
        return _gather(input_, dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.dim, ctx.process_group), None, None


def gather_forward_split_backward(input_, dim, process_group):
    return _GatherForwardSplitBackward.apply(input_, dim, process_group)

def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)