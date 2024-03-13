#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from internlm.core.context import global_context as gpc


class AsyncCommHandle(ABC):
    """A interface for asynchronous communication handles."""
    @abstractmethod
    def wait(self) -> None:
        """ wait asynchronous communication to complete. """


class DummyAsyncCommHandle(AsyncCommHandle):
    """A fake communication handle used to maintain consistency in code writing"""
    def wait(self) -> None:
        pass

DUMMY_HANDLE_CONST = DummyAsyncCommHandle()
"""
A shared dummy async communication handeles used to maintain consistency in code writing.
"""

# Raw operation, does not support autograd, but does support async
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=process_group, async_op=async_op)
    return input_, handle


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = reduce_scatter_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group)
        return grad_input, None


# Supports autograd, but does not support async
reduce_scatter = ReduceScatterFunc.apply


class AllReduceFunc(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_reduce_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        _ = ctx  # avoid lint warning W0613
        return grad_output, None


# Supports autograd, but does not support async
all_reduce = AllReduceFunc.apply


def _split(input_, parallel_mode, dim=-1):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # Split along last dimension.
    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = gpc.get_local_rank(parallel_mode)
    output = tensor_list[rank].contiguous()
    output = output.detach().clone()

    return output


def _gather(input_, parallel_mode, dim=-1):
    # skip if only one rank involved
    world_size = gpc.get_world_size(parallel_mode)
    if world_size == 1:
        return input_

    # all gather
    rank = gpc.get_local_rank(parallel_mode)
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    group = gpc.get_cpu_group(parallel_mode) if input_.device.type == "cpu" else gpc.get_group(parallel_mode)
    dist.all_gather(tensor_list, input_, group=group)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(input_):
        return _gather(input_, parallel_mode=None)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _gather(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.mode, ctx.dim), None, None


def gather_forward_split_backward(input_, parallel_mode, dim):
    return _GatherForwardSplitBackward.apply(input_, parallel_mode, dim)


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_: input matrix.
        parallel_mode: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(input_):
        return _split(input_, parallel_mode=None)

    @staticmethod
    def forward(ctx, input_, parallel_mode, dim):
        ctx.mode = parallel_mode
        ctx.dim = dim
        return _split(input_, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output, ctx.mode, ctx.dim), None, None


def split_forward_gather_backward(input_, parallel_mode, dim):
    return _SplitForwardGatherBackward.apply(input_, parallel_mode, dim)


def all_gather_raw(
    input_: Tensor,
    process_group: ProcessGroup,
    async_op: bool = False,
    gather_dim: int = 0,
    memory_pool_allocator: Callable = None,
):
    world_size = dist.get_world_size(process_group)
    if world_size <= 1:
        return input_, None

    if memory_pool_allocator is not None:
        output = memory_pool_allocator()
    else:
        shape = list(input_.shape)
        shape[gather_dim] = shape[gather_dim] * world_size
        output = torch.empty(shape, dtype=input_.dtype, device=input_.device)

    handle = dist.all_gather_into_tensor(output, input_.contiguous(), group=process_group, async_op=async_op)
    return output, handle


def reduce_scatter_raw(
    input_: Tensor,
    process_group: ProcessGroup,
    op=dist.ReduceOp.SUM,
    async_op: bool = False,
    memory_pool_allocator: Callable = None,
):
    world_size = dist.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0

    if world_size <= 1:
        return input_, None

    if memory_pool_allocator is not None:
        size = (input_.shape[0] // world_size, *input_.shape[1:])
        output = memory_pool_allocator(size)
    else:
        output = torch.empty(
            input_.shape[0] // world_size,
            *input_.shape[1:],
            dtype=input_.dtype,
            device=input_.device,
        ).contiguous()

    handle = dist.reduce_scatter_tensor(output, input_.contiguous(), op=op, group=process_group, async_op=async_op)
    return output, handle