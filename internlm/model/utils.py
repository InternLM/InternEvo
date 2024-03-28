#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

internlm_accelerator = get_accelerator()

custom_bwd = internlm_accelerator.return_custom_bwd()
custom_fwd = internlm_accelerator.return_custom_fwd()

logger = get_logger(__file__)


# Raw operation, does not support autograd, but does support async
def all_reduce_raw(input_: Tensor, process_group: ProcessGroup, async_op: bool = False):
    input_ = input_.contiguous()
    handle = torch.distributed.all_reduce(input_, group=process_group, async_op=async_op)
    return input_, handle


class ReduceScatterFunc(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_: Tensor, process_group: ProcessGroup, reduce_dim: int = 0) -> Tensor:
        ctx.process_group = process_group
        ctx.reduce_dim = reduce_dim
        output, _ = reduce_scatter_raw(input_, process_group, reduce_dim=reduce_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        gather_dim = ctx.reduce_dim
        grad_input, _ = all_gather_raw(grad_output, ctx.process_group, gather_dim=gather_dim)
        return grad_input, None, None


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
    reduce_dim: int = 0,
    memory_pool_allocator: Callable = None,
):
    world_size = dist.get_world_size(process_group)
    assert input_.shape[reduce_dim] % world_size == 0

    if world_size <= 1:
        return input_, None

    shape_list = list(input_.shape)
    shape_list[reduce_dim] = shape_list[reduce_dim] // world_size

    if memory_pool_allocator is not None:
        output = memory_pool_allocator(tuple(shape_list))
    else:
        output = torch.empty(
            shape_list,
            dtype=input_.dtype,
            device=input_.device,
        ).contiguous()

    handle = dist.reduce_scatter_tensor(output, input_.contiguous(), op=op, group=process_group, async_op=async_op)
    return output, handle


def linear_bias_wgrad_torch(my_input, grad_output, has_d_bias):
    assert my_input.dtype == grad_output.dtype
    grad_weight = torch.matmul(grad_output.t(), my_input)
    grad_bias = grad_output.sum(dim=0) if has_d_bias else None
    return grad_weight, grad_bias


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py
class FusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for tensor parallel in flash-attn implementation."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        weight,
        bias,
        return_residual=False,
        process_group=None,
        sequence_parallel=True,
        gather_dim=0,
        is_using_cuda: bool = True,
    ):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel
        ctx.gather_dim = gather_dim
        ctx.is_using_cuda = is_using_cuda

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True, gather_dim=gather_dim)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel and handle_x is not None:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel
        gather_dim = ctx.gather_dim

        if gpc.config.model.use_flash_attn:
            import fused_dense_lib as fused_dense_cuda

        if gpc.config.model.use_flash_attn and ctx.is_using_cuda:
            backward_func = fused_dense_cuda.linear_bias_wgrad
        else:
            backward_func = linear_bias_wgrad_torch

        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
            if process_group is not None and sequence_parallel:
                total_x, handle_x = all_gather_raw(x, process_group, async_op=True, gather_dim=gather_dim)
            else:
                total_x = x
        else:
            (weight,) = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, weight.t())
            else:
                grad_input = torch.addmm(
                    grad_input.reshape(batch_dim, grad_input.shape[-1]),
                    grad_output,
                    weight,
                )
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                if sequence_parallel:
                    grad_input, handle_grad_input = reduce_scatter_raw(
                        grad_input, process_group, async_op=True, reduce_dim=1
                    )
                else:
                    grad_input, handle_grad_input = all_reduce_raw(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if process_group is not None and sequence_parallel and handle_x is not None:
                handle_x.wait()
            grad_weight, grad_bias = backward_func(
                total_x.reshape(batch_dim, total_x.shape[-1]),
                grad_output,
                ctx.needs_input_grad[2],
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0] and handle_grad_input is not None:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class MegatronFusedDenseFunc(torch.autograd.Function):
    """
    FusedDenseFunc for tensor parallel in megatron implementation.
    The diffenrence between the implementation of flash-attn and megatron is that the total_x could be
    saved for backward in megatron, so that the all-gather in backward is ommited.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        weight,
        bias,
        return_residual=False,
        process_group=None,
        sequence_parallel=True,
        gather_dim=0,
        is_using_cuda: bool = True,
    ):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.process_group = process_group
        ctx.sequence_parallel = sequence_parallel
        ctx.is_using_cuda = is_using_cuda

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()
        if process_group is not None and sequence_parallel:
            # We want to kick off the all_gather early, before weight dtype conversion
            total_x, handle_x = all_gather_raw(x, process_group, async_op=True, gather_dim=gather_dim)
        else:
            total_x = x

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()
        if process_group is not None and sequence_parallel and handle_x is not None:
            handle_x.wait()
        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")
        output = F.linear(total_x, weight, bias)
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(total_x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()
        process_group = ctx.process_group
        sequence_parallel = ctx.sequence_parallel

        if gpc.config.model.use_flash_attn:
            import fused_dense_lib as fused_dense_cuda

        if gpc.config.model.use_flash_attn and ctx.is_using_cuda:
            backward_func = fused_dense_cuda.linear_bias_wgrad
        else:
            backward_func = linear_bias_wgrad_torch

        if ctx.compute_weight_gradient:
            total_x, weight = ctx.saved_tensors
        else:
            (weight,) = ctx.saved_tensors
            total_x = None
        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, weight.t())
            else:
                grad_input = torch.addmm(
                    grad_input.reshape(batch_dim, grad_input.shape[-1]),
                    grad_output,
                    weight,
                )
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            if process_group is not None:
                if sequence_parallel:
                    grad_input, handle_grad_input = reduce_scatter_raw(
                        grad_input, process_group, async_op=True, reduce_dim=1
                    )
                else:
                    grad_input, handle_grad_input = all_reduce_raw(grad_input, process_group, async_op=True)
        else:
            grad_input = None
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            grad_weight, grad_bias = backward_func(
                total_x.reshape(batch_dim, total_x.shape[-1]),
                grad_output,
                ctx.needs_input_grad[2],
            )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None
        if process_group is not None and ctx.needs_input_grad[0] and handle_grad_input is not None:
            handle_grad_input.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class ISPFusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for ISP, which is optimized based on flash implementation."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        weight,
        bias,
        module,
        communicator,
        return_residual=False,
        is_using_cuda: bool = True,
    ):
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.module = module
        ctx.communicator = communicator
        ctx.is_using_cuda = is_using_cuda

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()

        total_weight = communicator.all_gather(weight, module)
        total_bias = bias if bias is None else communicator.all_gather(bias, module, is_bias=True)

        if torch.is_autocast_enabled():
            total_weight = total_weight.to(dtype=torch.get_autocast_gpu_dtype())
            if total_bias:
                total_bias.to(dtype=torch.get_autocast_gpu_dtype())

        total_weight = total_weight.contiguous()
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *total_weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")

        output = F.linear(x, total_weight, total_bias)

        # release memory
        del total_weight
        del total_bias
        if ctx.compute_weight_gradient:
            ctx.save_for_backward(x, weight)
        else:
            ctx.save_for_backward(weight)
        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        module = ctx.module
        communicator = ctx.communicator

        if gpc.config.model.use_flash_attn:
            import fused_dense_lib as fused_dense_cuda

        if gpc.config.model.use_flash_attn and ctx.is_using_cuda:
            backward_func = fused_dense_cuda.linear_bias_wgrad
        else:
            backward_func = linear_bias_wgrad_torch

        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()

        if ctx.compute_weight_gradient:
            x, weight = ctx.saved_tensors
        else:
            x, weight = (None, *ctx.saved_tensors)

        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])

        total_weight = communicator.all_gather(weight, module)

        # compute weight grad
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            grad_weight, grad_bias = backward_func(
                x.reshape(batch_dim, x.shape[-1]),
                grad_output,
                ctx.needs_input_grad[2],
            )

            grad_weight, grad_weight_sync = communicator.reduce_scatter(grad_weight, module, op=dist.ReduceOp.AVG)
            if grad_bias is not None:
                grad_bias, grad_bias_sync = communicator.reduce_scatter(
                    grad_bias, module, op=dist.ReduceOp.AVG, is_bias=True
                )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None

        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = F.linear(grad_output, total_weight.t())
            else:
                grad_input = torch.addmm(
                    grad_input.reshape(batch_dim, grad_input.shape[-1]),
                    grad_output,
                    total_weight,
                )
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
        else:
            grad_input = None

        del total_weight

        if ctx.needs_input_grad[1]:
            if grad_weight_sync:
                grad_weight_sync.wait()
            if grad_bias is not None and grad_bias_sync is not None:
                grad_bias_sync.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None


def fused_dense_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
    gather_dim: int = 0,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    is_using_cuda = (internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU) and dtype_eligible
    return FusedDenseFunc.apply(
        x,
        weight,
        bias,
        return_residual,
        process_group,
        sequence_parallel,
        gather_dim,
        is_using_cuda,
    )


def megatron_fused_dense_func(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
    process_group: Optional[ProcessGroup] = None,
    sequence_parallel: bool = True,
    gather_dim: int = 0,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    is_using_cuda = (internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU) and dtype_eligible
    return MegatronFusedDenseFunc.apply(
        x,
        weight,
        bias,
        return_residual,
        process_group,
        sequence_parallel,
        gather_dim,
        is_using_cuda,
    )


def isp_fused_dense_func(
    x: Tensor,
    weight: Tensor,
    module,
    communicator,
    bias: Optional[Tensor] = None,
    return_residual: bool = False,
):
    dtype_eligible = x.dtype in [torch.float16, torch.bfloat16] or (
        x.dtype == torch.float32 and torch.is_autocast_enabled()
    )
    is_using_cuda = (internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU) and dtype_eligible
    return ISPFusedDenseFunc.apply(
        x,
        weight,
        bias,
        module,
        communicator,
        return_residual,
        is_using_cuda,
    )


def try_import_RMSNorm():
    """
    Try import MixFusedRMSNorm from apex, if failed, return our RMSNorm

    """
    try:
        from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as RMSNorm

        return RMSNorm
    except (ModuleNotFoundError, ImportError):
        logger.warning("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
        from internlm.model.ops.norm import RMSNormTorch as RMSNorm

        return RMSNorm


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "is_expert") and param.is_expert:
        return True
    return False


def Silu(w1_o, w2_o):
    return F.silu(w1_o) * w2_o


Silu = torch.jit.script(Silu)
