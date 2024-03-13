"""
Linear Modules
"""

from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
import torch.distributed as dist

from internlm.core.context import global_context as gpc, ParallelMode
from internlm.core.parallel.comm.isp import ISPCommunicator
from internlm.core.parallel.comm.tensor import TPCommunicator
from internlm.core.parallel.shard import get_tensor_split_parallel_mode
from internlm.core.parallel.shard import get_parallel_strategies_split_mode
from internlm.model.ops.linear import linear_forward_op, linear_backward_op
from internlm.utils.logger import get_logger

# from .utils import all_gather_raw, reduce_scatter_raw, all_reduce_raw

logger = get_logger(__file__)


# adpated from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py
class FusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for tensor parallel in flash-attn implementation."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        communicator: TPCommunicator,
        return_residual=False,
    ):
        """
        If process_group is not None and sequence_parallel=True, we're doing Tensor Parallel
        with sequence parallelism: we do an all_gather_raw of x before doing the matmul.
        """
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.communicator = communicator

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()

        # parallel strategy-specific communication callback 1-1: gathers x if necessary.
        # see more details in the communicator for different parallel strategies.
        # we want to kick off the all_gather early, before weight dtype conversion.
        total_x, handle_x = communicator.gather_input(x, async_op=True)

        if torch.is_autocast_enabled():
            weight = weight.to(dtype=torch.get_autocast_gpu_dtype())
            bias = bias.to(dtype=torch.get_autocast_gpu_dtype()) if bias is not None else None
        weight = weight.contiguous()

        # wait for x has been gathered.
        handle_x.wait()

        batch_shape, n = total_x.shape[:-1], total_x.shape[-1]
        batch_dim = batch_shape.numel()
        # https://github.com/pytorch/pytorch/blob/5b51849b48a7dbccd297286cc0110def4706f9e7/aten/src/ATen/native/cuda/Blas.cpp#L174
        if min(batch_dim, n, *weight.shape) > 65535 * 32:
            raise RuntimeError("fused_dense only supports matrix dims <= 2M")

        output = linear_forward_op(total_x, weight, bias)

        # parallel strategy-specific communication callback 2: reduce output if necessary.
        # see more details in the communicator for different parallel strategies.
        output, _ = communicator.reduce_output(output, async_op=False)

        saved_x = None if ctx.compute_weight_gradient is False else total_x if communicator.save_total_input() else x
        ctx.save_for_backward(saved_x, weight)

        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        communicator: TPCommunicator = ctx.communicator

        # parallel strategy-specific communication callback 3: gathers grad_output if necessary.
        # see more details in the communicator for different parallel strategies.
        grad_output, _ = communicator.gather_grad_output(grad_output, async_op=False)
        grad_output = grad_output.contiguous()

        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()

        x, weight = ctx.saved_tensors

        # parallel strategy-specific communication callback 1-2: gathers x if necessary.
        # see more details in the communicator for different parallel strategies.
        if ctx.needs_input_grad[1]:
            x, handle_x = communicator.gather_input(x, async_op=True)

        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])

        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = linear_forward_op(grad_output, weight.t())
            else:
                grad_input = torch.addmm(
                    grad_input.reshape(batch_dim, grad_input.shape[-1]),
                    grad_output,
                    weight,
                )
            grad_input = grad_input.reshape(*batch_shape, grad_input.shape[-1])
            # parallel strategy-specific communication callback 4: reduce grad_input if necessary.
            # see more details in the communicator for different parallel strategies.
            grad_input, handle_grad_input = communicator.reduce_grad_input(grad_input, async_op=True)
        else:
            grad_input = None

        # computes gradinets for weight and bias if necessary
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient

            # wait for x has been gathered
            handle_x.wait()

            x = x.reshape(batch_dim, x.shape[-1])
            grad_weight, grad_bias = linear_backward_op(x, grad_output, ctx.needs_input_grad[2])
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None

        # wait for grad_input has been gathered
        handle_grad_input.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


# TODO: 我们是否应该统一 ISPFusedDenseFunc 和 FusedDenseFunc，以及相关 communicator interface.
class ISPFusedDenseFunc(torch.autograd.Function):
    "FusedDenseFunc for ISP, which is optimized based on flash implementation."

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        module: nn.Module,
        communicator: ISPCommunicator,
        return_residual=False,
    ):
        ctx.compute_weight_gradient = weight.requires_grad
        ctx.return_residual = return_residual
        ctx.module = module
        ctx.communicator = communicator

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

        output = linear_forward_op(x, total_weight, total_bias)

        # release memory
        del total_weight
        del total_bias

        saved_x = None if ctx.compute_weight_gradient is False else x
        ctx.save_for_backward(saved_x, weight)

        return output if not return_residual else (output, x)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, *args):
        module: nn.Module = ctx.module
        communicator: ISPCommunicator = ctx.communicator
        x, weight = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        if ctx.return_residual:
            (grad_input,) = args
            grad_input = grad_input.contiguous()

        batch_shape = grad_output.shape[:-1]
        batch_dim = batch_shape.numel()
        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])

        total_weight = communicator.all_gather(weight, module)

        # compute weight grad
        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            grad_weight, grad_bias = linear_backward_op(
                x.reshape(batch_dim, x.shape[-1]),
                grad_output,
                ctx.needs_input_grad[2],
            )

            grad_weight, grad_weight_sync = communicator.reduce_scatter(
                grad_weight, module, op=dist.ReduceOp.AVG, is_bias=False
            )
            if grad_bias is not None:
                grad_bias, grad_bias_sync = communicator.reduce_scatter(
                    grad_bias, module, op=dist.ReduceOp.AVG, is_bias=True
                )
        else:
            grad_weight = None
            grad_bias = grad_output if ctx.needs_input_grad[2] else None

        if ctx.needs_input_grad[0]:
            if not ctx.return_residual:
                grad_input = linear_forward_op(grad_output, total_weight.t())
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
            grad_weight_sync.wait()
            if grad_bias is not None:
                grad_bias_sync.wait()

        return grad_input, grad_weight, grad_bias, None, None, None, None


def fused_dense_func(
    x: torch.Tensor,
    weight: torch.Tensor,
    communicator,
    module: Optional[nn.Module] = None,
    bias: Optional[torch.Tensor] = None,
    return_residual: bool = False,
    # process_group: Optional[dist.ProcessGroup] = None,
    # sequence_parallel: bool = True,
    # gather_dim: int = 0,
):
    if gpc.config.parallel.tensor.mode == "isp":
        return ISPFusedDenseFunc.apply(
            x,
            weight,
            bias,
            module,
            communicator,
            return_residual,
        )
    elif gpc.config.parallel.tensor.mode in ("mtp", "msp"):
        return FusedDenseFunc.apply(
            x,
            weight,
            bias,
            communicator,
            return_residual,
            save_total_x=True,
        )
    else:  # fsp
        return FusedDenseFunc.apply(
            x,
            weight,
            bias,
            communicator,
            return_residual,
            save_total_x=False,
        )


class ColumnParallelLinear(nn.Linear):
    """
    ColumnParallelLinear

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        sequence_parallel (bool): If sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
                                    we do an all_gather of x before doing the matmul.
                                    If not, then the input is already gathered.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
    """

    # class level communicator variable.
    _communicator = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        parallel_mode = get_tensor_split_parallel_mode()
        world_size = gpc.get_world_size(parallel_mode)
        rank = gpc.get_local_rank(parallel_mode)

        if out_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
        multiple = out_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(rank < mod)
        super().__init__(in_features, local_multiple * multiple_of, bias=bias, device=device, dtype=dtype)

        # self.process_group = process_group
        # self.sequence_parallel = sequence_parallel

    @staticmethod
    def register_communicator(communicator):
        ColumnParallelLinear._communicator = communicator

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self._communicator is not None, "ColumnParallelLinear should be register with a communicator first."

        return fused_dense_func(
            input,
            self.weight,
            module=self,
            communicator=self._communicator,
            bias=self.bias,
        )


class RowParallelLinear(nn.Linear):
    """
    RowParallelLinear

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        sequence_parallel (bool): If sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
                                    we do an all_gather of x before doing the matmul.
                                    If not, then the input is already gathered.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
    """

    # class level communicator variable.
    _communicator = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        parallel_mode = get_tensor_split_parallel_mode()
        world_size = gpc.get_world_size(parallel_mode)
        rank = gpc.get_local_rank(parallel_mode)

        if in_features % multiple_of:
            raise ValueError(f"in_features ({in_features}) must be a multiple of {multiple_of}")
        multiple = in_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(rank < mod)
        # Only rank 0 will have bias
        super().__init__(
            local_multiple * multiple_of,
            out_features,
            bias=bias and rank == 0,
            device=device,
            dtype=dtype,
        )
        # self.process_group = process_group
        # self.sequence_parallel = sequence_parallel

    @staticmethod
    def register_communicator(communicator):
        RowParallelLinear._communicator = communicator

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        assert self._communicator is not None, "RowParallelLinear should be register with a communicator first."

        return fused_dense_func(
            input,
            self.weight,
            module=self,
            communicator=self._communicator,
            bias=self.bias,
        )


class ScaleColumnParallelLinear(nn.Linear):
    """
    ScaleColumnParallelLinear for InternLM2.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
        norm_head (bool): Normalize the output embedding in order to let the calculation of logits not affected by
            the norm of embedding. The implementation is referred to baichuan2,
            see https://huggingface.co/baichuan-inc/Baichuan2-7B-Base for more information. False by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
        norm_head: bool = False,
    ) -> None:
        if norm_head:
            logger.info("Notice that norm head is enabled to normalize head weight.")

        world_size = dist.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by " f"world_size ({world_size})")
        super().__init__(in_features, out_features // world_size, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.weight_scale = weight_scale

        self.norm_head = norm_head
        self.first_eval_flag = True
        self.tmp_weight = None

    def forward(self, input, gather_dim=0):  # pylint: disable=W0622
        if self.weight_scale == 1:
            weight = self.weight
        else:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()

        if self.norm_head:
            if self.training:
                if not self.first_eval_flag:
                    self.first_eval_flag = True
                    self.tmp_weight = None
                # We normalized the output Embedding so that the dot product
                # is not affected by the norm of embedding. Ref: https://arxiv.org/pdf/2309.10305.pdf
                weight = nn.functional.normalize(weight)
            else:
                if self.first_eval_flag:
                    # cache l2 norm of head to accelerate infer.
                    self.first_eval_flag = False
                    self.tmp_weight = nn.functional.normalize(weight)

                weight = self.tmp_weight

        return fused_dense_func(
            input,
            weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            gather_dim=gather_dim,
        )


class RewardModelLinear(ScaleColumnParallelLinear):
    """
    RewardModelLinear.
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        sequence_parallel (bool): If sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
                                    we do an all_gather of x before doing the matmul.
                                    If not, then the input is already gathered.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        weight_scale (int): For training stability. 1 by default.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        weight_scale: int = 1,
    ) -> None:
        super().__init__(in_features, out_features, process_group, bias, device, dtype, weight_scale)

        dist.broadcast(self.weight, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], process_group)
        if bias:
            dist.broadcast(self.bias, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], process_group)


def new_linear_instance(
    name: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    multiple_of=1,
    device=None,
    dtype=None,
    is_reward: bool = False,
    weight_scale: int = 1,
    norm_head: bool = False,
) -> nn.Linear:

    name = str.lower(name)

    if name in ("head", "output"):
        if is_reward:
            return RewardModelLinear(
                in_features,
                out_features,
                None,  # TODO: fix process_group.
                bias,
                device,
                dtype,
                weight_scale,
            )
        else:
            return ScaleColumnParallelLinear(
                in_features,
                out_features,
                None,  # TODO: fix process_group.
                bias,
                device,
                dtype,
                weight_scale=weight_scale,
                norm_head=norm_head,
            )

    split_mode = get_parallel_strategies_split_mode(name)

    if split_mode == "column":
        return ColumnParallelLinear(
            in_features,
            out_features,
            bias,
            multiple_of,
            device,
            dtype,
        )
    elif split_mode == "row":
        return RowParallelLinear(
            in_features,
            out_features,
            bias,
            multiple_of,
            device,
            dtype,
        )
    else:
        raise ValueError(f"parallel strategies for linear is unsupported, which is named as {name}")
