#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
from torch import nn
from torch.distributed import ProcessGroup

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.utils import (
    all_reduce,
    fused_dense_func,
    isp_fused_dense_func,
    megatron_fused_dense_func,
    reduce_scatter,
)
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class BaseScaleColumnParallelLinear(nn.Linear):
    """
    Base class for ScaleColumnParallelLinear.

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
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % world_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by " f"world_size ({world_size})")
        super().__init__(in_features, out_features // world_size, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.weight_scale = weight_scale


class ScaleColumnParallelLinear(BaseScaleColumnParallelLinear):
    """
    ScaleColumnParallelLinear in flash implementation.
    """

    def forward(self, input, gather_dim=1, tp_mode: str = "mtp"):  # pylint: disable=W0622
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight

        _fused_func = fused_dense_func if tp_mode in ["mtp", "fsp", "isp"] else megatron_fused_dense_func
        return _fused_func(
            input,
            weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            gather_dim=gather_dim,
        )


class ScaleColumnParallelLinearWithNormHead(BaseScaleColumnParallelLinear):
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
        super().__init__(
            in_features, out_features, process_group, bias=bias, device=device, dtype=dtype, weight_scale=weight_scale
        )

        self.norm_head = norm_head
        if self.norm_head:
            logger.info("Notice that norm head is enabled to normalize head weight.")
        self.first_eval_flag = True
        self.tmp_weight = None

    def forward(self, input, gather_dim=1, tp_mode: str = "mtp"):  # pylint: disable=W0622
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
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

        _fused_func = fused_dense_func if tp_mode in ["mtp", "fsp", "isp"] else megatron_fused_dense_func
        return _fused_func(
            input,
            weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            gather_dim=gather_dim,
        )


class RewardModelLinear(BaseScaleColumnParallelLinear):
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
        torch.distributed.broadcast(self.weight, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], process_group)
        if bias:
            torch.distributed.broadcast(self.bias, gpc.get_ranks_in_group(ParallelMode.TENSOR)[0], process_group)

    def forward(self, input):  # pylint: disable=W0622
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        if self.weight_scale != 1:
            weight = self.weight * self.weight_scale + (1 - self.weight_scale) * self.weight.detach()
        else:
            weight = self.weight
        return fused_dense_func(
            input,
            weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
        )


class ColumnParallelLinearTorch(nn.Linear):
    """
    ColumnParallelLinearTorch.
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
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
        multiple = out_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        super().__init__(in_features, local_multiple * multiple_of, bias=bias, device=device, dtype=dtype)
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x, gather_dim=1):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        return fused_dense_func(
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
            gather_dim=gather_dim,
        )


class MegatronColumnParallelLinearTorch(ColumnParallelLinearTorch):
    """
    MegatronColumnParallelLinearTorch
    """

    def forward(self, x, gather_dim=1):
        # If self.sequence_parallel is True, we're doing Tensor Parallel with sequence parallelism:
        # we do an all_gather of x before doing the matmul.
        # If not, then the input is already gathered.
        return megatron_fused_dense_func(
            x,
            self.weight,
            self.bias,
            process_group=self.process_group,
            sequence_parallel=self.sequence_parallel,
            gather_dim=gather_dim,
        )


class RowParallelLinearTorch(nn.Linear):
    """
    RowParallelLinearTorch.
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
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if in_features % multiple_of:
            raise ValueError(f"in_features ({in_features}) must be a multiple of {multiple_of}")
        multiple = in_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        # Only rank 0 will have bias
        super().__init__(
            local_multiple * multiple_of,
            out_features,
            bias=bias and rank == 0,
            device=device,
            dtype=dtype,
        )
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def forward(self, x, reduce_dim=1):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = fused_dense_func(x, self.weight, self.bias)
        if self.sequence_parallel:
            return reduce_scatter(out, self.process_group, reduce_dim)
        else:
            return all_reduce(out, self.process_group)


class MegatronRowParallelLinearTorch(RowParallelLinearTorch):
    """
    MegatronRowParallelLinearTorch.
    """

    def forward(self, x, reduce_dim=1):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = megatron_fused_dense_func(x, self.weight, self.bias)
        if self.sequence_parallel:
            return reduce_scatter(out, self.process_group, reduce_dim)
        else:
            return all_reduce(out, self.process_group)


class ISPLinear(ColumnParallelLinearTorch):
    """
    Linear class for isp tensor parallel mode.
    """

    # class level communicator variable.
    __communicator = None

    @staticmethod
    def register_communicator(communicator):
        ISPLinear.__communicator = communicator

    def forward(self, x):
        assert self.__communicator is not None, "ISPLinear should be register with a communicator first."

        return isp_fused_dense_func(
            x,
            self.weight,
            module=self,
            communicator=self.__communicator,
            bias=self.bias,
        )


def get_linear_cls(tp_mode: str, parallel_mode: str):
    if parallel_mode == "column":
        if tp_mode in ["mtp", "fsp"]:
            cls = ColumnParallelLinearTorch
        elif tp_mode == "msp":
            cls = MegatronColumnParallelLinearTorch
        else:
            cls = ISPLinear
    elif parallel_mode == "row":
        if tp_mode in ["mtp", "fsp"]:
            cls = RowParallelLinearTorch
        elif tp_mode == "msp":
            cls = MegatronRowParallelLinearTorch
        else:
            cls = ISPLinear
    return cls
