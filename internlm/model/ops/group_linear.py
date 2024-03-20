#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math

import torch
from torch import nn
from torch.distributed import ProcessGroup

from internlm.model.utils import (
    all_reduce,
    fused_dense_func,
    isp_fused_dense_func,
    megatron_fused_dense_func,
    reduce_scatter,
)


class GroupedColumnParallelLinear(nn.Module):
    """
    GroupedColumnParallelLinear.
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        num_linear_in_group (int): number of linear modules
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
        num_linear_in_group: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        world_size = torch.distributed.get_world_size(process_group)
        if out_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
        multiple = out_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        self.weight = nn.Parameter(
            torch.empty(num_linear_in_group, local_multiple * multiple_of, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(num_linear_in_group, local_multiple * multiple_of, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gather_dim=1):  # pylint: disable=W0237
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
            is_grouped_linear=True,
        )


class GroupedMegatronColumnParallelLinear(GroupedColumnParallelLinear):
    """
    GroupedMegatronColumnParallelLinear
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
            is_grouped_linear=True,
        )


class GroupedRowParallelLinear(nn.Module):
    """
    GroupedRowParallelLinear.
    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        num_linear_in_group (int): number of linear modules
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
        num_linear_in_group: int,
        process_group: ProcessGroup,
        bias: bool = True,
        sequence_parallel=True,
        multiple_of=1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if in_features % multiple_of:
            raise ValueError(f"out_features ({out_features}) must be a multiple of {multiple_of}")
        multiple = in_features // multiple_of
        # We want to split @multiple across world_size, but it could be an uneven split
        div = multiple // world_size
        mod = multiple % world_size
        # The first @mod ranks get @div + 1 copies, the rest get @div copies
        local_multiple = div + int(torch.distributed.get_rank(process_group) < mod)
        self.weight = nn.Parameter(
            torch.empty(num_linear_in_group, out_features, local_multiple * multiple_of, device=device, dtype=dtype)
        )
        if bias and rank == 0:
            self.bias = nn.Parameter(torch.empty(num_linear_in_group, out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):  # pylint: disable=W0237
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = fused_dense_func(x, self.weight, self.bias, gather_dim=1, is_grouped_linear=True)
        if self.sequence_parallel:
            return reduce_scatter(out, self.process_group, 1)
        else:
            return all_reduce(out, self.process_group)


class GroupedMegatronRowParallelLinear(GroupedRowParallelLinear):
    """
    MegatronRowParallelLinearTorch.
    """

    def forward(self, x):
        """
        We're doing Tensor Parallel with sequence parallelism: we do the matmul and then
        a reduce_scatter of the result.
        """
        out = megatron_fused_dense_func(x, self.weight, self.bias, gather_dim=1, is_grouped_linear=True)
        if self.sequence_parallel:
            return reduce_scatter(out, self.process_group, 1)
        else:
            return all_reduce(out, self.process_group)


class GroupedISPLinear(GroupedColumnParallelLinear):
    """
    Group linear class for isp tensor parallel mode.
    """

    # class level communicator variable.
    __communicator = None

    @staticmethod
    def register_communicator(communicator):
        GroupedISPLinear.__communicator = communicator

    def forward(self, x):
        assert self.__communicator is not None, "ISPGroupLinear should be register with a communicator first."

        return isp_fused_dense_func(
            x,
            self.weight,
            module=self,
            communicator=self.__communicator,
            bias=self.bias,
            is_grouped_linear=True,
        )
