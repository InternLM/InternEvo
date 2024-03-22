#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable, Optional

import torch
from torch import nn

from internlm.core.context import global_context as gpc
from internlm.model.ops.linear import (
    ColumnParallelLinearTorch,
    ISPLinear,
    MegatronColumnParallelLinearTorch,
    MegatronRowParallelLinearTorch,
    RowParallelLinearTorch,
)
from internlm.model.utils import Silu


class BaseFeedForward(nn.Module):
    """
    Base FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
        column_cls (Optional[Callable]): The column parallel class for w1 and w3. None by default.
        row_cls (Optional[Callable]): The row parallel class for w2. None by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        comm_dim: int = 0,
        column_cls: Optional[Callable] = None,
        row_cls: Optional[Callable] = None,
    ):
        super().__init__()
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = column_cls(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
            comm_dim=comm_dim,
        )
        self.w2 = row_cls(
            hidden_features,
            out_features,
            process_group,
            bias=bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
            comm_dim=comm_dim,
        )
        self.w3 = column_cls(
            in_features,
            hidden_features,
            process_group,
            bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
            comm_dim=comm_dim,
        )

    def forward(self, x):
        w1_o = self.w1(x)
        w3_o = self.w3(x)
        out = self.w2(Silu(w1_o, w3_o))
        return out


class FeedForward(BaseFeedForward):
    """
    FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        comm_dim: int = 0,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            comm_dim,
            ColumnParallelLinearTorch,
            RowParallelLinearTorch,
        )


class MegatronFeedForward(BaseFeedForward):
    """
    FeedForward in megatron implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        comm_dim: int = 0,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            comm_dim,
            MegatronColumnParallelLinearTorch,
            MegatronRowParallelLinearTorch,
        )


class ISPFeedForward(BaseFeedForward):
    """
    FeedForward in ISP.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        comm_dim: int = 0,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            comm_dim,
            ISPLinear,
            ISPLinear,
        )


def get_mlp_cls(tp_mode: str):
    if tp_mode in ["mtp", "fsp"]:
        mlp_cls = FeedForward
    elif tp_mode == "msp":
        mlp_cls = MegatronFeedForward
    else:
        mlp_cls = ISPFeedForward
    return mlp_cls
