#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional

import torch
from torch import nn

from internlm.model.modules.linear import new_linear
from internlm.model.modules.utils import Silu
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class FeedForward(nn.Module):
    """
    Base FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
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
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__()

        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)
        self.w1 = new_linear("w1", in_features, hidden_features, bias, device=device, dtype=dtype)
        self.w2 = new_linear("w2", hidden_features, out_features, bias, device=device, dtype=dtype)
        self.w3 = new_linear("w3", in_features, hidden_features, bias, device=device, dtype=dtype)

    def forward(self, x):
        # TODO: support gelu...
        return self.w2(Silu(self.w1(x), self.w3(x)))


def new_fead_forward(
    in_features: int,
    hidden_features: int,
    out_features: int = None,
    bias: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    multiple_of: int = 256,
) -> FeedForward:
    return FeedForward(in_features, hidden_features, out_features, bias, device, dtype, multiple_of)
