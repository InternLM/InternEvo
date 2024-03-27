"""
layer norm modules
"""

from typing import List, Union

import torch
from torch import nn

from internlm.model.ops.norm import RMSNorm

Shape = Union[int, List[int], torch.Size]


def new_layer_norm(norm_type: str, normalized_shape: Shape, eps: float = 1e-5):
    if norm_type == "rmsnorm":
        return RMSNorm(normalized_shape, eps)
    else:  # default: layernorm
        return nn.LayerNorm(normalized_shape, eps)
