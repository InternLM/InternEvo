"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the linear layer operators.
"""

import torch
from typing import Optional, Tuple
from torch.nn.functional import linear as __torch_linear_forward_op

from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

from .utils import OpsBinding

try:
    from fused_dense_lib import linear_bias_wgrad as __flash_linear_backward_op

    flash_attn_impl = True
except ImportError:
    flash_attn_impl = False

logger = get_logger(__file__)

__bound_ops__ = OpsBinding(
    {
        "linear_forward_op": None,
        "linear_backward_op": None,
    }
)


def __select_ops_binding(dtype: torch.dtype, is_cuda: bool = True) -> None:
    dtype_eligible = dtype in (torch.float16, torch.bfloat16) or (
        dtype == torch.float32 and torch.is_autocast_enabled()
    )
    use_flash_attn = gpc.config.model.get("use_flash_attn", False)
    falsh_attn_eligible = flash_attn_impl and dtype_eligible and is_cuda

    if use_flash_attn and falsh_attn_eligible:
        __bound_ops__.linear_forward_op = __torch_linear_forward_op
        __bound_ops__.linear_backward_op = __flash_linear_backward_op
    else:
        __bound_ops__.linear_forward_op = __torch_linear_forward_op
        __bound_ops__.linear_backward_op = __linear_bias_wgrad_torch


def __linear_bias_wgrad_torch(_input: torch.Tensor, grad_output: torch.Tensor, has_d_bias: bool):
    assert _input.dtype == grad_output.dtype

    grad_weight = torch.matmul(grad_output.t(), _input)
    grad_bias = grad_output.sum(dim=0) if has_d_bias else None

    return grad_weight, grad_bias


def linear_forward_op(_input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    __is_cuda = _input.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda)
    __select_ops_binding(_input.dtype, __is_cuda)

    return __bound_ops__.linear_forward_op(_input, weight, bias)


def linear_backward_op(
    _input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    __is_cuda = _input.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda)
    __select_ops_binding(_input.dtype, __is_cuda)

    return __bound_ops__.linear_backward_op(_input, weight, bias)
