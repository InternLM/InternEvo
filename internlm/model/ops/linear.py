"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the linear layer operators.
"""

from typing import Optional, Tuple

import torch
from torch.nn.functional import linear as _torch_linear_forward_op

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import global_context as gpc

from test_torchao import per_col_quantize_int8, per_row_quantize_int8

try:
    from torchao.prototype.quantized_training.int8_mm import scaled_int8_mm
except (ModuleNotFoundError, ImportError):
    print('torchao not found', flush=True)

try:
    from fused_dense_lib import linear_bias_wgrad as _flash_linear_backward_op

    flash_attn_impl = True
except (ModuleNotFoundError, ImportError):
    flash_attn_impl = False

try:
    # grouped_gemm on GPU
    from grouped_gemm.backend import gmm as gmm_ops
except (ModuleNotFoundError, ImportError):
    # grouped_gemm on NPU
    try:
        from mindspeed.op_builder import GMMOpBuilder

        gmm_ops = GMMOpBuilder().load()
    except (ModuleNotFoundError, ImportError):
        gmm_ops = None

internlm_accelerator = get_accelerator()


def _select_ops_binding(dtype: torch.dtype, is_cuda: bool = True) -> None:
    dtype_eligible = dtype in (torch.float16, torch.bfloat16) or (
        dtype == torch.float32 and torch.is_autocast_enabled()
    )
    use_flash_attn = gpc.config.model.get("use_flash_attn", False)
    is_gpu_backend = internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU
    flash_attn_eligible = flash_attn_impl and dtype_eligible and is_cuda

    if use_flash_attn and is_gpu_backend and flash_attn_eligible:
        if not gpc.config.int8_training:
            return _torch_linear_forward_op, _linear_bias_wgrad_torch #_flash_linear_backward_op
        else:
            return _int8_forward_op, _int8_backward_op
    else:
        assert False
        return _torch_linear_forward_op, _linear_bias_wgrad_torch
    

def _quantize_int8(A, B):
    A_int8, row_scale = per_row_quantize_int8(A, sr=True)
    B_int8, col_scale = per_col_quantize_int8(B, sr=True)
    return A_int8, B_int8, row_scale, col_scale


def _int8_forward_op(_input, weight, bias):
    assert _input.dtype == weight.dtype
    dtype = _input.dtype
    _input_int8, weight_t_int8, row_scale, col_scale = _quantize_int8(_input, weight.t())
    assert _input_int8.dtype == torch.int8
    output = scaled_int8_mm(_input_int8, weight_t_int8, row_scale, col_scale).to(dtype)
    assert bias is None
    if bias is not None:
        output += bias
    
    return output


def _int8_backward_op(_input: torch.Tensor, grad_output: torch.Tensor, has_d_bias: bool):
    assert _input.dtype == grad_output.dtype
    dtype = _input.dtype
    grad_output_t_int8, _input_int8, row_scale, col_scale = _quantize_int8(grad_output.t(), _input)
    assert _input_int8.dtype == torch.int8
    grad_weight = scaled_int8_mm(grad_output_t_int8, _input_int8, row_scale, col_scale).to(dtype)
    assert not has_d_bias
    grad_bias = grad_output.sum(dim=0) if has_d_bias else None

    return grad_weight, grad_bias


def _linear_bias_wgrad_torch(_input: torch.Tensor, grad_output: torch.Tensor, has_d_bias: bool):
    assert _input.dtype == grad_output.dtype

    grad_weight = torch.matmul(grad_output.t(), _input)
    grad_bias = grad_output.sum(dim=0) if has_d_bias else None

    return grad_weight, grad_bias


def linear_forward_op(_input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    _is_cuda = internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU
    _forward_op, _ = _select_ops_binding(_input.dtype, _is_cuda)

    return _forward_op(_input, weight, bias)


def linear_backward_op(
    _input: torch.Tensor, weight: torch.Tensor, has_d_bias: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _is_cuda = internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU
    _, _backward_op = _select_ops_binding(_input.dtype, _is_cuda)

    return _backward_op(_input, weight, has_d_bias)


def _gmm_forward_op_npu(_input: torch.Tensor, weight: torch.Tensor, batch_sizes: torch.Tensor) -> torch.Tensor:
    group_list = torch.cumsum(batch_sizes, dim=-1)
    group_list = group_list.tolist()
    group_type = 0

    return gmm_ops.npu_gmm([_input], [weight], [], group_list, group_type)[0]


def _gmm_backward_op_npu(
    _input: torch.Tensor, weight: torch.Tensor, batch_sizes: torch.Tensor, grad_output: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    group_list = torch.cumsum(batch_sizes, dim=-1)
    group_list = group_list.tolist()

    dx, dw, _ = gmm_ops.npu_gmm_backward([grad_output], [_input], [weight], group_list)

    return dx[0], dw[0]


def gmm_forward_op(_input: torch.Tensor, weight: torch.Tensor, batch_sizes: torch.Tensor) -> torch.Tensor:
    if internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU:
        return gmm_ops(_input, weight, batch_sizes)
    elif internlm_accelerator.get_accelerator_backend() is AcceleratorType.NPU:
        return _gmm_forward_op_npu(_input, weight, batch_sizes)


def gmm_backward_op(
    _input: torch.Tensor, weight: torch.Tensor, batch_sizes: torch.Tensor, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    if internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU:
        if kwargs.get("is_grad_input", False):
            trans_a, trans_b = (False, True)
            return gmm_ops(_input, weight, batch_sizes, trans_a=trans_a, trans_b=trans_b), None
        else:
            trans_a, trans_b = (True, False)
            return None, gmm_ops(_input, weight, batch_sizes, trans_a=trans_a, trans_b=trans_b)

    elif internlm_accelerator.get_accelerator_backend() is AcceleratorType.NPU:
        input_weight = kwargs.get("input_weight", None)
        grad_output = weight
        return _gmm_backward_op_npu(_input, input_weight, batch_sizes, grad_output)
