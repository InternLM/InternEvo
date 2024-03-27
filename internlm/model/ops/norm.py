# adopted from https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm

import numbers

import torch
from torch.nn import init
from torch.nn.parameter import Parameter

from internlm.utils.logger import get_logger

logger = get_logger(__file__)

try:
    from apex.normalization.fused_layer_norm import \
        mixed_dtype_fused_rms_norm_affine

    apex_rmsnorm_impl = True
except (ModuleNotFoundError, ImportError):
    logger.warning("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
    apex_rmsnorm_impl = False


def manual_rms_norm(my_input, weight, normalized_shape, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = my_input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    my_input = my_input * torch.rsqrt(variance + eps)

    if weight is None:
        return my_input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        my_input = my_input.to(weight.dtype)

    return weight * my_input


class RMSNorm(torch.nn.Module):
    """A custom PyTorch module for RMS normalization."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.empty(*normalized_shape))
        self.reset_parameters()

    def forward(self, _input: torch.Tensor):
        if apex_rmsnorm_impl:
            _norm_func = mixed_dtype_fused_rms_norm_affine
        else:
            _norm_func = manual_rms_norm

        return _norm_func(_input, self.weight, self.normalized_shape, self.eps)

    def reset_parameters(self):
        init.ones_(self.weight)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, ".format(**self.__dict__)
