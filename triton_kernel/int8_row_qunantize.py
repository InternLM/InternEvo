import math

import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

# rowwise quantize

# TODO: autotune this better.
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    output = libdevice.llrint(127.0 * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_scale + pid, max_val / 127)

@triton.jit
def _quantize_rowwise_sr(
    x_ptr,
    output_ptr,
    output_scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    abs_x = tl.abs(x).to(tl.float32)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    scale = 127.0 / max_val
    scaled_x = x * scale

    # sr + clamp
    floor_x = tl.math.floor(scaled_x)
    frac_x = scaled_x - floor_x
    rand = tl.rand(1024, offsets)                
    rounded = tl.where(rand < frac_x, floor_x + 1, floor_x)
    output = tl.where(rounded > 127, 127, rounded)
    output = tl.where(output < -127, -127, output)

    tl.store(output_ptr + offsets, output.to(tl.int8), mask=row_mask)
    tl.store(output_scale + pid, max_val / 127)

def quantize_rowwise(x: torch.Tensor, sr=True):
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_scale = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)

    if not sr:
        _quantize_rowwise[grid](x, output, output_scale, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    else:
        _quantize_rowwise_sr[grid](x, output, output_scale, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)

    return output, output_scale


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from test_torchao import per_row_quantize_int8
    torch.manual_seed(1024)
    x = torch.randn(4096, 2048, dtype=torch.float32, device="cuda")
    x_int8, row_scale = per_row_quantize_int8(x, sr=True)
    x_int8_2, row_scale_2 = quantize_rowwise(x, sr=True)
    x_int8_3, row_scale_3 = quantize_rowwise(x, sr=False)
    assert x_int8.shape == x_int8_2.shape == x_int8_3.shape
    assert row_scale.shape == row_scale_2.shape == row_scale_3.shape
    assert torch.equal(row_scale, row_scale_2)
    assert torch.equal(row_scale, row_scale_3)

    print("tensor1:\n", x_int8, row_scale)
    print("tensor2:\n", x_int8_2, row_scale_2)
    print("tensor3:\n", x_int8_3, row_scale_3)

    error1 = (x_int8 - x_int8_2).float().abs().mean()
    error2 = (x_int8_2 - x_int8_3).float().abs().mean()
    error3 = (x_int8 - x_int8_3).float().abs().mean()
    print("x 平均误差:", error1.item(), error2.item(), error3.item())

    error4 = (row_scale - row_scale_2).float().abs().mean()
    error5 = (row_scale_2 - row_scale_3).float().abs().mean()
    error6 = (row_scale - row_scale_3).float().abs().mean()
    print("scale 平均误差:", error4.item(), error5.item(), error6.item())
    