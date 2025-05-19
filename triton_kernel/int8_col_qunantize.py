import math

import torch

import triton
import triton.language as tl
from triton.language.extra import libdevice

# This kernel does fused columnwise quantization and transpose.

# TODO: autotune this better.
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_stages=16),
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=16, num_warps=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_columnwise_and_transpose(
    x_ptr,
    output_ptr,
    output_scale,
    n_elements,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # M * N
    pid = tl.program_id(axis=0)
    block_start = pid
    p2_arange = tl.arange(0, P2)
    p2_arange_mask = p2_arange < M
    arange = p2_arange * N
    offsets = block_start + arange
    x = tl.load(x_ptr + offsets, mask=p2_arange_mask)
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(p2_arange_mask, abs_x, 0), axis=0)
    output = libdevice.llrint(127.0 * (x / max_val))

    # N * M
    # pid * N + M_arange
    # new_start = pid * M
    # new_offsets = new_start + p2_arange
    # tl.store(output_ptr + new_offsets, output, mask=p2_arange_mask)
    # tl.store(output_scale + pid, max_val / 127)

    tl.store(output_ptr + offsets, output, mask=p2_arange_mask)
    tl.store(output_scale + pid, max_val / 127)

@triton.jit
def _quantize_columnwise_and_transpose_sr(
    x_ptr,
    output_ptr,
    output_scale,
    n_elements,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    # M * N
    pid = tl.program_id(axis=0)
    block_start = pid
    p2_arange = tl.arange(0, P2)
    p2_arange_mask = p2_arange < M
    arange = p2_arange * N
    offsets = block_start + arange
    x = tl.load(x_ptr + offsets, mask=p2_arange_mask)

    abs_x = tl.abs(x).to(tl.float32)
    max_val = tl.max(tl.where(p2_arange_mask, abs_x, 0), axis=0)
    scale = 127.0 / max_val
    scaled_x = x * scale

    # sr + clamp
    floor_x = tl.math.floor(scaled_x)
    frac_x = scaled_x - floor_x
    rand = tl.rand(1024, offsets)                
    rounded = tl.where(rand < frac_x, floor_x + 1, floor_x)
    output = tl.where(rounded > 127, 127, rounded)
    output = tl.where(output < -127, -127, output)

    # N * M
    # pid * N + M_arange
    # new_start = pid * M
    # new_offsets = new_start + p2_arange
    # tl.store(output_ptr + new_offsets, output.to(tl.int8), mask=p2_arange_mask)
    # tl.store(output_scale + pid, max_val / 127)
    

    tl.store(output_ptr + offsets, output.to(tl.int8), mask=p2_arange_mask)
    tl.store(output_scale + pid, max_val / 127)

def quantize_columnwise_and_transpose(x: torch.Tensor, sr=True):
    M, N = x.shape
    # output = torch.empty(N, M, device=x.device, dtype=torch.int8)
    output = torch.empty(*x.shape, device=x.device, dtype=torch.int8)
    output_scale = torch.empty(x.shape[1], device=x.device, dtype=torch.float32)

    P2 = int(2 ** (math.ceil(math.log2(M))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    if not sr:
        _quantize_columnwise_and_transpose[grid](x, output, output_scale, n_elements, M, N, BLOCK_SIZE=M, P2=P2)
    else:
        _quantize_columnwise_and_transpose_sr[grid](x, output, output_scale, n_elements, M, N, BLOCK_SIZE=M, P2=P2)

    return output, output_scale


if __name__ == "__main__":
    # torch.manual_seed(1024)
    # x = torch.randn(4096, 2048, dtype=torch.float32, device="cuda")
    # x_int8_2, row_scale_2 = quantize_columnwise_and_transpose(x, sr=True)
    # print(x.shape, x_int8_2.shape, row_scale_2.shape)
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from test_torchao import per_col_quantize_int8
    torch.manual_seed(1024)
    x = torch.randn(4096, 2048, dtype=torch.float32, device="cuda")
    x_int8, scale = per_col_quantize_int8(x, sr=True)
    x_int8_2, scale_2 = quantize_columnwise_and_transpose(x, sr=True)
    x_int8_3, scale_3 = quantize_columnwise_and_transpose(x, sr=False)
    assert x_int8.shape == x_int8_2.shape == x_int8_3.shape
    assert scale.shape == scale_2.shape == scale_3.shape
    assert torch.equal(scale, scale_2)
    assert torch.equal(scale, scale_3)

    # print("tensor1:\n", x_int8, scale)
    # print("tensor2:\n", x_int8_2, scale_2)
    # print("tensor3:\n", x_int8_3, scale_3)

    print("shape1:\n", x_int8.shape, scale.shape)
    print("shape2:\n", x_int8_2.shape, scale_2.shape)
    print("shape3:\n", x_int8_3.shape, scale_3.shape)

    error1 = (x_int8 - x_int8_2).float().abs().mean()
    error2 = (x_int8_2 - x_int8_3).float().abs().mean()
    error3 = (x_int8 - x_int8_3).float().abs().mean()
    print("x 平均误差:", error1.item(), error2.item(), error3.item())

    error4 = (scale - scale_2).float().abs().mean()
    error5 = (scale_2 - scale_3).float().abs().mean()
    error6 = (scale - scale_3).float().abs().mean()
    print("scale 平均误差:", error4.item(), error5.item(), error6.item())