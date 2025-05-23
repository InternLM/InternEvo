import torch
from torch.nn.functional import linear
from internlm.core.context import global_context as gpc

try:
    from torchao.prototype.quantized_training.int8_mm import scaled_int8_mm, scaled_int8_mm_cuda
except (ModuleNotFoundError, ImportError):
    print('torchao not found', flush=True)

def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    floor_x = x.floor()                  # ⬅️ 得到整数部分 ⌊x⌋
    frac_x = x - floor_x                 # ⬅️ 得到小数部分 δ = x - ⌊x⌋
    rand = torch.rand_like(x)           # ⬅️ [0, 1) 均匀分布
    return torch.where(rand < frac_x, floor_x + 1, floor_x)


def per_row_quantize_int8(x: torch.Tensor, sr=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    x_amax = x.abs().float().amax(dim=1).clamp(min=1e-4)
    scale = 127 / x_amax  # int8范围是 [-127,127]（保留0）
    if sr:
        x_scaled = stochastic_round((x * scale[:, None])).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale[:, None]).round().clamp(-127, 127).to(torch.int8)

    return x_scaled, x_amax / 127


def per_col_quantize_int8(x: torch.Tensor, sr=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    x_amax = x.abs().float().amax(dim=0).clamp(min=1e-4)
    scale = 127 / x_amax  # int8范围是 [-127,127]（保留0）
    if sr:
        x_scaled = stochastic_round((x * scale[None, :])).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale[None, :]).round().clamp(-127, 127).to(torch.int8)

    return x_scaled, x_amax / 127

def per_token_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, row_scale: torch.Tensor, col_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B) * col_scale.view(-1) * row_scale.view(-1, 1)


# scale in float32
def per_tensor_quantize_int8(x: torch.Tensor, sr=True) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    x_amax = x.abs().float().amax().clamp(min=1e-4)
    scale = 127 / x_amax
    if sr:
        x_scaled = stochastic_round(x * scale).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale).round().clamp(-127, 127).to(torch.int8)

    return x_scaled, x_amax / 127

def per_tensor_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B) * a_scale * b_scale


def per_token_tensor_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, row_scale: torch.Tensor, b_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B)  * b_scale * row_scale.view(-1, 1)

def per_tensor_token_scaled_int8_mm(
    A: torch.Tensor, B: torch.Tensor, a_scale: torch.Tensor, col_scale: torch.Tensor
) -> torch.Tensor:
    return torch._int_mm(A, B)  * col_scale.view(-1) * a_scale


def ceil_div(a, b):
    return (a + b - 1) // b


def pad_for_128x128_if_needed(x: torch.Tensor):
    assert x.dim() == 2

    if x.shape[0] % 128 == 0 and x.shape[1] % 128 == 0:
        return x

    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    return x_padded


def per_group_row_quantize_int8(x: torch.Tensor, sr=True, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    if gpc.config.int8_pad:
        x = pad_for_128x128_if_needed(x)
    assert x.size(1) % group_size == 0, f"{x.shape}"
    num_groups = x.size(1) // group_size
    x_amax = x.abs().float().reshape(x.size(0), num_groups, group_size).amax(dim=2).clamp(min=1e-4)
    scale = 127 / x_amax
    scale = scale.unsqueeze(2).expand(-1, -1, group_size).reshape(x.size(0), x.size(1))
    if sr:
        x_scaled = stochastic_round((x * scale)).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale).round().clamp(-127, 127).to(torch.int8)
        
    return x_scaled, x_amax / 127


def per_group_col_quantize_int8(x: torch.Tensor, sr=True, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    num_groups = x.size(0) // group_size
    x_amax = x.abs().float().reshape(num_groups, group_size, x.size(1)).amax(dim=1).clamp(min=1e-4)
    scale = 127 / x_amax
    scale = scale.unsqueeze(1).expand(-1, group_size, -1).reshape(x.size(0), x.size(1))

    if sr:
        x_scaled = stochastic_round((x * scale)).clamp(-127, 127).to(torch.int8)
    else:
        x_scaled = (x * scale).round().clamp(-127, 127).to(torch.int8)
        
    return x_scaled, x_amax / 127

def per_block_quantize_int8(x: torch.Tensor, sr: bool = True, row_group_size: int = 128, col_group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2, f"{x.dim()}, {x.shape}"
    if gpc.config.int8_pad:
        x = pad_for_128x128_if_needed(x)
    assert x.size(0) % row_group_size == 0 and x.size(1) % col_group_size == 0, f"{x.shape}"
    M, K = x.shape
    num_row_groups = M // row_group_size
    num_col_groups = K // col_group_size

    x_blocks = x.float().reshape(num_row_groups, row_group_size, num_col_groups, col_group_size)
    block_amax = x_blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)  # shape: [num_row_groups, num_col_groups]
    scale = 127.0 / block_amax  # shape: [num_row_groups, num_col_groups]
    scale_expanded = scale.unsqueeze(1).unsqueeze(3).expand(-1, row_group_size, -1, col_group_size).reshape(M, K)

    if sr:
        x_q = stochastic_round(x * scale_expanded).clamp(-127, 127).to(torch.int8)
    else:
        x_q = (x * scale_expanded).round().clamp(-127, 127).to(torch.int8)

    inv_scale = block_amax / 127.0 
    return x_q, inv_scale


def per_group_scaled_int8_mm(
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    a_inv_scale: torch.Tensor,
    b_inv_scale: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    # A_q: [M, K], B_q: [K, N]
    M, K = A_q.shape
    _, N = B_q.shape
    G = K // group_size
    # accumulate results
    out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)
    # perform per-group int8 matmul with scaling
    for g in range(G):
        i0, i1 = g * group_size, (g + 1) * group_size
        # partial product
        part = torch._int_mm(A_q[:, i0:i1], B_q[i0:i1, :])
        # apply inverse scales per group
        # a_inv_scale[g] is shape [M], b_inv_scale[g] is shape [N]
        scales = a_inv_scale[:, g].unsqueeze(1) * b_inv_scale[g, :].unsqueeze(0)
        out += part.to(torch.float32) * scales
    return out


def per_rowgroup_block_scaled_int8_mm(
    A_q: torch.Tensor,  # [M, K]
    B_q: torch.Tensor,  # [K, N]
    A_inv_scale: torch.Tensor,  # [M, K // group_size]
    B_inv_scale: torch.Tensor,  # [K // row_group_size, N // col_group_size]
    row_group_size: int = 128,
    col_group_size: int = 128,
    group_size: int = 128
) -> torch.Tensor:
    M, K = A_q.shape
    _, N = B_q.shape
    out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)

    num_k_groups = K // group_size
    num_b_row_groups = K // row_group_size
    num_b_col_groups = N // col_group_size

    for g in range(num_k_groups):
        i0 = g * group_size
        i1 = (g + 1) * group_size
        A_part = A_q[:, i0:i1]  # [M, group_size]
        B_part = B_q[i0:i1, :]  # [group_size, N]
        partial = torch._int_mm(A_part, B_part)  # [M, N]
        # A_inv_scale: [M, num_k_groups] → [M, 1]
        a_scale = A_inv_scale[:, g].unsqueeze(1)  # [M, 1]

        for c in range(num_b_col_groups):
            j0 = c * col_group_size
            j1 = (c + 1) * col_group_size
            # y 的行 block 是第 g 个（由 i0 决定）
            b_scale = B_inv_scale[g, c].unsqueeze(0)  # [1,]
            scale = a_scale * b_scale  # [M, 1] * [1,] = broadcast 到 [M, col_group_size]
            out[:, j0:j1] += partial[:, j0:j1].float() * scale

    return out

def per_block_scaled_int8_mm(
    A_q: torch.Tensor,  # [M, K]
    B_q: torch.Tensor,  # [K, N]
    A_inv_scales: torch.Tensor,  # [M // R, K // C]
    B_inv_scales: torch.Tensor,  # [K // R, N // C]
    row_group_size: int = 128,
    col_group_size: int = 128
) -> torch.Tensor:
    M, K = A_q.shape
    _, N = B_q.shape
    assert K % col_group_size == 0 and M % row_group_size == 0 and N % col_group_size == 0

    out = torch.zeros((M, N), dtype=torch.float32, device=A_q.device)

    num_row_blocks = M // row_group_size
    num_col_blocks = N // col_group_size
    num_k_blocks = K // col_group_size

    for r in range(num_row_blocks):
        for c in range(num_col_blocks):
            out_block = torch.zeros((row_group_size, col_group_size), dtype=torch.float32, device=A_q.device)
            for k in range(num_k_blocks):
                A_block = A_q[r*row_group_size:(r+1)*row_group_size, k*col_group_size:(k+1)*col_group_size]
                B_block = B_q[k*col_group_size:(k+1)*col_group_size, c*col_group_size:(c+1)*col_group_size]
                # int8 matmul
                partial = torch._int_mm(A_block, B_block)  # [row_group_size, col_group_size]
                a_scale = A_inv_scales[r, k]
                b_scale = B_inv_scales[k, c]
                scale = a_scale * b_scale
                out_block += partial.float() * scale

            out[r*row_group_size:(r+1)*row_group_size, c*col_group_size:(c+1)*col_group_size] = out_block
    return out


if __name__ == "__main__":
    torch.manual_seed(1024)
    x = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    weight = torch.randn(4096, 2048, dtype=torch.float32, device="cuda")

    sr = True
    x_int8, row_scale = per_row_quantize_int8(x, sr)
    w_int8, col_scale = per_col_quantize_int8(weight, sr)
    tensor_scaled_x_int8, a_scale = per_tensor_quantize_int8(x, sr)
    tensor_scaled_w_int8, b_scale = per_tensor_quantize_int8(weight, sr)
    group_scaled_x_int8, group_row_scale = per_group_row_quantize_int8(x, sr)
    group_scaled_w_int8, group_col_scale = per_group_col_quantize_int8(weight, sr)
    block_scaled_x_int8, block_x_scale = per_block_quantize_int8(x, sr)
    block_scaled_w_int8, block_w_scale = per_block_quantize_int8(weight, sr)

    out1 = x @ weight
    out2 = linear(x, weight.t())
    out3 = scaled_int8_mm_cuda(x_int8, w_int8, row_scale, col_scale)
    out4 = scaled_int8_mm(x_int8, w_int8, row_scale, col_scale)
    out5 = torch.matmul(x, weight)
    out6 = per_token_scaled_int8_mm(x_int8, w_int8, row_scale, col_scale)
    out7 = per_tensor_scaled_int8_mm(tensor_scaled_x_int8, tensor_scaled_w_int8, a_scale, b_scale)
    out8 = per_group_scaled_int8_mm(group_scaled_x_int8, group_scaled_w_int8, group_row_scale, group_col_scale)
    out9 = per_token_tensor_scaled_int8_mm(x_int8, tensor_scaled_w_int8, row_scale, b_scale)
    out10 = per_tensor_token_scaled_int8_mm(tensor_scaled_x_int8, w_int8, a_scale, col_scale)
    out11 = per_rowgroup_block_scaled_int8_mm(group_scaled_x_int8, block_scaled_w_int8, group_row_scale, block_w_scale)
    out12 = per_block_scaled_int8_mm(block_scaled_x_int8, block_scaled_w_int8, block_x_scale, block_w_scale)
    assert torch.equal(out1, out2)
    assert torch.equal(out1, out5)
    assert torch.equal(out3, out4)
    # assert torch.equal(out3, out6)
    print(out1.dtype, out2.dtype, out3.dtype, out4.dtype, out6.dtype)
    print(out1.device, out2.device, out3.device, out4.device, out6.device)
    print(out1.shape, out2.shape, out3.shape, out4.shape, out6.shape)

    error = (out3 - out1).abs().mean()
    print("fp32 tensor:\n", out1)
    print("int8 tensor:\n", out3)
    print("torch_int_mm token scaled int8 tensor:\n", out6)
    print("tensor scaled int8 tensor:\n", out7)
    print("group scaled int8 tensor:\n", out8)
    print("fp32 torchao token scaled int8 平均误差:", error.item())
    # print("token scaled int8 实现误差:", (out3 - out6).abs().mean())
    print("tensor scaled int8 误差:", (out7 - out1).abs().mean())
    # print("tensor-token scaled int8 误差:", (out3 - out7).abs().mean())
    print("group scaled int8 误差:", (out8 - out1).abs().mean())
    # print("group-token scaled int8 误差:", (out8 - out3).abs().mean())
    print("token tensor scaled int8 误差:", (out9 - out1).abs().mean())
    print("tensor token scaled int8 误差:", (out10 - out1).abs().mean())
    print("rowgroup block scaled int8 误差:", (out11 - out1).abs().mean())
    print("block scaled int8 误差:", (out12 - out1).abs().mean())



# torch.manual_seed(1024)
# x = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
# weight = torch.randn(4096, 2048, dtype=torch.float32, device="cuda")

# x_bf16 = x.to(torch.bfloat16)
# weight_bf16 = weight.to(torch.bfloat16)

# out1 = linear(x, weight.t())
# out2 = linear(x_bf16, weight_bf16.t())
# out3 = out1.to(torch.bfloat16)

# print(out1.dtype, out1)
# print(out3.dtype, out3)
# error = (out1 - out3).abs().mean()
# print("平均误差:", error.item())




# a = torch.tensor([[1,2,3], [3,4,5]]).float()
# b = torch.tensor([[1,2], [3,4], [5, 6]]).float()
# out1 = a @ b

# c, d = per_row_quantize_int8(a)
# e, f = per_col_quantize_int8(b)
# out2 = (c @ e).to(torch.float32) * d[:, None] * f[None, :]
# print(c.dtype, d.dtype)
# print(out1.dtype, out2.dtype)
# print(out1, out2)


# print(a)
# c, d = per_row_quantize_int8(a)
# print(c, d)
# print(c.dtype, d.dtype)
# print(c.float() * d[:, None])
# print()
# c, d = per_col_quantize_int8(a)
# print(c, d)
# print(c.dtype, d.dtype)
# print(c.float() * d[None, :])