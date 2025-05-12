import torch
from torch.nn.functional import linear

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


if __name__ == "__main__":
    torch.manual_seed(1024)
    x = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    weight = torch.randn(4096, 2048, dtype=torch.float32, device="cuda")

    sr = True
    x_int8, row_scale = per_row_quantize_int8(x, sr)
    w_int8, col_scale = per_col_quantize_int8(weight, sr)

    out1 = x @ weight
    out2 = linear(x, weight.t())
    out3 = scaled_int8_mm_cuda(x_int8, w_int8, row_scale, col_scale)
    out4 = scaled_int8_mm(x_int8, w_int8, row_scale, col_scale)
    out5 = torch.matmul(x, weight)
    assert torch.equal(out1, out2)
    assert torch.equal(out1, out5)
    assert torch.equal(out3, out4)
    print(out1.dtype, out2.dtype, out3.dtype, out4.dtype)
    print(out1.device, out2.device, out3.device, out4.device)
    print(out1.shape, out2.shape, out3.shape, out4.shape)

    error = (out3 - out1).abs().mean()
    print("fp32 tensor:\n", out1)
    print("int8 tensor:\n", out3)
    print("平均误差:", error.item())




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