import pytest
import torch
from torch import nn

from internlm.accelerator import get_accelerator
from internlm.model.ops.rotary_emb import (
    ApplyRotaryEmb,
    rotary_emb_in_rotate_half_style,
)
from internlm.utils.common import get_current_device

internlm_accelerator = get_accelerator()


MICRO_BSZ_LIST = [1, 2]
DTYPE_LIST = [torch.bfloat16, torch.float16]
INTERLEAVED = [True, False]


def npu_rope_fwd(B, dtype, interleaved, H=128, N=32, S=4096, rope_base=10000):
    device = get_current_device()
    # qkv = torch.randn((B, S, 3, N, H), dtype=dtype, device=device)
    q = torch.randn((B, S, N, H), dtype=dtype, device=device)

    q = nn.init.normal_(q, mean=0.0, std=1.0)

    inv_freq = 1.0 / (rope_base ** (torch.arange(0, H, 2, device=device, dtype=torch.float32) / H))
    t = torch.arange(S, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq.to(device=t.device))
    cos, sin = torch.cos(freqs), torch.sin(freqs)

    # Test normal torch.
    out1 = ApplyRotaryEmb.apply(q.clone(), cos.clone(), sin.clone(), interleaved, False)

    # Test rotate_half torch.
    out2 = rotary_emb_in_rotate_half_style(
        x=q.clone(), cos=cos.clone(), sin=sin.clone(), interleaved=interleaved, use_fused_rope=False
    )

    # Test rotate_half torch_npu fused.
    out3 = rotary_emb_in_rotate_half_style(
        x=q.clone(), cos=cos.clone(), sin=sin.clone(), interleaved=interleaved, use_fused_rope=True
    )

    assert torch.allclose(out1, out2, rtol=1e-4, atol=1e-5)
    assert torch.allclose(out2, out3, rtol=1e-4, atol=1e-5)
    assert torch.allclose(out1, out3, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
@pytest.mark.parametrize("test_dtype", DTYPE_LIST)
@pytest.mark.parametrize("interleaved", INTERLEAVED)
def test_NPU_fa(micro_bsz, test_dtype, interleaved):
    npu_rope_fwd(B=micro_bsz, dtype=test_dtype, interleaved=interleaved)
