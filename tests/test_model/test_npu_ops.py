"""
TODO: add NPU CI
"""

import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.model.modules.multi_head_attention import (
    AscendFlashSelfAttention,
    CrossAttention,
    SelfAttention,
)
from internlm.model.utils import try_import_RMSNorm

RMSNorm = try_import_RMSNorm()

HEAD_NUM = 32
HIDDEN_SZIE = 4096
SEQ_LEN = 2048
MICRO_BSZ = 1
HEAD_DIM = HIDDEN_SZIE // HEAD_NUM
VOCAB_SIZE = 32000

MICRO_BSZ_LIST = [1, 2]
DTYPE_LIST = [torch.bfloat16, torch.float16]
NUM_KV_HEAD_LIST = [8, 32]
USE_PADDING = [True, False]

internlm_accelerator = get_accelerator()


def check_mean_and_std(name, out1, out2):
    named1_mean = out1.to(dtype=torch.float64).mean()
    named1_std = out1.to(dtype=torch.float64).std()
    named2_mean = out2.to(dtype=torch.float64).mean()
    named2_std = out2.to(dtype=torch.float64).std()
    check_statistic_equality(name, named1_mean, named2_mean, eq=True, is_mean=True)
    check_statistic_equality(name, named1_std, named2_std, eq=True, is_mean=False)


def check_statistic_equality(name, value1, value2, eq=False, is_mean=True, threshold=1e-9):
    if (abs(value1 - value2) < threshold) ^ eq:
        if eq:
            print(
                f"On {name}, "
                f"we have {'mean' if is_mean else 'std'}s of fa_out "
                f"very {'close' if not eq else 'different'}, "
                f"from :{value1} "
                f"and  :{value2}",
                flush=True,
            )
        else:
            print(
                f"On {name}, "
                f"we have {'mean' if is_mean else 'std'}s of fa_out "
                f"very {'close' if not eq else 'different'}, "
                f"from :{value1} "
                f"and  :{value2}",
                flush=True,
            )


def do_cmp_attn(
    name,
    B,  # pylint: disable=W0613
    S,  # pylint: disable=W0613
    N,
    N_KV,
    q,
    k,
    v,
    dtype,
    attention_mask,  # pylint: disable=W0613
    softmax_scale,
    attention_dropout=0.0,
    **attn_args,  # pylint: disable=W0613
):

    npu_attn_cls = CrossAttention if N != N_KV else SelfAttention
    npu_attn = npu_attn_cls(
        causal=True,
        softmax_scale=softmax_scale,
        attention_dropout=attention_dropout,
    ).to(dtype)
    npu_flash_attn = AscendFlashSelfAttention(
        causal=True,
        softmax_scale=softmax_scale,
        attention_dropout=attention_dropout,
    ).to(dtype)

    if N == N_KV:
        a = npu_attn(torch.concat([q, k, v], dim=2))  # pylint: disable=E1102
    else:
        a = npu_attn(q.squeeze(dim=2), torch.concat([k, v], dim=2))  # pylint: disable=E1102

    b = npu_flash_attn(q=q, k=k, v=v)  # pylint: disable=E1102
    assert not torch.isnan(a).any() and not torch.isnan(b).any()
    assert not torch.isinf(a).any() and not torch.isinf(b).any()
    assert torch.allclose(a.to(torch.float64), b.to(torch.float64), atol=5e-2, rtol=1e-4), f"{name} not pass"


def npu_transform(B, S, N, N_KV, D, dtype, use_padding):
    if use_padding:
        x = torch.LongTensor([[i + 1 if i < S // 2 else 0 for i in range(S)] for _ in range(B)]).npu()  # padding S-1024
    else:
        x = torch.LongTensor([[i + 1 for i in range(S)] for _ in range(B)]).npu()  # no-padiing

    wq = torch.zeros((N * D, N * D), dtype=dtype, device="npu")
    wk = torch.zeros((N_KV * D, N * D), dtype=dtype, device="npu")
    wv = torch.zeros((N_KV * D, N * D), dtype=dtype, device="npu")
    wembed = torch.zeros((VOCAB_SIZE, HIDDEN_SZIE), dtype=dtype, device="npu")

    # It is very important to set appropriate initialization values for parameters so
    # that the values fall within an appropriate precision range to prevent overflow or underflow.
    with torch.no_grad():
        wq = nn.init.normal_(wq.data)
        wk = nn.init.normal_(wk.data)
        wv = nn.init.normal_(wv.data)
        wembed = nn.init.normal_(wembed.data, std=0.02)

    embed_x = F.embedding(x, wembed).to(dtype)
    q = F.linear(embed_x, wq)  # pylint: disable=E1102
    k = F.linear(embed_x, wk)  # pylint: disable=E1102
    v = F.linear(embed_x, wv)  # pylint: disable=E1102

    q = rearrange(q, "b s (one h d) -> b s one h d", b=B, s=S, d=D, one=1)
    k = rearrange(k, "b s (one h d) -> b s one h d", b=B, s=S, d=D, one=1)
    v = rearrange(v, "b s (one h d) -> b s one h d", b=B, s=S, d=D, one=1)

    do_cmp_attn(
        f"B_{B}_S_{S}_N_{N}_N_KV_{N_KV}_D_{D}_{dtype}",
        B,
        S,
        N,
        N_KV,
        q,
        k,
        v,
        dtype,
        None,
        1 / math.sqrt(HIDDEN_SZIE // HEAD_NUM),
    )


@pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
@pytest.mark.parametrize("test_dtype", DTYPE_LIST)
@pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
@pytest.mark.parametrize("use_padding", USE_PADDING)
def test_NPU_fa(micro_bsz, test_dtype, num_kv_head, use_padding):
    print(internlm_accelerator.get_accelerator_backend(), flush=True)
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        npu_transform(micro_bsz, SEQ_LEN, HEAD_NUM, num_kv_head, HIDDEN_SZIE // HEAD_NUM, test_dtype, use_padding)


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_npu_ops.py"])
