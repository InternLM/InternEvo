"""
TODO: add NPU CI
"""

import math
import random

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import Config
from internlm.core.context import global_context as gpc
from internlm.model.ops.attention import SelfAttention
from internlm.model.ops.utils import pack_output_after_attn, unpack_qkv_before_attn
from internlm.utils.common import get_current_device, set_random_seed

HEAD_NUM = 32
HIDDEN_SZIE = 4096
SEQ_LEN = [2048, 4096]
HEAD_DIM = HIDDEN_SZIE // HEAD_NUM
VOCAB_SIZE = 32000
NUM_KV_HEAD_LIST = [1, 8, 32]
MICRO_BSZ_LIST = [1, 2]
DTYPE_LIST = [torch.bfloat16, torch.float16]

internlm_accelerator = get_accelerator()


def init_qkv(B, S, N_KV, dtype, device):
    x = torch.LongTensor([[i + 1 for i in range(S)] for _ in range(B)]).to(device)
    cu_seqlens = [0] + sorted(random.sample(list(range(x.numel())), 4))
    if cu_seqlens[-1] != x.numel():
        cu_seqlens.append(x.numel())
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=device)
    x = rearrange(x, "b s -> (b s)").unsqueeze(0)

    KV_DIM = HEAD_DIM * N_KV
    Q_PER_KV = HEAD_NUM // N_KV
    wqkv = torch.rand((HIDDEN_SZIE + 2 * KV_DIM, HIDDEN_SZIE), dtype=dtype, device=device)
    wembed = torch.rand((VOCAB_SIZE, HIDDEN_SZIE), dtype=dtype, device=device)

    # It is very important to set appropriate initialization values for parameters so
    # that the values fall within an appropriate precision range to prevent overflow or underflow.
    with torch.no_grad():
        wqkv.data = nn.init.normal_(wqkv.data)
        wembed = nn.init.normal_(wembed.data, std=0.02)

    embed_x = F.embedding(x, wembed).to(dtype)
    qkv = F.linear(embed_x, wqkv)  # pylint: disable=E1102
    qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=Q_PER_KV + 2, d=HEAD_DIM)
    q, k, v = (qkv[..., :Q_PER_KV, :], qkv[..., -2, :], qkv[..., -1, :])
    q = rearrange(q, "b t h gs d -> b t (h gs) d")
    kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
    return q, kv, cu_seqlens


def fixed_length_fa(q, kv, cu_seqlens, packed_len, attn_cls, use_fa=False):
    q = unpack_qkv_before_attn(q, cu_seqlens)
    kv = unpack_qkv_before_attn(kv, cu_seqlens)
    gpc._config = Config(dict(model=dict(use_flash_attn=use_fa, dtype=torch.bfloat16)))
    c = attn_cls(q=q, kv=kv)  # fix length self attention in npu
    c = rearrange(c, "b s h d -> b s (h d)")
    return pack_output_after_attn(c, cu_seqlens, packed_length=packed_len)


def var_length_fa(q, kv, cu_seqlens, max_seqlen, attn_cls):
    gpc._config = Config(dict(model=dict(use_flash_attn=True, dtype=torch.bfloat16)))
    b = attn_cls(  # pylint: disable=E1102
        q=q,
        kv=kv,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )
    return rearrange(b, "b s h d -> b s (h d)")


def assert_equal(a, b, atol_bf16=5e-2, rtol_bf16=1e-4, atol_fp16=5e-2, rtol_fp16=1e-4):
    assert a.dtype == b.dtype
    assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    if a.dtype is torch.bfloat16:
        assert torch.allclose(a, b, atol=atol_bf16, rtol=rtol_bf16), f"a: {a}, b: {b}"
    elif a.dtype is torch.float16:
        assert torch.allclose(a, b, atol=atol_fp16, rtol=rtol_fp16), f"a: {a}, b: {b}"
    else:
        assert False


def npu_fwd_transform(B, S, N_KV, dtype):

    set_random_seed(1024)
    softmax_scale = 1 / math.sqrt(HEAD_DIM)
    cross_attn = SelfAttention(causal=True, softmax_scale=softmax_scale, attention_dropout=0.0).to(dtype)
    npu_flash_attn = SelfAttention(causal=True, softmax_scale=softmax_scale, attention_dropout=0.0).to(dtype)

    with torch.no_grad():
        q, kv, cu_seqlens = init_qkv(B, S, N_KV, dtype, get_current_device())

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    q, kv = q.requires_grad_(), kv.requires_grad_()
    a = fixed_length_fa(q, kv, cu_seqlens, B * S, cross_attn, use_fa=False)

    q_2, kv_2 = q.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
    b = fixed_length_fa(q_2, kv_2, cu_seqlens, B * S, npu_flash_attn, use_fa=True)

    q_3, kv_3 = q.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
    c = var_length_fa(q_3, kv_3, cu_seqlens, max_seqlen, npu_flash_attn)

    # assert_equal(a, b, atol_bf16=1e-1)
    assert_equal(a, c, atol_bf16=1e-1)
    print("test npu_fwd_transform done!", flush=True)

    return a, b, c, q, q_2, q_3, kv, kv_2, kv_3


def npu_transform(B, S, N_KV, dtype):
    a, b, c, q, q_2, q_3, kv, kv_2, kv_3 = npu_fwd_transform(B, S, N_KV, dtype)  # pylint: disable=W0612
    g = torch.randn_like(b)
    g.uniform_(-2, 2)

    b.backward(g.clone(), retain_graph=True)
    a.backward(g.clone(), retain_graph=True)
    c.backward(g.clone(), retain_graph=True)

    # assert_equal(q.grad, W0612.grad, atol_bf16=1e-1)
    assert_equal(q.grad, q_3.grad, atol_bf16=1e-1)
    # assert_equal(kv.grad, kv_2.grad, atol_bf16=5e-1, rtol_bf16=1e-3)
    assert_equal(kv.grad, kv_3.grad, atol_bf16=5e-1)

    print("test npu_transform done!", flush=True)


def deeplink_fwd_transform(B, S, N_KV, dtype):
    from deeplink_ext.internevo_ops import FlashSelfAttention

    from internlm.model.modules.multi_head_attention import CrossAttention

    set_random_seed(1024)
    softmax_scale = 1 / math.sqrt(HEAD_DIM)
    cross_attn = CrossAttention(causal=True, softmax_scale=softmax_scale, attention_dropout=0.0).to(dtype)
    dp_flash_attn = FlashSelfAttention(causal=True, softmax_scale=softmax_scale, attention_dropout=0.0).to(dtype)

    with torch.no_grad():
        q, kv, cu_seqlens = init_qkv(B, S, N_KV, dtype, get_current_device())

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    q, kv = q.requires_grad_(), kv.requires_grad_()
    a = fixed_length_fa(q, kv, cu_seqlens, B * S, cross_attn)

    q_2, kv_2 = q.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
    b = var_length_fa(q_2, kv_2, cu_seqlens, max_seqlen, dp_flash_attn)

    assert_equal(a, b)
    print("test deeplink_fwd_transform done!", flush=True)

    return a, b, q, q_2, kv, kv_2


def deeplink_transform(B, S, N_KV, dtype):
    a, b, q, q_2, kv, kv_2 = deeplink_fwd_transform(B, S, N_KV, dtype)

    g = torch.randn_like(b)
    g.uniform_(-2, 2)

    b.backward(g.clone(), retain_graph=True)
    a.backward(g.clone(), retain_graph=True)

    assert_equal(q.grad, q_2.grad, atol_bf16=1e-1)
    assert_equal(kv.grad, kv_2.grad, atol_bf16=1e-1)

    print("test deeplink_transform done!", flush=True)


@pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
@pytest.mark.parametrize("test_dtype", DTYPE_LIST)
@pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
@pytest.mark.parametrize("seqlen", SEQ_LEN)
def test_NPU_fa_fwd(micro_bsz, test_dtype, num_kv_head, seqlen):
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        npu_fwd_transform(micro_bsz, seqlen, num_kv_head, test_dtype)


@pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
@pytest.mark.parametrize("test_dtype", DTYPE_LIST)
@pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
@pytest.mark.parametrize("seqlen", SEQ_LEN)
def test_NPU_fa_bwd(micro_bsz, test_dtype, num_kv_head, seqlen):
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        npu_transform(micro_bsz, seqlen, num_kv_head, test_dtype)


# @pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
# @pytest.mark.parametrize("test_dtype", DTYPE_LIST)
# @pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
# def test_deeplink_fa_fwd(micro_bsz, test_dtype, num_kv_head):
#     if internlm_accelerator.get_accelerator_backend() == AcceleratorType.DIPU:
#         deeplink_fwd_transform(micro_bsz, SEQ_LEN, num_kv_head, test_dtype)


# @pytest.mark.parametrize("micro_bsz", MICRO_BSZ_LIST)
# @pytest.mark.parametrize("test_dtype", DTYPE_LIST)
# @pytest.mark.parametrize("num_kv_head", NUM_KV_HEAD_LIST)
# def test_deeplink_fa_bwd(micro_bsz, test_dtype, num_kv_head):
#     if internlm_accelerator.get_accelerator_backend() == AcceleratorType.DIPU:
#         deeplink_transform(micro_bsz, SEQ_LEN, num_kv_head, test_dtype)


if __name__ == "__main__":
    pytest.main(["-s", "-q", "test_npu_ops.py"])
