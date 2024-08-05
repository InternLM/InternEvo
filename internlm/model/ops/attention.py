"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the attention operators.
"""

import math
from typing import Callable, Tuple

import torch
from einops import rearrange, repeat
from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import global_context as gpc
from internlm.core.parallel.comm.isp import (
    auto_wrap_distributed_attention,
    auto_wrap_func_distributed_attention,
)
from internlm.model.ops.utils import pack_output_after_attn, unpack_qkv_before_attn
from internlm.utils.common import get_current_device
from internlm.utils.utils import (
    CuSeqlenType,
    QKVPackType,
    check_attention_argument,
    params_dispatch_with_condition,
)

try:
    from torch_npu import npu_fusion_attention as _origin_npu_fixedlen_qkvsplited_func

    is_torch_npu = True
except (ModuleNotFoundError, ImportError):
    is_torch_npu = False

try:
    # TODO: add support of deeplink
    from deeplink_ext.internevo_ops import FlashCrossAttention, FlashSelfAttention

    del FlashCrossAttention, FlashSelfAttention

    deeplink_flash_attn_impl = True
except (ModuleNotFoundError, ImportError):
    deeplink_flash_attn_impl = False

try:
    from flash_attn.flash_attn_interface import (
        flash_attn_func as _flash_fixedlen_qkvsplited_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_kvpacked_func as _flash_fixedlen_kvpacked_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as _flash_fixedlen_qkvpacked_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as _flash_varlen_qkvsplited_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_kvpacked_func as _flash_varlen_kvpacked_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_varlen_qkvpacked_func as _flash_varlen_qkvpacked_func,
    )

    gpu_flash_attn_impl = True
except (ModuleNotFoundError, ImportError):
    gpu_flash_attn_impl = False

internlm_accelerator = get_accelerator()
device_backend = internlm_accelerator.get_accelerator_backend()


def _nyi_attn(func_name, *args, **kwargs):  # pylint: disable=W0613
    assert False, f"{func_name} is not yet implemented"


# gpu flash attention operators


def _flash_float32_compatibility_wrapper(input_idxs: Tuple, flash_func: Callable, *args, **kwargs):
    if gpc.config.model.dtype is torch.float32:
        inputs = [args[idx] for idx in input_idxs]
        input_dtype = inputs[0].dtype
        other_args = [args[idx] for idx in range(len(inputs), len(args))]

        with internlm_accelerator.amp.autocast(dtype=torch.bfloat16):
            for idx in input_idxs:
                if inputs[idx].dtype is torch.float32:
                    inputs[idx] = inputs[idx].to(torch.bfloat16)
            return flash_func(*inputs, *other_args, **kwargs).to(input_dtype)

    return flash_func(*args, **kwargs)


def _flash_varlen_qkvpacked_attn(
    qkv: torch.Tensor, cu_seqlens, max_seqlen, dropout_p, softmax_scale=None, causal=False
):
    # compatible data format: [1, packelen, 3, n_head, headim]
    qkv = qkv.squeeze(dim=0)

    # input_idxs: 0: qkv
    output = _flash_float32_compatibility_wrapper(
        (0), _flash_varlen_qkvpacked_func, qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal
    )

    return output.unsqueeze(dim=0)


def _flash_fixedlen_qkvpacked_attn(qkv: torch.Tensor, dropout_p=0.0, softmax_scale=None, causal=False):
    # input_idxs: 0: qkv
    return _flash_float32_compatibility_wrapper(
        (0), _flash_fixedlen_qkvpacked_func, qkv, dropout_p, softmax_scale, causal
    )


def _flash_varlen_kvpacked_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    # compatible data format: [1, packelen, 3, n_head, headim]
    q, kv = q.squeeze(dim=0), kv.squeeze(dim=0)

    # input_idxs: 0: q, 1: kv
    output = _flash_float32_compatibility_wrapper(
        (0, 1),
        _flash_varlen_kvpacked_func,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
    )

    return output.unsqueeze(dim=0)


def _flash_fixedlen_kvpacked_attn(q: torch.Tensor, kv: torch.Tensor, dropout_p=0.0, softmax_scale=None, causal=False):
    # input_idxs: 0: q, 1: kv
    return _flash_float32_compatibility_wrapper(
        (0, 1), _flash_fixedlen_kvpacked_func, q, kv, dropout_p, softmax_scale, causal
    )


def _flash_varlen_qkvsplited_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    # compatible data format: [1, packelen, 3, n_head, headim]
    q, k, v = q.squeeze(dim=0), k.squeeze(dim=0), v.squeeze(dim=0)

    # input_idxs: 0: q, 1: k, 2: v
    output = _flash_float32_compatibility_wrapper(
        (0, 1, 2),
        _flash_varlen_qkvsplited_func,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
    )

    return output.unsqueeze(dim=0)


def _flash_fixedlen_qkvsplited_attn(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False):
    # input_idxs: 0: q, 1: k, 2: v
    return _flash_float32_compatibility_wrapper(
        (0, 1, 2), _flash_fixedlen_qkvsplited_func, q, k, v, dropout_p, softmax_scale, causal
    )


# npu flash attention operators
def _npu_varlen_qkvsplited_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,  # pylint: disable=W0613
    max_seqlen_k,  # pylint: disable=W0613
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    return _flash_float32_compatibility_wrapper(
        (0, 1, 2),
        _npu_varlen_qkvsplited_func,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
    )


def _npu_varlen_qkvsplited_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,  # pylint: disable=W0613
    max_seqlen_k,  # pylint: disable=W0613
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    use_fixlen=False,
):
    """Support Huawei Ascend's torch_npu flash attention.
    Tested version:
        torch: 2.1.0+cpu
        torch_npu: 2.1.0.post3+git7c4136d
        cann: 8.0.RC1.alpha003
    """
    packed_length = q.size(dim=1)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])

    if use_fixlen:

        q = unpack_qkv_before_attn(q, cu_seqlens=cu_seqlens_q)
        k = unpack_qkv_before_attn(k, cu_seqlens=cu_seqlens_k)
        v = unpack_qkv_before_attn(v, cu_seqlens=cu_seqlens_k)

        output = _npu_fixedlen_qkvsplited_attn(q, k, v, dropout_p, softmax_scale, causal)

        output = pack_output_after_attn(output, cu_seqlens_q, packed_length)
    else:
        output = _npu_fused_varlen_qkvsplited_attn(
            q, k, v, dropout_p, softmax_scale, causal, max_seqlen_q, max_seqlen_k, cu_seqlens_q, cu_seqlens_k
        )

    return output


def _npu_fixedlen_qkvsplited_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale=None,
    causal=False,
):
    assert causal is True
    assert q.dtype in (torch.bfloat16, torch.float16)

    if len(q.shape) == 5:  # [batch, seqlen, 1, n_head, headdim]
        q, k, v = q.squeeze(dim=2), k.squeeze(dim=2), v.squeeze(dim=2)

    _, seqlen, n_head, _ = q.shape
    sparse_mode = 0
    attention_mask = torch.triu(torch.ones(seqlen, seqlen, device=get_current_device()), 1).bool()

    return _origin_npu_fixedlen_qkvsplited_func(
        query=q,
        key=k,
        value=v,
        head_num=n_head,
        input_layout="BSND",  # If necessary, expose the interface
        pse=None,
        atten_mask=attention_mask,
        scale=softmax_scale,
        sparse_mode=sparse_mode,  # If necessary, expose the interface
        pre_tockens=seqlen,  # Used for sparse calculations, representing the left boundary of the slides window
        next_tockens=0,  # If necessary, expose the interface
        keep_prob=1 - dropout_p,
        inner_precise=0,  # If necessary, expose the interface
    )[0]


def _npu_fused_varlen_qkvsplited_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale=None,
    causal=False,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    deterministic=False,
):
    assert causal is True
    assert q.dtype in (torch.bfloat16, torch.float16)

    if len(q.shape) == 4:  # [1, packedseqlen, n_head, headdim]
        q, k, v = q.squeeze(dim=0), k.squeeze(dim=0), v.squeeze(dim=0)

    S, N = max(max_seqlen_q, max_seqlen_k), q.shape[1]
    device = get_current_device()
    sparse_mode = 0

    if max_seqlen_k > 2048 and max_seqlen_q > 2048:
        sparse_mode = 2
        max_seqlen_k = 2048
        max_seqlen_q = 2048

    attention_mask = torch.triu(torch.ones(max_seqlen_q, max_seqlen_k, device=device), 1).bool()
    cu_seqlens_q = cu_seqlens_q[1:].tolist()
    cu_seqlens_kv = cu_seqlens_kv[1:].tolist()

    return _origin_npu_fixedlen_qkvsplited_func(
        query=q,
        key=k,
        value=v,
        head_num=N,
        input_layout="TND",
        pse=None,
        atten_mask=attention_mask,
        scale=softmax_scale,
        sparse_mode=sparse_mode,
        pre_tockens=S,  # Used for sparse calculations, representing the left boundary of the slides window
        next_tockens=0,
        keep_prob=1 - dropout_p,
        inner_precise=0 if not deterministic else 2,
        actual_seq_kvlen=cu_seqlens_kv,
        actual_seq_qlen=cu_seqlens_q,
    )[0].unsqueeze(dim=0)


def _npu_varlen_qkvpacked_attn(
    qkv: torch.Tensor, cu_seqlens, max_seqlen, dropout_p, softmax_scale=None, causal=False  # pylint: disable=W0613
):
    # TODO: support npu native varlen flash attention
    q, k, v = qkv.unbind(dim=2)
    return _npu_varlen_qkvsplited_attn(q, k, v, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal)


def _npu_fixedlen_qkvpacked_attn(qkv: torch.Tensor, dropout_p: float, softmax_scale=None, causal=False):
    q, k, v = qkv.unbind(dim=2)
    return _npu_fixedlen_qkvsplited_attn(q, k, v, dropout_p, softmax_scale, causal)


def _npu_varlen_kvpacked_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,  # pylint: disable=W0613
    max_seqlen_k,  # pylint: disable=W0613
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
):
    # TODO: support npu native varlen flash attention
    k, v = kv.unbind(dim=2)
    k, v = k.squeeze(dim=2), v.squeeze(dim=2)
    return _npu_varlen_qkvsplited_attn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
    )


def _npu_fixedlen_kvpacked_attn(q: torch.Tensor, kv: torch.Tensor, dropout_p: float, softmax_scale=None, causal=False):
    k, v = kv.unbind(dim=2)
    k, v = k.squeeze(dim=2), v.squeeze(dim=2)
    return _npu_fixedlen_qkvsplited_attn(q, k, v, dropout_p, softmax_scale, causal)


# deeplink flash attention operators


def _deeplink_varlen_qkvpacked_attn(*args, **kwargs):
    # TODO: support deeplink version flash attention
    _nyi_attn("_deeplink_varlen_qkvpacked_attn", *args, **kwargs)


def _deeplink_fixedlne_qkvpacked_attn(*args, **kwargs):
    # TODO: support deeplink version flash attention
    _nyi_attn("_deeplink_fixedlne_qkvpacked_attn", *args, **kwargs)


def _deeplink_varlen_kvpacked_attn(*args, **kwargs):
    # TODO: support deeplink version flash attention
    _nyi_attn("_deeplink_varlen_kvpacked_attn", *args, **kwargs)


def _deeplink_fixedlen_kvpacked_attn(*args, **kwargs):
    # TODO: support deeplink version flash attention
    _nyi_attn("_deeplink_fixedlen_kvpacked_attn", *args, **kwargs)


def _deeplink_varlen_qkvsplited_attn(*args, **kwargs):
    # TODO: support deeplink version flash attention
    _nyi_attn("_deeplink_varlen_qkvsplited_attn", *args, **kwargs)


def _deeplink_fixedlen_qkvsplited_attn(*args, **kwargs):
    # TODO: support deeplink version flash attention
    _nyi_attn("_deeplink_fixedlen_qkvsplited_attn", *args, **kwargs)


# torch attention operators
# adpated from https://github.com/Dao-AILab/flash-attention/blob/v2.2.1/flash_attn/modules/mha.py
def _torch_fixedlen_qkvpacked_attn(qkv: torch.Tensor, dropout, softmax_scale=None, causal=False, key_padding_mask=None):
    batch_size, seqlen = qkv.shape[0], qkv.shape[1]
    q, k, v = qkv.unbind(dim=2)

    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

    if key_padding_mask is not None:
        padding_mask = torch.full((batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device)
        padding_mask.masked_fill_(key_padding_mask, 0.0)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)

    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    attention_drop = dropout(attention)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)

    return output


# adpated from https://github.com/Dao-AILab/flash-attention/blob/v2.2.1/flash_attn/modules/mha.py
def _torch_fixedlen_kvpacked_attn(
    q: torch.Tensor, kv: torch.Tensor, dropout, softmax_scale=None, causal=False, key_padding_mask=None
):
    batch_size, seqlen_q = q.shape[0], q.shape[1]
    seqlen_k = kv.shape[1]

    assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
    if kv.shape[3] != q.shape[2]:  # MQA/GQA
        kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
    k, v = kv.unbind(dim=2)
    softmax_scale = softmax_scale or 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if key_padding_mask is not None:
        padding_mask = torch.full((batch_size, seqlen_k), -10000.0, dtype=scores.dtype, device=scores.device)
        padding_mask.masked_fill_(key_padding_mask, 0.0)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

    if causal:
        # causal mask needs to take into account the difference between seqlen_q and seqlen_k
        row_idx = rearrange(torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
        sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        causal_mask = col_idx > row_idx + sk - seqlen_q
        scores = scores.masked_fill(causal_mask, -10000.0)

    attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
    attention_drop = dropout(attention)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)

    return output


def _torch_fixedlen_qkvsplited_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout, softmax_scale=None, causal=False, key_padding_mask=None
):
    kv = torch.stack([k, v], dim=2)
    return _torch_fixedlen_kvpacked_attn(q, kv, dropout, softmax_scale, causal, key_padding_mask)


def _torch_varlen_qkvsplited_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,  # pylint: disable=W0613
    max_seqlen_k,  # pylint: disable=W0613
    dropout,
    softmax_scale=None,
    causal=False,
    key_padding_mask=None,
):
    kv = torch.stack([k, v], dim=2)
    packed_length = q.size(dim=1)

    q = unpack_qkv_before_attn(q, cu_seqlens=cu_seqlens_q)
    kv = unpack_qkv_before_attn(kv, cu_seqlens=cu_seqlens_k)

    output = _torch_fixedlen_kvpacked_attn(q, kv, dropout, softmax_scale, causal, key_padding_mask)

    return pack_output_after_attn(output, cu_seqlens_q, packed_length)


def _torch_varlen_qkvpacked_attn(
    qkv: torch.Tensor,
    cu_seqlens,
    max_seqlen,  # pylint: disable=W0613
    dropout,
    softmax_scale=None,
    causal=False,
    key_padding_mask=None,
):

    packed_length = qkv.size(dim=1)
    qkv = unpack_qkv_before_attn(qkv, cu_seqlens=cu_seqlens)

    output = _torch_fixedlen_qkvpacked_attn(qkv, dropout, softmax_scale, causal, key_padding_mask)

    return pack_output_after_attn(output, cu_seqlens, packed_length)


def _torch_varlen_kvpacked_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,  # pylint: disable=W0613
    max_seqlen_k,  # pylint: disable=W0613
    dropout,
    softmax_scale=None,
    causal=False,
    key_padding_mask=None,
):

    packed_length = q.size(dim=1)

    q = unpack_qkv_before_attn(q, cu_seqlens=cu_seqlens_q)
    kv = unpack_qkv_before_attn(kv, cu_seqlens=cu_seqlens_k)

    output = _torch_fixedlen_kvpacked_attn(q, kv, dropout, softmax_scale, causal, key_padding_mask)

    return pack_output_after_attn(output, cu_seqlens_q, packed_length)


@auto_wrap_distributed_attention
class SelfAttention(nn.Module):
    """Implements scaled dot-product attention with optional softmax scaling.

    This class implements the scaled dot-product attention mechanism, which can be optionally scaled
    by a softmax scaling factor. It supports configurations for causal attention and applies dropout
    to the attention scores.

    Arguments:
        causal (bool): If True, applies causal attention to mask future tokens. Defaults to False.
        softmax_scale (Optional[float]): Scaling factor for attention scores before applying softmax.
            Defaults to 1/sqrt(d_keys) where d_keys is the dimension of the keys, computed at runtime.
        attention_dropout (float): Dropout rate for attention scores. Defaults to 0.0.
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = nn.Dropout(attention_dropout)

        if device_backend == AcceleratorType.NPU:
            assert self.causal, "Ascend flash attention does not spport causal=False yet!"

    @params_dispatch_with_condition(condition=check_attention_argument)
    def forward(self):
        """Placeholder for multihead softmax attention implementation.

        This method serves as a placeholder and should not be reached during execution. It is expected
        to be overridden by specific implementations for different attention mechanisms.

        Raises:
            AssertionError: Always raised to indicate the method should not be called directly.
        """
        assert False, "Never arrive here"

    @forward.register(conditions=(str(QKVPackType.QKVPACKED), str(CuSeqlenType.WithOut)))
    def _qkv_without_cu_seqlens(self, qkv, softmax_scale=None, causal=None, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_fixedlen_qkvpacked_attn(qkv, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_fixedlen_qkvpacked_attn(qkv, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_fixedlne_qkvpacked_attn(qkv, self.dropout.p, softmax_scale, causal)
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_fixedlen_qkvpacked_attn(qkv, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.WithOut)))
    def _q_kv_without_cu_seqlens(self, q, kv, softmax_scale=None, causal=None, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal)
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_fixedlen_kvpacked_attn(q, kv, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.WithOut)))
    def _q_k_v_without_cu_seqlens(self, q, k, v, softmax_scale=None, causal=None, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal)
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_fixedlen_qkvsplited_attn(q, k, v, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register(conditions=(str(QKVPackType.QKVPACKED), str(CuSeqlenType.With)))
    def _qkv_with_cu_seqlens(
        self,
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale=None,
        causal=None,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_varlen_qkvpacked_attn(qkv, cu_seqlens, max_seqlen, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_varlen_qkvpacked_attn(qkv, cu_seqlens, max_seqlen, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_varlen_qkvpacked_attn(
                    qkv, cu_seqlens, max_seqlen, self.dropout.p, softmax_scale, causal
                )
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_varlen_qkvpacked_attn(
                qkv, cu_seqlens, max_seqlen, self.dropout, softmax_scale, causal, key_padding_mask
            )

    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.With)))
    def _q_kv_with_cu_seqlens(
        self,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=None,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_varlen_kvpacked_attn(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, self.dropout.p, softmax_scale, causal
                )
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_varlen_kvpacked_attn(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, self.dropout.p, softmax_scale, causal
                )
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_varlen_kvpacked_attn(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, self.dropout.p, softmax_scale, causal
                )
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_varlen_kvpacked_attn(
                q,
                kv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout,
                softmax_scale,
                causal,
                key_padding_mask,
            )

    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.With)))
    def _q_k_v_with_cu_seqlens(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=None,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_varlen_qkvsplited_attn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.dropout.p,
                    softmax_scale,
                    causal,
                )
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_varlen_qkvsplited_attn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.dropout.p,
                    softmax_scale,
                    causal,
                )
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_varlen_qkvsplited_attn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.dropout.p,
                    softmax_scale,
                    causal,
                )
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_varlen_qkvsplited_attn(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout,
                softmax_scale,
                causal,
                key_padding_mask,
            )


@auto_wrap_distributed_attention
class CrossAttention(nn.Module):
    """Implements scaled dot product attention with softmax.

    This class provides the functionality for cross attention mechanism using scaled dot product attention
    with optional softmax scaling and dropout for attention weights.

    Arguments:
        causal (bool): If True, applies causality to prevent tokens from attending to future tokens. Default is False.
        softmax_scale (float, optional): The scaling factor to apply to the dot products before softmax. If None,
            it defaults to 1/sqrt(d_keys) where d_keys is the dimension of the keys, computed at runtime.
        attention_dropout (float): The dropout rate to apply to the attention.

    Raises:
        AssertionError: If `device_backend` is NPU and `causal` is False, since Ascend flash attention does not
            support non-causal attention yet.
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = nn.Dropout(attention_dropout)

        if device_backend == AcceleratorType.NPU:
            assert self.causal, "Ascend flash attention does not support causal=False yet!"

    @params_dispatch_with_condition(condition=check_attention_argument)
    def forward(self):
        """Placeholder for cross attention implementation.

        This method is a placeholder and should not be reached in execution as it is expected to be
        overridden by specific implementations for different attention parameters.

        Raises:
            AssertionError: Always raised to indicate the method should not be called directly.
        """
        assert False, "Never arrive here"

    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.WithOut)))
    def _q_kv_without_cu_seqlens(self, q, kv, softmax_scale=None, causal=None, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal)
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_fixedlen_kvpacked_attn(q, kv, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.WithOut)))
    def _q_k_v_without_cu_seqlens(self, q, k, v, softmax_scale=None, causal=None, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal)
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal)
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_fixedlen_qkvsplited_attn(q, k, v, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register(conditions=(str(QKVPackType.KVPACKED), str(CuSeqlenType.With)))
    def _q_kv_with_cu_seqlens(
        self,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=None,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_varlen_kvpacked_attn(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, self.dropout.p, softmax_scale, causal
                )
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_varlen_kvpacked_attn(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, self.dropout.p, softmax_scale, causal
                )
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_varlen_kvpacked_attn(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, self.dropout.p, softmax_scale, causal
                )
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_varlen_kvpacked_attn(
                q,
                kv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout,
                softmax_scale,
                causal,
                key_padding_mask,
            )

    @forward.register(conditions=(str(QKVPackType.QKVSPLITED), str(CuSeqlenType.With)))
    def _q_k_v_with_cu_seqlens(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=None,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False):
            if device_backend == AcceleratorType.GPU and gpu_flash_attn_impl:
                return _flash_varlen_qkvsplited_attn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.dropout.p,
                    softmax_scale,
                    causal,
                )
            elif device_backend == AcceleratorType.NPU and is_torch_npu:
                return _npu_varlen_qkvsplited_attn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.dropout.p,
                    softmax_scale,
                    causal,
                )
            elif device_backend == AcceleratorType.DIPU and deeplink_flash_attn_impl:
                return _deeplink_varlen_qkvsplited_attn(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.dropout.p,
                    softmax_scale,
                    causal,
                )
            else:
                raise NotImplementedError(f"Unsupported device type: {device_backend} for flash attention")
        else:
            return _torch_varlen_qkvsplited_attn(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout,
                softmax_scale,
                causal,
                key_padding_mask,
            )


@auto_wrap_func_distributed_attention
def hf_q_k_v_without_cu_seqlens(
    query_states,
    key_states,
    value_states,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
):
    attn_output = _flash_fixedlen_qkvsplited_func(  # TODO: currently only support GPU environment
        query_states, key_states, value_states, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal
    )
    return attn_output


@auto_wrap_func_distributed_attention
def hf_q_k_v_with_cu_seqlens(
    query_states,
    key_states,
    value_states,
    cumulative_len,
    max_seqlen,
    dropout_p=0.0,
    causal=True,
):
    q_unpad, k_unpad, v_unpad = query_states.flatten(0, 1), key_states.flatten(0, 1), value_states.flatten(0, 1)
    attn_output = _flash_varlen_qkvsplited_func(  # TODO: currently only support GPU environment
        q_unpad,
        k_unpad,
        v_unpad,
        cumulative_len,
        cumulative_len,
        max_seqlen,
        max_seqlen,
        dropout_p=dropout_p,
        return_attn_probs=False,
        causal=causal,
    )
    attn_output = attn_output.unsqueeze(0)
    return attn_output
