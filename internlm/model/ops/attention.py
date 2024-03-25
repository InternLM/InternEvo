"""
attention operators
"""

import math
from functools import singledispatch

import torch
from einops import rearrange, repeat
from torch import nn

from internlm.core.context import global_context as gpc


try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as __flash_varlen_qkvpacked_attn
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as __flash_fixedlen_qkvpacked_attn
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func as __flash_varlen_kvpacked_attn
    from flash_attn.flash_attn_interface import flash_attn_kvpacked_func as __flash_fixedlen_kvpacked_attn
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as __flash_varlen_qkvsplited_attn
    from flash_attn.flash_attn_interface import flash_attn_func as __flash_fixedlen_qkvsplited_attn

    flash_attn_impl = True
except (ModuleNotFoundError, ImportError):
    flash_attn_impl = False

# adpated from https://github.com/Dao-AILab/flash-attention/blob/v2.2.1/flash_attn/modules/mha.py
def __torch_fixedlen_qkvpacked_attn(qkv, dropout, softmax_scale=None, causal=False, key_padding_mask=None):
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
def __torch_fixedlen_kvpacked_attn(q, kv, dropout, softmax_scale=None, causal=False, key_padding_mask=None):
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


def __torch_nyi_attn(*args, **kwargs):
    assert False, "Not yet implemented"


class SelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = nn.Dropout(attention_dropout)

    @singledispatch
    def forward(self, obj: object):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        assert False, "Never arrive here"

    @forward.register
    def _(self, qkv, softmax_scale=None, causal=None, return_attn_probs=False, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_fixedlen_qkvpacked_attn(qkv, self.dropout.p, softmax_scale, causal, return_attn_probs)
        else:
            return __torch_fixedlen_qkvpacked_attn(qkv, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register
    def _(self, q, kv, softmax_scale=None, causal=False, return_attn_probs=False, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal, return_attn_probs)
        else:
            return __torch_fixedlen_kvpacked_attn(q, kv, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register
    def _(self, q, k, v, softmax_scale=None, causal=False, return_attn_probs=False, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal, return_attn_probs)
        else:
            return __torch_nyi_attn(q, k, v, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register
    def _(
        self,
        qkv,
        cu_seqlens,
        max_seqlen,
        softmax_scale=None,
        causal=False,
        return_attn_probs=False,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_varlen_qkvpacked_attn(
                qkv, cu_seqlens, max_seqlen, self.dropout.p, softmax_scale, causal, return_attn_probs
            )
        else:
            return __torch_nyi_attn(qkv, cu_seqlens, max_seqlen, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register
    def _(
        self,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        return_attn_probs=False,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_varlen_kvpacked_attn(
                q,
                kv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout.p,
                softmax_scale,
                causal,
                return_attn_probs,
            )
        else:
            return __torch_nyi_attn(
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

    @forward.register
    def _(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        return_attn_probs=False,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_varlen_qkvsplited_attn(
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
                return_attn_probs,
            )
        else:
            return __torch_nyi_attn(
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
                key_padding_mask,
            )


class CrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    @singledispatch
    def forward(self, obj: object):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        assert False, "Never arrive here"

    @forward.register
    def _(self, q, kv, softmax_scale=None, causal=False, return_attn_probs=False, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_fixedlen_kvpacked_attn(q, kv, self.dropout.p, softmax_scale, causal, return_attn_probs)
        else:
            return __torch_fixedlen_kvpacked_attn(q, kv, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register
    def _(self, q, k, v, softmax_scale=None, causal=False, return_attn_probs=False, key_padding_mask=None):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_fixedlen_qkvsplited_attn(q, k, v, self.dropout.p, softmax_scale, causal, return_attn_probs)
        else:
            return __torch_nyi_attn(q, k, v, self.dropout, softmax_scale, causal, key_padding_mask)

    @forward.register
    def _(
        self,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        return_attn_probs=False,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_varlen_kvpacked_attn(
                q,
                kv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                self.dropout.p,
                softmax_scale,
                causal,
                return_attn_probs,
            )
        else:
            return __torch_nyi_attn(
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

    @forward.register
    def _(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        return_attn_probs=False,
        key_padding_mask=None,
    ):
        softmax_scale = self.softmax_scale if softmax_scale is None else softmax_scale
        causal = self.causal if causal is None else causal

        if gpc.config.model.get("use_flash_attn", False) and flash_attn_impl:
            return __flash_varlen_qkvsplited_attn(
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
                return_attn_probs,
            )
        else:
            return __torch_nyi_attn(
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
                key_padding_mask,
            )
