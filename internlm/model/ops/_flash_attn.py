# Copyright (c) InternLM. All rights reserved.
import torch

from internlm.accelerator import get_accelerator
from internlm.core.context import global_context as gpc
from internlm.core.parallel.comm import get_offload_manager

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_backward,
        _flash_attn_varlen_forward,
    )

    gpu_flash_attn_impl = True
except (ModuleNotFoundError, ImportError):
    gpu_flash_attn_impl = False

internlm_accelerator = get_accelerator()
device_backend = internlm_accelerator.get_accelerator_backend()


class FlashAttnVarlenKVPackedFunc_V263(torch.autograd.Function):
    """
    Varlen KVPacked Func from Flash Attn v2.6.3.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        layer_idx,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k, v = kv[:, 0], kv[:, 1]

        _ckpt_block_num = int(gpc.config.model.checkpoint * gpc.config.isp_num_layers)

        if gpc.is_forward is False and gpc.config.selective_checkpoint and layer_idx < _ckpt_block_num:
            out, out_padded, softmax_lse, S_dmask, rng_state = get_offload_manager().get_fa_output_with_layer(layer_idx)
        else:
            (
                out,
                q,
                k,
                v,
                out_padded,
                softmax_lse,
                S_dmask,
                rng_state,
            ) = _flash_attn_varlen_forward(  # pylint: disable=E1123
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=return_softmax and dropout_p > 0,
                block_table=None,
            )

        # store attn forward output to avoid re-computation of attn when activation checkpoint is enabled
        if gpc.is_forward and gpc.config.selective_checkpoint and layer_idx < _ckpt_block_num:
            get_offload_manager().insert_fa_output_with_layer(
                layer_idx=layer_idx, output=(out, out_padded, softmax_lse, S_dmask, rng_state)
            )

        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):  # pylint: disable=W0613
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _flash_attn_varlen_backward(  # pylint: disable=E1121,E1124
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenKVPackedFunc_V221(torch.autograd.Function):
    """
    Varlen KVPacked Func from Flash Attn v2.2.1.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
        layer_idx,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k, v = kv[:, 0], kv[:, 1]

        _ckpt_block_num = int(gpc.config.model.checkpoint * gpc.config.isp_num_layers)

        if gpc.is_forward is False and gpc.config.selective_checkpoint and layer_idx < _ckpt_block_num:
            out, out_padded, softmax_lse, S_dmask, rng_state = get_offload_manager().get_fa_output_with_layer(layer_idx)
        else:
            out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                causal=causal,
                return_softmax=return_softmax and dropout_p > 0,
            )

        # store attn forward output to avoid re-computation of attn when activation checkpoint is enabled
        if gpc.is_forward and gpc.config.selective_checkpoint and layer_idx < _ckpt_block_num:
            get_offload_manager().insert_fa_output_with_layer(
                layer_idx=layer_idx, output=(out, out_padded, softmax_lse, S_dmask, rng_state)
            )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):  # pylint: disable=W0613
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dkv[:, 0],
            dkv[:, 1],
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    layer_idx=0,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.
    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.
    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        kv: (total_k, 2, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """

    assert gpu_flash_attn_impl is True and flash_attn.__version__ in [
        "2.2.1",
        "2.6.3",
    ], "flash-attn should be installed and version must be v2.2.1 or v2.6.3"

    if flash_attn.__version__ == "2.2.1":
        return FlashAttnVarlenKVPackedFunc_V221.apply(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            return_attn_probs,
            layer_idx,
        )

    return FlashAttnVarlenKVPackedFunc_V263.apply(
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        layer_idx,
    )
