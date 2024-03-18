#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from ..utils import gather_forward_split_backward, split_forward_gather_backward


class Embedding1D(nn.Module):
    """
    1D Embedding.

    Args:
        num_embeddings (int): The size of vocab.
        embedding_dim (int): The dimention of model.
        padding_idx (int): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                            therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                            i.e. it remains as a fixed "pad". None by default.
        dtype (Optional[torch.dtype]): Data type None by default.

    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *args,
        padding_idx: int = None,
        dtype: torch.dtype = None,
        **kwargs,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embed_dim = embedding_dim
        embed_dim_per_partition = embedding_dim // gpc.tensor_parallel_size

        self.padding_idx = padding_idx
        self.embed_args = args
        self.embed_kwargs = kwargs

        self.weight = nn.Parameter(torch.empty((num_embeddings, embed_dim_per_partition), dtype=dtype))

    def forward(self, input_: Tensor) -> Tensor:
        output_parallel = F.embedding(input_, self.weight, self.padding_idx, *self.embed_args, **self.embed_kwargs)

        output = gather_forward_split_backward(output_parallel, ParallelMode.TENSOR, dim=-1)

        if gpc.config.parallel.sequence_parallel:
            output = split_forward_gather_backward(output, ParallelMode.TENSOR, dim=1)

        return output


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim) or (total, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


def apply_rotary_torch(
    x,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    x_pt = x.detach().clone().requires_grad_()

    is_varlen = cu_seqlens is not None
    if not is_varlen:
        seqlen = x.shape[1]
        if isinstance(seqlen_offsets, torch.Tensor):
            batch_size = seqlen_offsets.shape[0]
            arange = rearrange(torch.arange(seqlen, device=cos.device), "s -> 1 s")
            idx = rearrange(seqlen_offsets, "b -> b 1") + arange
            cos_pt = rearrange(cos[idx.flatten()], "(b s) d -> b s d", b=batch_size)
            sin_pt = rearrange(sin[idx.flatten()], "(b s) d -> b s d", b=batch_size)
        else:
            cos_pt = cos[seqlen_offsets : seqlen_offsets + seqlen]
            sin_pt = sin[seqlen_offsets : seqlen_offsets + seqlen]
    else:
        cos_pt = cos
        sin_pt = sin

    output = apply_rotary_emb_torch(x_pt, cos_pt, sin_pt, interleaved)
    return output


def apply_rotary_packed_torch(x1, x2, cos, sin, conj):
    assert x1.device == x2.device == cos.device == sin.device, "All inputs must be on the same device"
    assert x1.dtype == x2.dtype == cos.dtype == sin.dtype, "All inputs must have the same dtype"
    assert x1.size() == x2.size(), "Input x1 and x2 must have the same sizes"
    assert cos.size() == sin.size(), "Input cos and sin must have the same sizes"

    if conj:
        out1 = x1 * cos + x2 * sin
        out2 = -x1 * sin + x2 * cos
    else:
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

    return out1, out2


class ApplyRotaryEmb(torch.autograd.Function):
    """
    ApplyRotaryEmb
    """

    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        if gpc.config.model.use_flash_attn:
            from flash_attn.ops.triton.rotary import apply_rotary

            out = apply_rotary(
                x,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=interleaved,
                inplace=inplace,
            )
        else:
            out = apply_rotary_torch(x, cos, sin, interleaved, seqlen_offsets, cu_seqlens)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.cu_seqlens = cu_seqlens
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        if gpc.config.model.use_flash_attn:
            from flash_attn.ops.triton.rotary import apply_rotary

            dx = apply_rotary(
                do,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=ctx.interleaved,
                inplace=ctx.inplace,
                conjugate=True,
            )
        else:
            dx = apply_rotary_torch(do, cos, sin, ctx.interleaved, seqlen_offsets, cu_seqlens)
        return dx, None, None, None, None, None, None, None


apply_rotary_emb = ApplyRotaryEmb.apply


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
    ):
        """
            qkv: (batch_size, seqlen, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        three = qkv.shape[2]
        assert three == 3

        if gpc.config.model.use_flash_attn:
            from flash_attn.ops.triton.rotary import apply_rotary

        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
            if gpc.config.model.use_flash_attn:
                apply_rotary(qk, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
            else:
                qk = apply_rotary_torch(qk, cos, sin, interleaved, seqlen_offsets)
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            if gpc.config.model.use_flash_attn:
                apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
                apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
            else:
                q = apply_rotary_torch(q, cos, sin, interleaved, seqlen_offsets)
                k = apply_rotary_torch(k, cos_k, sin_k, interleaved, seqlen_offsets)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors

        if gpc.config.model.use_flash_attn:
            from flash_attn.ops.triton.rotary import apply_rotary

        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
            if gpc.config.model.use_flash_attn:
                apply_rotary(
                    dqk,
                    cos,
                    sin,
                    seqlen_offsets=seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
            else:
                dqk = apply_rotary_torch(dqk, cos, sin, ctx.interleaved, seqlen_offsets)
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            if gpc.config.model.use_flash_attn:
                apply_rotary(
                    dq,
                    cos,
                    sin,
                    seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
                apply_rotary(
                    dk,
                    cos_k,
                    sin_k,
                    seqlen_offsets,
                    interleaved=ctx.interleaved,
                    inplace=True,
                    conjugate=True,
                )
            else:
                dq = apply_rotary_torch(dq, cos, sin, ctx.interleaved, seqlen_offsets)
                dk = apply_rotary_torch(dk, cos_k, sin_k, ctx.interleaved, seqlen_offsets)
        return dqkv, None, None, None, None, None, None


apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply


class ApplyRotaryEmbPackedQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbPackedQKV_

    Currently, packed qkv calculation is not supported in flash attention,
    a CUDA memory access error may occur in rotary_kernel function. Therefore,
    we need this class here, still using the implemention of flash attention v1.0.
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None):
        """
            qkv: (total, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        _, three, _, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q1, q2 = qkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        if gpc.config.model.use_flash_attn:
            import rotary_emb

            rotary_emb.apply_rotary(
                q1, q2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), q1, q2, False
            )
        else:
            q1, q2 = apply_rotary_packed_torch(
                q1, q2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), False
            )
        k1, k2 = qkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        if gpc.config.model.use_flash_attn:
            rotary_emb.apply_rotary(
                k1, k2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), k1, k2, False
            )
        else:
            k1, k2 = apply_rotary_packed_torch(
                k1, k2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), False
            )
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq1, dq2 = dqkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        if gpc.config.model.use_flash_attn:
            import rotary_emb

            rotary_emb.apply_rotary(
                dq1, dq2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), dq1, dq2, True
            )
        else:
            dq1, dq2 = apply_rotary_packed_torch(
                dq1, dq2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), True
            )
        dk1, dk2 = dqkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        if gpc.config.model.use_flash_attn:
            rotary_emb.apply_rotary(
                dk1, dk2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), dk1, dk2, True
            )
        else:
            dk1, dk2 = apply_rotary_packed_torch(
                dk1, dk2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), True
            )
        return dqkv, None, None, None, None


apply_rotary_emb_packed_qkv_ = ApplyRotaryEmbPackedQKV_.apply


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base > 0, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.scale_base = scale_base
        self.scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base > 0
            else None
        )

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self, qkv: torch.Tensor, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._forward(qkv, kwargs.pop("indexes"))
        if kwargs.get("inference_params", None) is not None:
            return self._eval_forward(qkv, seqlen_offset=kwargs.get("inference_params", None).sequence_len_offset)
        else:
            return self._eval_forward(qkv)

    def _forward(self, qkv: torch.Tensor, indexes=0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(qkv, indexes)
        if self.scale is None:
            return apply_rotary_emb_packed_qkv_(qkv, self._cos_cached[indexes], self._sin_cached[indexes])
        else:
            return apply_rotary_emb_packed_qkv_(
                qkv,
                self._cos_cached[indexes],
                self._sin_cached[indexes],
                self._cos_k_cached[indexes],
                self._sin_k_cached[indexes],
            )

    def _eval_forward(self, qkv, seqlen_offset=0):
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset + qkv.shape[1])
        if self.scale is None:
            return apply_rotary_emb_qkv_(qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])
        else:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:],
                self._sin_k_cached[seqlen_offset:],
            )

    def _single_forward(
        self, x, indexes=0, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None
    ):
        assert self.scale is None
        self._update_cos_sin_cache(x, indexes)
        x = x[None, ...]
        ret = apply_rotary_emb(
            x, self._cos_cached[indexes], self._sin_cached[indexes], False, False, 0, cu_seqlens, max_seqlen
        ).squeeze(0)
        return ret

    def _single_eval_forward(self, x, seqlen_offset=0):
        assert self.scale is None
        self._update_cos_sin_cache(x, seqlen_offset + x.shape[1])
        return apply_rotary_emb(x, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])


class LinearRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev.

    Reference implementation:
        https://github.com/huggingface/transformers/blob/200009566639b5a83604e522a41df3a9 \
            5b6056ed/src/transformers/models/llama/modeling_llama.py#L159C1-L176C1
    """

    def __init__(
        self, dim: int, base=10000, scale_base=0, device=None, max_position_embeddings=2048, scaling_factor=1.0
    ):
        super().__init__(dim=dim, base=base, scale_base=scale_base, device=device)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1

        t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq.to(device=t.device))
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            # We want the multiplication by scale to happen in fp32
            self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla.

    Reference implementation:
        https://github.com/huggingface/transformers/blob/eb8489971ac1415f67b0abdd1584fde8 \
            b659ced9/src/transformers/models/llama/modeling_llama.py#L147
    """

    def __init__(
        self, dim: int, base=10000, scale_base=0, device=None, max_position_embeddings=2048, scaling_factor=1.0
    ):
        super().__init__(dim=dim, base=base, scale_base=scale_base, device=device)
        self.max_position_embeddings = max_position_embeddings
        self.scaling_factor = scaling_factor

    def _update(self, seqlen, x):
        self._seq_len_cached = seqlen
        if seqlen > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seqlen / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
        else:
            inv_freq = self.inv_freq

        t = torch.arange(seqlen, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq.to(device=t.device))
        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)
        else:
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
            # We want the multiplication by scale to happen in fp32
            self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        if seqlen <= self.max_position_embeddings:
            # Reset the tables if the sequence length has changed,
            # or if we're on a new device (possibly due to tracing for instance)
            if (
                self._seq_len_cached > self.max_position_embeddings
                or seqlen > self._seq_len_cached
                or self._cos_cached.device != x.device
                or self._cos_cached.dtype != x.dtype
            ):
                self._update(seqlen, x)
        else:
            self._update(seqlen, x)
