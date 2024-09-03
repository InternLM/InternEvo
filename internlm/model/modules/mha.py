#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
from typing import Callable, Dict, Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from internlm.model.modules.embedding import new_rotary_embedding
from internlm.model.modules.linear import new_linear
from internlm.model.modules.utils import update_kv_cache
from internlm.model.ops.attention import CrossAttention, SelfAttention
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


def _convert_cu_seqlens_for_qksplited(kwargs: Dict):
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    max_seqlen = kwargs.pop("max_seqlen", None)

    if cu_seqlens is not None:
        kwargs["cu_seqlens_q"] = cu_seqlens
        kwargs["cu_seqlens_k"] = cu_seqlens
        kwargs["max_seqlen_q"] = max_seqlen
        kwargs["max_seqlen_k"] = max_seqlen

    return kwargs


class MHA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        max_position_embeddings (int): max position embeddings, 2048 by default.
        bias (bool): Whether the bias is needed for linears. True by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        use_dynamic_ntk_rope (bool): whether use dynamic ntk rope, false by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        qk_interleaved (Optional[bool]): whether the odd and even columns of wq and wk is interleaved. True by default.
        enable_qkv_fusion (bool): whether wq, wk and wv lienar is fused. True by default.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_position_embeddings: int = 2048,
        bias: bool = True,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        rope_base: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qk_interleaved: Optional[bool] = True,
        enable_qkv_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // num_heads
        self.enable_qkv_fusion = enable_qkv_fusion

        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.max_position_embeddings = max_position_embeddings
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=1.0,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        if self.enable_qkv_fusion:
            # bias=True is according to https://spaces.ac.cn/archives/9577
            self.wqkv = new_linear("wqkv", embed_dim, 3 * embed_dim, bias, **factory_kwargs)
        else:
            self.wq = new_linear("wq", embed_dim, embed_dim, bias, **factory_kwargs)
            self.wk = new_linear("wk", embed_dim, self.kv_dim, bias, **factory_kwargs)
            self.wv = new_linear("wv", embed_dim, self.kv_dim, bias, **factory_kwargs)

        self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = CrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

        # output projection always have the bias (for now)
        self.out_proj = new_linear("out_proj", embed_dim, embed_dim, bias=True, **factory_kwargs)

    def register_checkpoint_compatibility_hooks(
        self, pre_load_hook: Optional[Callable] = None, pre_save_hook: Optional[Callable] = None
    ):
        # Here we explicitly expose the checkpoint compatibility interface of the module,
        # hoping that model developers will make good use of it when adapting.
        # Is this interface already meeting all reasonable requirements?
        self._register_load_state_dict_pre_hook(pre_load_hook, with_module=True)
        self._register_state_dict_hook(pre_save_hook)

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        # wqkv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)

            q = qkv[:, :, 0].squeeze(2)
            k = qkv[:, :, 1].squeeze(2)
            v = qkv[:, :, 2].squeeze(2)
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        # rotary embedding
        indexes = kwargs.pop("indexes", 0)
        max_seqlen = kwargs.get("max_seqlen", None)
        q = self.rotary_emb(
            q, offsets=indexes, cache_type="query", interleaved=self.interleaved, max_seqlen=max_seqlen, in_place=True
        )
        k = self.rotary_emb(
            k, offsets=indexes, cache_type="key", interleaved=self.interleaved, max_seqlen=max_seqlen, in_place=True
        )

        # self attention
        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)
        context = self.inner_attn(q, k, v, **kwargs)

        # wo
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)"))

    def _convert_unpacked_qkv_to_packed(
        self, q: torch.Tensor, kv: torch.Tensor, batch_size: int, attention_mask: torch.Tensor
    ):
        cu_seqlens = torch.concat(
            [
                torch.tensor([0], dtype=torch.int32, device=attention_mask.device),
                attention_mask.sum(dim=-1).to(dtype=torch.int32),
            ],
            dim=0,
        ).cumsum(dim=0, dtype=torch.int32)

        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens

        max_seqlen_q = attention_mask.shape[-1]
        max_seqlen_k = attention_mask.shape[-1]

        q_packed = (
            q.masked_select(attention_mask.view(batch_size, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1]).unsqueeze(0)
        )
        kv_packed = (
            kv.masked_select(attention_mask.view(batch_size, -1, 1, 1, 1))
            .view(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
            .unsqueeze(0)
        )

        return q_packed, kv_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

    def _inference(self, x, inference_params, **kwargs):  # pylint: disable=W0613
        assert inference_params is not None, "inference_params is required for inference"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        attention_mask = inference_params.attention_mask
        sequence_len_offset = inference_params.sequence_len_offset
        batch_size = x.shape[0]

        # wqkv, output: q, kv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)

            q = qkv[:, :, 0].squeeze(2)
            kv = qkv[:, :, 1:]
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)
            kv = torch.stack([k, v], dim=2)

        # rotary embedding, output: q, kv
        # q shape: [bsz, nheads, head_dim]
        # kv shape: [bsz, seqlen, 2, nheads, head_dim]
        if self.use_dynamic_ntk_rope:
            # update kv cache fisrt when enable dynamic ntk rope.
            kv = update_kv_cache(kv, inference_params, self.layer_idx)

            if sequence_len_offset != 0:
                if sequence_len_offset > self.max_position_embeddings:
                    logger.warning(
                        "Notice your prompt's length is longer than model's max_position_embeddings: "
                        f"{self.max_position_embeddings}, which will cause deviations in dynamic ntk calculations."
                    )

                if self.rotary_emb_dim > 0:
                    q = self.rotary_emb(
                        q, offsets=sequence_len_offset, cache_type="query", interleaved=self.interleaved
                    )
                    k = kv[:, :, 0].squeeze(2)
                    self.rotary_emb(
                        k, offsets=0, cache_type="key", interleaved=self.interleaved, in_place=True
                    )  # in-place is important
            else:
                if self.rotary_emb_dim > 0:
                    q = self.rotary_emb(q, offsets=0, cache_type="query", interleaved=self.interleaved)
                    k = kv[:, :, 0].squeeze(2)
                    self.rotary_emb(
                        k, offsets=0, cache_type="key", interleaved=self.interleaved, in_place=True
                    )  # in-place is important
        else:
            assert self.rotary_emb_dim > 0, "You should use rotary_emb."

            k, v = kv[:, :, 0].squeeze(2), kv[:, :, 1].squeeze(2)

            if attention_mask is None:
                q = self.rotary_emb(q, offsets=sequence_len_offset, cache_type="query", interleaved=self.interleaved)
                k = self.rotary_emb(k, offsets=sequence_len_offset, cache_type="key", interleaved=self.interleaved)
            else:
                if sequence_len_offset == 0:
                    q = self.rotary_emb(
                        q, offsets=0, cache_type="query", interleaved=self.interleaved, left_padding_mask=attention_mask
                    )
                    k = self.rotary_emb(
                        k, offsets=0, cache_type="key", interleaved=self.interleaved, left_padding_mask=attention_mask
                    )
                else:
                    if sequence_len_offset > self.max_position_embeddings:
                        logger.warning(
                            "Notice your prompt's length is longer than model's max_position_embeddings: "
                            f"{self.max_position_embeddings}, which will cause deviations in dynamic ntk calculations."
                        )

                    empties = attention_mask[..., -1].sum(dim=-1)
                    indexes4q = sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device) - empties
                    indexes4k = sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device) - empties
                    q = self.rotary_emb(q, offsets=indexes4q, cache_type="query", interleaved=self.interleaved)
                    k = self.rotary_emb(k, offsets=indexes4k, cache_type="key", interleaved=self.interleaved)

            kv = torch.stack([k, v], dim=2)
            # update kv cache after rotary embedding when disable dynamic ntk rope.
            kv = update_kv_cache(kv, inference_params, self.layer_idx)

        # self-attention
        if attention_mask is None:
            context = self.inner_cross_attn(q, kv)
        else:
            if sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                attn_mask = attention_mask[:, None, ...]
                attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
                attn_mask4flsh = ~attn_mask[:, :, -1, :].view(batch_size, -1)

                output = self.inner_attn(*self._convert_unpacked_qkv_to_packed(q, kv, batch_size, attn_mask4flsh))
                output = output.to(x.dtype)

                context = torch.zeros_like(q).masked_scatter_(attn_mask4flsh.view(batch_size, -1, 1, 1), output)
            else:
                attn_mask = attention_mask[:, -1, :].view(batch_size, 1, 1, -1)

                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                sp = k.shape
                scores = torch.einsum(
                    "blhd,bnhd->bhln",
                    q,
                    k.reshape(sp[0], sp[1], q.size(2), sp[3]),
                ) / math.sqrt(q.size(-1))
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                context = torch.einsum(
                    "bhmn,bnhd->bmhd",
                    scores,
                    v.reshape(sp[0], sp[1], q.size(2), sp[3]),
                )

        # wo
        return self.out_proj(rearrange(context, "b s h d -> b s (h d)"))


class GQA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of attention heads for key and value.
        max_position_embeddings (int): max position embeddings, 2048 by default.
        bias (bool): Whether the bias is needed for linears. Will be used when initializing QKV matrix and
                     output projection. False by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        use_dynamic_ntk_rope (bool): whether use dynamic ntk rope, false by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        qk_interleaved (Optional[bool]): whether the odd and even columns of wq and wk is interleaved. True by default.
        enable_qkv_fusion (bool): whether wq, wk and wv lienar is fused. True by default.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 2048,
        bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rope_base: int = 10000,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qk_interleaved: Optional[bool] = True,
        enable_qkv_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.causal = causal

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_per_kv = num_heads // num_kv_heads
        self.head_dim = self.embed_dim // num_heads
        self.kv_dim = self.head_dim * num_kv_heads
        self.enable_qkv_fusion = enable_qkv_fusion

        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.max_position_embeddings = max_position_embeddings
        self.interleaved = qk_interleaved

        factory_kwargs = {"device": device, "dtype": dtype}

        assert self.use_dynamic_ntk_rope is False, "Not support dynamic ntk rope yet."
        assert self.embed_dim % num_heads == 0, "embedding dim must be divisible by num_heads"

        if self.rotary_emb_dim > 0:
            self.rotary_emb = new_rotary_embedding(
                self.rotary_emb_dim,
                base=rope_base,
                scale_base=rotary_emb_scale_base,
                device=device,
                max_position_embeddings=max_position_embeddings,
                scaling_factor=1.0,
                rotary_type="dynamic_ntk" if self.use_dynamic_ntk_rope else "native",
            )

        if enable_qkv_fusion:
            self.wqkv = new_linear("wqkv", embed_dim, embed_dim + 2 * self.kv_dim, bias, **factory_kwargs)
        else:
            self.wq = new_linear("wq", embed_dim, embed_dim, bias, **factory_kwargs)
            self.wk = new_linear("wk", embed_dim, self.kv_dim, bias, **factory_kwargs)
            self.wv = new_linear("wv", embed_dim, self.kv_dim, bias, **factory_kwargs)

        self.inner_attn = SelfAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout, layer_idx=layer_idx
        )
        self.inner_cross_attn = CrossAttention(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout, layer_idx=layer_idx
        )

        self.wo = new_linear("wo", embed_dim, embed_dim, bias, **factory_kwargs)

    def register_checkpoint_compatibility_hooks(
        self, pre_load_hook: Optional[Callable] = None, pre_save_hook: Optional[Callable] = None
    ):
        # Here we explicitly expose the checkpoint compatibility interface of the module,
        # hoping that model developers will make good use of it when adapting.
        # Is this interface already meeting all reasonable requirements?
        self._register_load_state_dict_pre_hook(pre_load_hook, with_module=True)
        self._register_state_dict_hook(pre_save_hook)

    def forward(self, x, inference_params=None, **kwargs):
        if inference_params is None:
            return self._training(x=x, **kwargs)
        else:
            return self._inference(x=x, inference_params=inference_params, **kwargs)

    def _training(self, x, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
        """
        # wqkv
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=self.q_per_kv + 2, d=self.head_dim)
            q, k, v = (qkv[..., : self.q_per_kv, :], qkv[..., -2, :], qkv[..., -1, :])
            q = rearrange(q, "b s h gs d -> b s (h gs) d")
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        kwargs = _convert_cu_seqlens_for_qksplited(kwargs)

        # rotary embedding
        if self.rotary_emb_dim > 0:
            indexes = kwargs.pop("indexes", 0)
            max_seqlen_q = kwargs.get("max_seqlen_q", None)
            max_seqlen_k = kwargs.get("max_seqlen_k", None)

            q = self.rotary_emb(
                q, offsets=indexes, max_seqlen=max_seqlen_q, cache_type="query", interleaved=self.interleaved
            )
            k = self.rotary_emb(
                k, offsets=indexes, max_seqlen=max_seqlen_k, cache_type="key", interleaved=self.interleaved
            )

        kv = torch.concat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)

        # self attention
        context = self.inner_attn(q, kv, **kwargs)

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))

    def _convert_unpacked_qkv_to_packed(
        self, q: torch.Tensor, kv: torch.Tensor, batch_size: int, attention_mask: torch.Tensor
    ):
        cu_seqlens = torch.concat(
            [
                torch.tensor([0], dtype=torch.int32, device=attention_mask.device),
                attention_mask.sum(dim=-1).to(dtype=torch.int32),
            ],
            dim=0,
        ).cumsum(dim=0, dtype=torch.int32)

        cu_seqlens_q = cu_seqlens
        cu_seqlens_k = cu_seqlens

        max_seqlen_q = attention_mask.shape[-1]
        max_seqlen_k = attention_mask.shape[-1]

        q_packed = (
            q.masked_select(attention_mask.view(batch_size, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1]).unsqueeze(0)
        )
        kv_packed = (
            kv.masked_select(attention_mask.view(batch_size, -1, 1, 1, 1))
            .view(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])
            .unsqueeze(0)
        )

        return q_packed, kv_packed, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

    def _inference(self, x, inference_params, **kwargs):  # pylint: disable=W0613
        assert inference_params is not None, "inference_params is required for inference"
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        attention_mask = inference_params.attention_mask
        sequence_len_offset = inference_params.sequence_len_offset
        window_size = inference_params.window_size

        batch_size = x.shape[0]

        # wqkv, output: q, k, v
        if self.enable_qkv_fusion:
            qkv = self.wqkv(x)
            qkv = rearrange(qkv, "b s (h gs d) -> b s h gs d", gs=self.q_per_kv + 2, d=self.head_dim)
            q, k, v = (qkv[..., : self.q_per_kv, :], qkv[..., -2, :], qkv[..., -1, :])
            q = rearrange(q, "b s h gs d -> b s (h gs) d")
        else:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            q = rearrange(q, "b s (h d) -> b s h d", d=self.head_dim)
            k = rearrange(k, "b s (h d) -> b s h d", d=self.head_dim)
            v = rearrange(v, "b s (h d) -> b s h d", d=self.head_dim)

        # rotary embedding, output: q, kv
        assert self.rotary_emb_dim > 0
        if attention_mask is None:
            raise NotImplementedError(
                "You should make sure you are aware that you are changing the method of generating."
                "According to your generation function instead of inference/seq_generator_module.py, "
                "You may implement here for normal running."
            )
        else:
            if inference_params.sequence_len_offset == 0:
                q = self.rotary_emb(
                    q, offsets=0, cache_type="query", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
                k = self.rotary_emb(
                    k, offsets=0, cache_type="key", interleaved=self.interleaved, left_padding_mask=attention_mask
                )
            else:
                empties = attention_mask[..., -1].sum(dim=-1)
                indexes4q = sequence_len_offset * torch.ones(q.size(0), dtype=torch.int, device=q.device) - empties
                indexes4k = sequence_len_offset * torch.ones(k.size(0), dtype=torch.int, device=k.device) - empties
                q = self.rotary_emb(q, offsets=indexes4q, cache_type="query", interleaved=self.interleaved)
                k = self.rotary_emb(k, offsets=indexes4k, cache_type="key", interleaved=self.interleaved)

        kv = torch.stack([k, v], dim=2)

        if window_size is None or window_size > sequence_len_offset:
            kv = update_kv_cache(kv, inference_params, self.layer_idx)
        else:  # window_size <= sequence_len_offset
            assert kv.size(1) == 1, "update kv length more than 1"

            inference_params.key_value_memory_dict[self.layer_idx][
                :, inference_params.keep_first : inference_params.window_size - 1, ...
            ] = inference_params.key_value_memory_dict[self.layer_idx][
                :, -(inference_params.window_size - 1 - inference_params.keep_first) :, ...
            ].clone()
            inference_params.real_sequence_len_offset = inference_params.sequence_len_offset
            inference_params.sequence_len_offset = inference_params.window_size - 1

            kv = update_kv_cache(kv, inference_params, self.layer_idx)

            inference_params.sequence_len_offset = inference_params.real_sequence_len_offset

        # When using FP16, there is a high probability of NAN in the KV.
        # Since NAN cannot be removed by multiplying with and 0, it needs
        # to be removed manually here.
        kv = torch.where(torch.isnan(kv), 0, kv)

        # attention
        if attention_mask is None:
            context = self.inner_cross_attn(q, kv)
        else:
            if sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                attn_mask = attention_mask[:, None, ...]
                attn_mask = torch.logical_or(torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask)
                attn_mask4flsh = ~attn_mask[:, :, -1, :].view(batch_size, -1)

                output = self.inner_attn(*self._convert_unpacked_qkv_to_packed(q, kv, batch_size, attn_mask4flsh))
                output = output.to(x.dtype)

                context = torch.zeros_like(q).masked_scatter_(attn_mask4flsh.view(batch_size, -1, 1, 1), output)

            else:
                attn_mask = attention_mask[:, -1, :].view(batch_size, 1, 1, -1)
                if window_size is not None and window_size <= sequence_len_offset:
                    attn_mask = torch.concat(
                        [
                            attn_mask[..., : inference_params.keep_first],
                            attn_mask[..., -(window_size - inference_params.keep_first) :],
                        ],
                        dim=-1,
                    )

                k, v = torch.chunk(kv, 2, dim=2)
                k = k.squeeze(2)
                v = v.squeeze(2)
                sp = k.shape
                expansion = q.size(2) // k.size(2)
                scores = torch.einsum(
                    "blhd,bnhd->bhln",
                    q,
                    k.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                ) / math.sqrt(q.size(-1))
                scores = scores.masked_fill(attn_mask, -65000.0)
                scores = F.softmax(scores, dim=-1)  # bsz x h x L x L
                context = torch.einsum(
                    "bhmn,bnhd->bmhd",
                    scores,
                    v.unsqueeze(3).expand(-1, -1, -1, expansion, -1).reshape(sp[0], sp[1], q.size(2), sp[3]),
                )

        # wo
        return self.wo(rearrange(context, "b s h d -> b s (h d)"))
