#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import Module

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import global_context as gpc
from internlm.model.modules.embedding import (
    DynamicNTKScalingRotaryEmbedding,
    RotaryEmbedding,
)
from internlm.model.ops.linear import get_linear_cls

internlm_accelerator = get_accelerator()


# adpated from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
class _SeqAllToAll(torch.autograd.Function):
    "sequence alltoall"

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input_: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        if dist.get_world_size(group) <= 1:
            return input_

        seq_world_size = dist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input_, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        if dist.get_world_size(ctx.group) <= 1:
            return (None, *grad_output, None, None)

        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


# adpated from https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/sequence/layer.py
class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        first_scatter_idx (int): scatter_idx for the first all2all comm
        first_gather_idx (int): gather_idx for the first all2all comm
        second_scatter_idx (int): scatter_idx for the second all2all comm
        second_gather_idx (int): gather_idx for the second all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self._scatter_gather_idx = {}

        # scatter_gather_idx contains the scatter and gather index for different data packed mode
        # key is the data packed mode, which should be in ['qkv', 'kv', 'q', 'output']
        # value is the scatter and gather index in all2all
        self._scatter_gather_idx["qkv"] = [2, 0]  # qkv shape:[sequence, 3, head, head_dim]
        self._scatter_gather_idx["kv"] = [2, 0]  # kv shape: [sequence, 2, head, head_dim]
        self._scatter_gather_idx["q"] = [1, 0]  # q/k/v shape: [sequence, head, head_dim]
        self._scatter_gather_idx["output"] = [0, 1]  # output shape: [sequence, head, head_dim]

    def forward(
        self, qkv: Tensor = None, kv: Tensor = None, q: Tensor = None, k: Tensor = None, v: Tensor = None, **kwargs: Any
    ) -> Tensor:
        if gpc.is_evaluating is True or gpc.config.data.use_packed_dataset is False:
            # when conducting evaluation, the scatter and gather index should add 1.
            eval_scatter_gather_idx = {key: [x + 1 for x in value] for key, value in self._scatter_gather_idx.items()}
            return self._forward(qkv=qkv, kv=kv, q=q, k=k, v=v, scatter_gather=eval_scatter_gather_idx, **kwargs)
        else:
            return self._forward(qkv=qkv, kv=kv, q=q, k=k, v=v, scatter_gather=self._scatter_gather_idx, **kwargs)

    def _forward(
        self,
        qkv: Tensor = None,
        kv: Tensor = None,
        q: Tensor = None,
        k: Tensor = None,
        v: Tensor = None,
        scatter_gather: dict = None,
        **kwargs: Any,
    ) -> Tensor:
        """forward

        Arguments:
            qkv (Tensor): packed qkv input to the layer
            kv (Tensor): packed kv input to the layer
            q (Tensor): q input to the layer
            k (Tensor): k input to the layer
            v (Tensor): v input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        if qkv is not None:
            qkv = _SeqAllToAll.apply(self.spg, qkv, scatter_gather["qkv"][0], scatter_gather["qkv"][1])
            context_layer = self.local_attn(qkv, **kwargs)
        elif kv is not None:
            q = _SeqAllToAll.apply(self.spg, q, scatter_gather["q"][0], scatter_gather["q"][1])
            kv = _SeqAllToAll.apply(self.spg, kv, scatter_gather["kv"][0], scatter_gather["kv"][1])
            context_layer = self.local_attn(q, kv, **kwargs)
        else:
            q = _SeqAllToAll.apply(self.spg, q, scatter_gather["q"][0], scatter_gather["q"][1])
            k = _SeqAllToAll.apply(self.spg, k, scatter_gather["q"][0], scatter_gather["q"][1])
            v = _SeqAllToAll.apply(self.spg, v, scatter_gather["q"][0], scatter_gather["q"][1])
            context_layer = self.local_attn(q, k, v, **kwargs)
        output = _SeqAllToAll.apply(self.spg, context_layer, scatter_gather["output"][0], scatter_gather["output"][1])

        # out e.g., [s/p::h]
        return output


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
        self.drop = nn.Dropout(attention_dropout)

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, S)
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
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
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


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

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            key_padding_mask: boolean mask to apply to the attention weights. True means to keep,
                False means to mask out. (B, Sk)
        """
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        causal = self.causal if causal is None else causal
        seqlen_k = kv.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[4] == q.shape[3]
        if kv.shape[3] != q.shape[2]:  # MQA/GQA
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
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
        attention_drop = self.drop(attention)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_sequence_len,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        if not inference_params.fused_ft_kernel:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]
        else:
            # For FT, k_cache has shape (b, h, headdim / packsize, s, packsize)
            # where packsize = 4 if fp32, 8 if fp16 or bf16.
            # v_cache has shape (b, h, s, headdim)
            k_cache, v_cache = inference_params.key_value_memory_dict[layer_idx]
            kv_cache = None
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.sequence_len_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else v_cache.shape[0])
    assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else v_cache.shape[2])
    # Copy key and values.
    if not inference_params.fused_ft_kernel:
        assert kv_cache is not None
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        kv = kv_cache[batch_start:batch_end, :sequence_end, ...]
        return kv
    else:
        assert inference_params.sequence_len_offset == 0
        # FT kernel requires different layouts for the k_cache and v_cache.
        assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]
        packsize = 4 if kv.dtype == torch.float32 else 8
        if kv_cache is not None:
            kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
            k_cache = rearrange(
                kv_cache[:, :, 0], "b s h (d packsize) -> b h d s packsize", packsize=packsize
            ).contiguous()
            v_cache = rearrange(kv_cache[:, :, 1], "b s h d -> b h s d").contiguous()
            inference_params.key_value_memory_dict[layer_idx] = (k_cache, v_cache)
        else:
            k_cache[batch_start:batch_end, :, :, :sequence_end, :] = rearrange(
                kv[:, :, 0], "b s h (d packsize) -> b h d s packsize", packsize=packsize
            )
            v_cache[batch_start:batch_end, :, :sequence_end, :] = rearrange(kv[:, :, 1], "b s h d -> b h s d")
        return kv


class MHA(nn.Module):
    """
    Multi-head self-attention and cross-attention.

    Args:
        embed_dim (int): The dimention of hidden state.
        num_heads (int): The number of attention heads.
        process_group (torch.distributed.ProcessGroup): The group of the current device for `parallel_mode`.
        max_position_embeddings (int): max position embeddings, 2048 by default.
        dropout (float): The dropout rate for cross attention and self attention. 0.0 by default.
        softmax_scale (float): The temperature to use for the softmax attention.
        causal (boolean): Whether to apply causal attention mask. False by default.
        layer_idx (int): The index of current layer. None by default.
        use_dynamic_ntk_rope (bool): whether use dynamic ntk rope, false by default.
        rotary_emb_dim (int): The dimention of Rotary Embedding. 0 by default.
        rotary_emb_scale_base (int): The scaling factor of Rotary Embedding. If scale_base > 0, this implements
                                    XPos(Sun et al., https://arxiv.org/abs/2212.10554). 0 by default.
        use_flash_attn (bool): Whether to use flash-attn. True by default.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        sequence_process_group: Optional[torch.distributed.ProcessGroup],
        max_position_embeddings: int = 2048,
        dropout: float = 0.0,
        softmax_scale: float = None,
        causal: bool = False,
        layer_idx: int = None,
        use_dynamic_ntk_rope: bool = False,
        rotary_emb_dim: int = 0,
        rotary_emb_scale_base: int = 0,
        use_flash_attn: bool = True,
        rope_base: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tp_mode: str = "mtp",
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings
        self.use_dynamic_ntk_rope = use_dynamic_ntk_rope
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        self.tp_mode = tp_mode

        if self.rotary_emb_dim > 0:
            if self.use_dynamic_ntk_rope:
                self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    self.rotary_emb_dim,
                    base=rope_base,
                    scale_base=rotary_emb_scale_base,
                    device=device,
                    max_position_embeddings=max_position_embeddings,
                    scaling_factor=1.0,  # Currently do not support dynamic scaling.
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.rotary_emb_dim, base=rope_base, scale_base=rotary_emb_scale_base, device=device
                )

        # notice here should change bias=True
        Wqkv_cls = get_linear_cls(self.tp_mode, "column")
        self.Wqkv = Wqkv_cls(
            embed_dim,
            3 * embed_dim,
            process_group,
            bias=True,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )  # according to https://spaces.ac.cn/archives/9577

        if gpc.config.model.use_flash_attn:
            from flash_attn.modules.mha import FlashCrossAttention, FlashSelfAttention

            inner_attn_cls = FlashSelfAttention
            inner_cross_attn_cls = FlashCrossAttention
        else:
            inner_attn_cls = SelfAttention
            inner_cross_attn_cls = CrossAttention
        self.inner_attn = inner_attn_cls(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        if self.tp_mode == "isp":
            self.inner_attn = DistributedAttention(self.inner_attn, sequence_process_group=sequence_process_group)
            self.inner_cross_attn = DistributedAttention(
                self.inner_cross_attn, sequence_process_group=sequence_process_group
            )

        # output projection always have the bias (for now)
        out_proj_cls = get_linear_cls(self.tp_mode, "row")
        self.out_proj = out_proj_cls(
            embed_dim,
            embed_dim,
            process_group,
            bias=True,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._packed_forward(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, seqlen=seqlen, inference_params=inference_params, **kwargs)

    def _forward(self, x, seqlen=None, inference_params=None, **kwargs):  # pylint: disable=W0613
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        bsz, _, _ = x.shape
        qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
        else:
            qkv = rearrange(qkv, "(b s) (three h d) -> b s three h d", s=seqlen, three=3, d=self.head_dim)

        if inference_params is None:
            kwargs["inference_params"] = inference_params
            qkv = self.rotary_emb(qkv, **kwargs)
            if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
                with internlm_accelerator.amp.autocast(dtype=torch.bfloat16):
                    if qkv.dtype not in [torch.float16, torch.bfloat16]:
                        qkv = qkv.to(torch.bfloat16)
                    context = self.inner_attn(qkv).to(x.dtype)
            else:
                context = self.inner_attn(qkv)

        else:
            if self.use_dynamic_ntk_rope:
                q = qkv[:, :, 0]
                assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
                kv = _update_kv_cache(qkv[:, :, 1:], inference_params, self.layer_idx)
                if inference_params.sequence_len_offset != 0:
                    # q shape: [bsz, 1, nheads, head_dim]
                    # kv shape: [bsz, seqlen, 2, nheads, head_dim]
                    bsz, seq_len, _, nheads, head_dim = kv.shape
                    q = torch.cat([q.new_zeros(size=(bsz, seq_len - 1, nheads, head_dim)), q], dim=1).unsqueeze(2)
                    qkv = torch.cat([q, kv], dim=2)
                    if self.rotary_emb_dim > 0:
                        qkv = self.rotary_emb(qkv)
                    q = qkv[:, [-1], 0]
                    kv = qkv[:, :, 1:]
                else:
                    if inference_params.sequence_len_offset > self.max_position_embeddings:
                        warnings.warn(
                            "Notice your prompt's length is longer than model's max_position_embeddings: "
                            f"{self.max_position_embeddings}, which will cause deviations in dynamic ntk calculations."
                        )
                    if self.rotary_emb_dim > 0:
                        kwargs["inference_params"] = inference_params
                        qkv = self.rotary_emb(qkv, **kwargs)
                        q = qkv[:, :, 0]
                        kv = qkv[:, :, 1:]
            else:
                assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
                q, k, v = (x.squeeze(2) for x in qkv.chunk(chunks=3, dim=2))
                kv = torch.stack([k, v], dim=2)
                assert self.rotary_emb_dim > 0, "You should use rotary_emb."

                if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
                    empties = inference_params.attention_mask[..., -1].sum(dim=-1)
                    if inference_params.sequence_len_offset == 0:
                        moved_q = q.clone()
                        moved_k = k.clone()
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                moved_q[i][: -empties[i]] = q[i][empties[i] :]
                                moved_k[i][: -empties[i]] = k[i][empties[i] :]
                        moved_q = self.rotary_emb._single_eval_forward(moved_q, seqlen_offset=0)
                        moved_k = self.rotary_emb._single_eval_forward(moved_k, seqlen_offset=0)
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                q[i][empties[i] :] = moved_q[i][: -empties[i]]
                                k[i][empties[i] :] = moved_k[i][: -empties[i]]
                            else:
                                q[i] = moved_q[i]
                                k[i] = moved_k[i]
                    elif not self.use_dynamic_ntk_rope:
                        if inference_params.sequence_len_offset > self.max_position_embeddings:
                            warnings.warn(
                                "Notice your prompt's length is longer than model's max_position_embeddings: "
                                f"{self.max_position_embeddings}, may cause deviations in dynamic ntk calculations."
                            )
                        q = q.squeeze(1)
                        k = k.squeeze(1)
                        q = self.rotary_emb._single_forward(
                            q,
                            inference_params.sequence_len_offset
                            * torch.ones(q.size(0), dtype=torch.int, device=q.device)
                            - empties,
                        ).unsqueeze(1)
                        k = self.rotary_emb._single_forward(
                            k,
                            inference_params.sequence_len_offset
                            * torch.ones(k.size(0), dtype=torch.int, device=k.device)
                            - empties,
                        ).unsqueeze(1)
                    else:
                        q = q.squeeze(1)
                        q = self.rotary_emb._single_forward(
                            q,
                            inference_params.sequence_len_offset
                            * torch.ones(q.size(0), dtype=torch.int, device=q.device)
                            - empties,
                        ).unsqueeze(1)
                        moved_k = k.clone()
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                moved_k[i][: -empties[i]] = k[i][empties[i] :]
                        moved_k = self.rotary_emb._single_eval_forward(moved_k, seqlen_offset=0)
                        for i in range(len(empties)):
                            if empties[i] != 0:
                                k[i][empties[i] :] = moved_k[i][: -empties[i]]
                            else:
                                k[i] = moved_k[i]
                else:
                    q = self.rotary_emb._single_forward(q, inference_params.sequence_len_offset)
                    k = self.rotary_emb._single_forward(k, inference_params.sequence_len_offset)

                kv = torch.stack([k, v], dim=2)
                kv = _update_kv_cache(kv, inference_params, self.layer_idx)

            if hasattr(inference_params, "attention_mask") and inference_params.attention_mask is not None:
                if inference_params.sequence_len_offset == 0:  # First entrance, attnmask (bs*seqlen*seqlen)
                    attn_mask = inference_params.attention_mask[:, None, ...]
                    attn_mask = torch.logical_or(
                        torch.ones_like(attn_mask, dtype=torch.bool).triu(diagonal=1), attn_mask
                    )
                    attn_mask4flsh = ~attn_mask[:, :, -1, :].view(bsz, -1)
                    cu_seqlens = torch.concat(
                        [
                            torch.tensor([0], dtype=torch.int32, device=attn_mask4flsh.device),
                            attn_mask4flsh.sum(dim=-1).to(dtype=torch.int32),
                        ],
                        dim=0,
                    )
                    cu_seqlens = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                    max_seqlen_q = attn_mask4flsh.shape[-1]
                    max_seqlen_k = attn_mask4flsh.shape[-1]
                    total_q = q.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1)).view(-1, q.shape[-2], q.shape[-1])
                    total_kv = kv.masked_select(attn_mask4flsh.view(bsz, -1, 1, 1, 1)).view(
                        -1, kv.shape[-3], kv.shape[-2], kv.shape[-1]
                    )

                    if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
                        with internlm_accelerator.amp.autocast(dtype=torch.bfloat16):
                            if total_q.dtype not in [torch.float16, torch.bfloat16]:
                                total_q = total_q.to(torch.bfloat16)
                            if total_kv.dtype not in [torch.float16, torch.bfloat16]:
                                total_kv = total_kv.to(torch.bfloat16)

                    if gpc.config.model.use_flash_attn:
                        try:
                            from flash_attn.flash_attn_interface import (
                                flash_attn_unpadded_func,
                            )
                        except ImportError:
                            try:
                                from flash_attn.flash_attn_interface import (
                                    flash_attn_unpadded_kvpacked_func as flash_attn_unpadded_func,
                                )
                            except ImportError:
                                try:
                                    from flash_attn.flash_attn_interface import (
                                        flash_attn_varlen_kvpacked_func as flash_attn_unpadded_func,
                                    )
                                except ImportError:
                                    raise ImportError("Please check your flash_attn version >= 1.0.5.")

                        output = flash_attn_unpadded_func(
                            total_q,
                            total_kv,
                            cu_seqlens,
                            cu_seqlens,
                            max_seqlen_q,
                            max_seqlen_k,
                            0.0,
                            None,
                            True,
                            False,
                        ).to(x.dtype)
                    else:
                        attn_scores = torch.matmul(total_q, total_kv.transpose(-2, -1)) / (cu_seqlens**0.5)
                        attn_weights = F.softmax(attn_scores, dim=-1)
                        output = torch.matmul(attn_weights, total_kv)

                    context = torch.zeros_like(q)
                    context = context.masked_scatter_(attn_mask4flsh.view(bsz, -1, 1, 1), output)

                else:
                    attn_mask = inference_params.attention_mask[:, -1, :].view(bsz, 1, 1, -1)

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
            else:
                context = self.inner_cross_attn(q, kv, causal=True)

        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")

        out = self.out_proj(context)
        return out

    def _packed_forward(self, x, inference_params=None, **kwargs):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)  # bsz x total x hsz
        qkv = rearrange(
            qkv, "b t (three h d) -> b t three h d", three=3, d=self.head_dim
        )  # bsz x total x 3 x n_head x d
        qkv = self.rotary_emb(qkv, **kwargs)
        kwargs.pop("indexes")

        # for packed data, batch dimension with a size of 1 should be directly squeezed off.
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU:
            qkv = qkv.squeeze(0)
        if inference_params is None:
            if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
                with internlm_accelerator.amp.autocast(dtype=torch.bfloat16):
                    if qkv.dtype not in [torch.float16, torch.bfloat16]:
                        qkv = qkv.to(torch.bfloat16)
                    context = self.inner_attn(qkv, **kwargs).to(x.dtype)
            else:
                context = self.inner_attn(qkv, **kwargs)

        else:
            raise RuntimeError("Not support this right now")

        context = rearrange(context, "b h d -> b (h d)")  # recover the shape
        # restore bsz dimension
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU:
            context = context.unsqueeze(0)

        out = self.out_proj(context)

        return out
