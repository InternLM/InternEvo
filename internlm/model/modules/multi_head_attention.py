#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import enum
import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from internlm.accelerator import internlm_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.modules.attn import (
    AscendFlashSelfAttention,
    CrossAttention,
    DistributedAttention,
    FlashCrossAttention,
    FlashSelfAttention,
    SelfAttention,
)
from internlm.model.modules.embedding import (
    DynamicNTKScalingRotaryEmbedding,
    RotaryEmbedding,
)
from internlm.model.ops.linear import get_linear_cls
from internlm.utils.common import get_current_device


class AttnType(enum.Enum):
    TORCH = 1
    FLASH = 2
    ASCEND_FLASH = 3
    RING = 4
    MAMBA = 5


convert_attn_type = {
    "flash": AttnType.FLASH,
    "torch": AttnType.TORCH,
    "npu_flash": AttnType.ASCEND_FLASH,
    "ring": AttnType.RING,
    "mamba": AttnType.MAMBA,
}


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


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    reset_attention_mask,
    reset_position_ids=False,
    eod_mask_loss=True,
    gen_loss_mask=False,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=get_current_device())).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Loss mask.
    if gen_loss_mask:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
        if eod_mask_loss:
            loss_mask[data == eod_token] = 0.0
    else:
        loss_mask = None

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == gpc.config.model.vocab_size + 1]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # reset EOD token.
                data[b][i] = eod_token
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5
    return attention_mask, loss_mask, position_ids


def get_attn_cls(attn_type: AttnType, tp_mode: str, use_gqa: bool, attn_args: dict):
    if attn_type == AttnType.FLASH:
        if not use_gqa:
            inner_attn_cls = FlashSelfAttention
            inner_cross_attn_cls = FlashCrossAttention
        else:
            from flash_attn.flash_attn_interface import (
                flash_attn_varlen_kvpacked_func as flash_attn_unpadded_func,
            )

            inner_attn = flash_attn_unpadded_func
            inner_cross_attn = CrossAttention
    elif attn_type == AttnType.ASCEND_FLASH:
        inner_attn_cls = AscendFlashSelfAttention
        inner_cross_attn_cls = CrossAttention
    elif attn_type == AttnType.TORCH:
        inner_attn_cls = SelfAttention
        inner_cross_attn_cls = CrossAttention
    else:
        raise ValueError(f"Unexcept attention type: {attn_type}")

    if not use_gqa:
        causal = attn_args.pop("causal")
        softmax_scale = attn_args.pop("softmax_scale")
        attention_dropout = attn_args.pop("attention_dropout")
        sequence_process_group = attn_args.pop("sequence_process_group")

        inner_attn = inner_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=attention_dropout, **attn_args
        )
        inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=attention_dropout, **attn_args
        )

    if tp_mode == "isp":
        inner_attn = DistributedAttention(inner_attn, sequence_process_group=sequence_process_group)
        inner_cross_attn = DistributedAttention(inner_cross_attn, sequence_process_group=sequence_process_group)

    return inner_attn, inner_cross_attn


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
        attn_type=AttnType.FLASH,
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
        self.num_attention_heads_per_partition = num_heads // gpc.get_world_size(parallel_mode=ParallelMode.TENSOR)
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
            comm_dim=0 if gpc.config.data.use_flash_style_data_format else 1,
            **factory_kwargs,
        )  # according to https://spaces.ac.cn/archives/9577

        attn_args = {
            "causal": self.causal,
            "softmax_scale": softmax_scale,
            "attention_dropout": dropout,
            "sequence_process_group": sequence_process_group,
        }
        self.inner_attn, self.inner_cross_attn = get_attn_cls(attn_type, self.tp_mode, False, attn_args)

        # output projection always have the bias (for now)
        out_proj_cls = get_linear_cls(self.tp_mode, "row")
        self.out_proj = out_proj_cls(
            embed_dim,
            embed_dim,
            process_group,
            bias=True,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            comm_dim=0 if gpc.config.data.use_flash_style_data_format else 1,
            **factory_kwargs,
        )

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get("indexes", None) is not None and "attention_mask" not in kwargs:
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
            if gpc.config.model.dtype is torch.float32 and gpc.config.model.attn_type == AttnType.FLASH:
                with internlm_accelerator.amp.autocast(dtype=torch.bfloat16):
                    if qkv.dtype not in [torch.float16, torch.bfloat16]:
                        qkv = qkv.to(torch.bfloat16)
                        context = self.inner_attn(qkv)
            else:
                if gpc.config.model.attn_type == AttnType.ASCEND_FLASH:
                    mask_size = kwargs["attention_mask"].size()
                    assert len(mask_size) == 4, mask_size
                    assert mask_size[0] == gpc.config.data.micro_bsz
                    assert mask_size[1] == 1
                    assert mask_size[2] == mask_size[3] == gpc.config.data.seq_len

                    q = qkv[:, :, 0]
                    k = qkv[:, :, 1]
                    v = qkv[:, :, 2]
                    context = self.inner_attn(q, k, v, kwargs["attention_mask"])
                else:
                    context = self.inner_attn(qkv).to(x.dtype)
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
        qkv = self.Wqkv(x)  # total x hsz'
        qkv = rearrange(qkv, "t (three h d) -> t three h d", three=3, d=self.head_dim)  # total x 3 x n_head x d
        qkv = self.rotary_emb(qkv, **kwargs)
        kwargs.pop("indexes")
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
        out = self.out_proj(context)

        return out
