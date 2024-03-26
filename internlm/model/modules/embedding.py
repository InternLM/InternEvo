#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.comm.utils import (
    gather_forward_split_backward,
    split_forward_gather_backward,
)
from internlm.model.ops.rotay_emb import apply_rotary_emb


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
            output = split_forward_gather_backward(output, ParallelMode.TENSOR, dim=-2)

        return output


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

    def _update_cos_sin_cache(self, x: torch.Tensor, indexes: Union[int, torch.LongTensor] = 0):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if isinstance(indexes, int):
            seqlen = indexes + x.shape[1] + 1
        else:
            seqlen = indexes.max().item() + 1  # 移除item的调用，这可能会造成cpu和gpu的同步
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

    def _get_slice(self, tensor: torch.Tensor, offsets: Union[int, torch.LongTensor] = 0):
        if isinstance(offsets, int):
            return tensor[offsets:]
        else:
            return tensor[offsets]

    def _covert_padding(
        self, x: torch.Tensor, empties: torch.Tensor, covert_type: str = "left2right", in_place: bool = False
    ):
        # TODO: impl in_place = True.
        assert not in_place, "in_place = True is NYI."
        assert covert_type in ("left2right", "right2left"), f"Unknown covert type {covert_type}"

        ret = x.clone()  # TODO: check it.

        for i in range(len(empties)):
            if empties[i] == 0:
                continue

            if covert_type == "left2right":
                ret[i][: -empties[i]] = x[i][empties[i] :]
            else:  # right2left
                ret[i][empties[i] :] = x[i][: -empties[i]]

        return ret

    def forward(
        self,
        x: torch.Tensor,
        offsets: Union[int, torch.LongTensor] = 0,
        cache_type: str = "query",  # 有没有可能优化一下？
        interleaved: bool = False,  # TODO: 标准化模型设置 interleaved
        in_place: bool = False,
        left_padding_mask: Optional[torch.Tensor] = None,
    ):
        # assert self.scale is None
        assert cache_type in ("query", "key"), f"Unknown cache type {cache_type}"
        assert isinstance(offsets, (int, torch.LongTensor)), f"Invalid offsets type {type(offsets)}"

        if left_padding_mask is not None:
            empties = left_padding_mask[..., -1].sum(dim=-1)
            x = self._covert_padding(x, empties, covert_type="left2right", in_place=in_place)

        self._update_cos_sin_cache(x, offsets)

        cos_cached = self._cos_k_cached if cache_type == "key" and self.scale is not None else self._cos_cached
        sin_cached = self._sin_k_cached if cache_type == "key" and self.scale is not None else self._sin_cached
        ret = apply_rotary_emb(
            x, self._get_slice(cos_cached, offsets), self._get_slice(sin_cached, offsets), interleaved, in_place
        )

        if left_padding_mask is not None:
            ret = self._covert_padding(ret, empties, covert_type="right2left", in_place=in_place)

        return ret


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
            seqlen = indexes + x.shape[1] + 1

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
            seqlen = indexes + x.shape[1] + 1  # eval_forward
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


def new_rotary_embedding(
    dim: int,
    base=10000,
    scale_base=0,
    device=None,
    max_position_embeddings=2048,
    scaling_factor=1.0,
    rotary_type: str = "native",
) -> RotaryEmbedding:
    assert rotary_type in ("native", "linear_scale", "dynamic_ntk"), f"Unknown rotary type {rotary_type}"

    if rotary_type == "linear_scale":
        return LinearRotaryEmbedding(dim, base, scale_base, device, max_position_embeddings, scaling_factor)
    elif rotary_type == "dynamic_ntk":
        return DynamicNTKScalingRotaryEmbedding(dim, base, scale_base, device, max_position_embeddings, scaling_factor)
    else:  # native
        return RotaryEmbedding(dim, base, scale_base, device)
