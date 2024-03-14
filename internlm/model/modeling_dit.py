
from typing import Any
import torch
import torch.nn as nn
import torch.distributed as dist

# from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


from einops import rearrange
from internlm.model import DistributedAttention
from flash_attn.modules.mha import (
    FlashSelfAttention,
    SelfAttention,
)
from flash_attn import (
    flash_attn_func
)

from ring_flash_attn import ring_flash_attn_qkvpacked_func, ring_flash_attn_func


class FlashAttnFuncWrapper():
    def __init__(self, scale, attention_dropout, causal=False) -> None:
        self.causal = causal
        self.scale = scale
        self.attention_dropout = attention_dropout
    def __call__(self, q, k, v) -> Any:
        # (batch_size, seqlen, nheads, headdim)
        return flash_attn_func(q, k, v,
                               dropout_p=self.attention_dropout,
                               softmax_scale = self.scale,
                               causal=self.causal)

class RingAttnFuncWrapper(FlashAttnFuncWrapper):
    def __init__(self, scale, attention_dropout, causal=False, group=None,) -> None:
        self.causal = causal
        self.scale = scale
        self.attention_dropout = attention_dropout
        self.group=group

    def __call__(self, q=None, k=None, v=None, qkv=None,) -> Any:
        if qkv is not None:
            return ring_flash_attn_qkvpacked_func(qkv, dropout_p=self.attention_dropout,
                                                  causal=self.causal, group=self.group)
        else:
            return ring_flash_attn_func(q, k, v, dropout_p=self.attention_dropout,
                                                  causal=self.causal, group=self.group)


class DiTMHA(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            fused_attn: bool = True,
            dtype = torch.bfloat16,
            sequence_parallel: bool = False,
            sequence_parallel_type: str = 'ulysses',
            parallel_group = None,
            ring_attention = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if dtype not in [torch.float16, torch.bfloat16]:
            fused_attn = False
            print("Disabled fused_attn since dtype: ", dtype)
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm = qk_norm
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sequence_parallel = sequence_parallel
        self.parallel_group = parallel_group
        self.ring_attention = ring_attention
        # FlashSelfAttention: (B, S, 3, H, D)
        # FlashAttnFuncWrapper: (batch_size, seqlen, nheads, headdim)
        assert not (sequence_parallel and ring_attention), "seuquence_paralllel and ring_attention can not both be True"
        if ring_attention:
            self.inner_attn = RingAttnFuncWrapper(self.scale, self.attn_drop.p, causal=False, group=parallel_group)
        else:
            inner_attn_cls =FlashSelfAttention if fused_attn else SelfAttention
            if qk_norm:
                inner_attn_cls = FlashAttnFuncWrapper if fused_attn else SelfAttention
            self.inner_attn = inner_attn_cls(causal=False, softmax_scale=self.scale,
                                            attention_dropout=self.attn_drop.p if self.training else 0.)
            if sequence_parallel:
                # DistributedAttention: [sequence, 3, head, head_dim] / [sequence, head, head_dim]
                self.inner_attn = DistributedAttention(self.inner_attn, sequence_process_group=parallel_group, varlen=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        new_shape = [B, N, 3, self.num_heads, self.head_dim]
        qkv = self.qkv(x).reshape(*new_shape) # B, N, 3, nh, hdim
        # qkv = qkv.permute(2,0,1,3,4)        # 3, B, N, nh, hdim
        # q, k, v = qkv.unbind(0)             # B, N, nh, hdim
        if self.qk_norm:
            dim=2
            q, k, v = qkv.unbind(qkv, dim=dim)  # B,N, 3, nh, hdim -> B,N, nh, hdim
            q, k = self.q_norm(q), self.k_norm(k)
            x = self.inner_attn(q=q, k=k, v=v)
        else:
            if self.ring_attention:
                x = self.inner_attn(qkv=qkv)
            else:
                x = self.inner_attn(qkv)

        # x: B, N, nh, hdim
        x = x.reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
