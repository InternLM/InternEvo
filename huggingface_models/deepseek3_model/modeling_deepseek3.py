"""              self-written DeepSeek V3 model in PyTorch, trainable with FSDP+EP              """
""" modified from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py """
import warnings
import math
from typing import List, Optional, Tuple, Union, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from flash_attn import flash_attn_varlen_func, flash_attn_func

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

# python3 -m pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.3
import grouped_gemm
from grouped_gemm.backend import gmm

from .configuration_deepseek3 import DeepseekV3Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekV3Config"


def apply_ac_to_transformer_block(module: torch.nn.Module, checkpoint):
    ac_freq = round(1 / checkpoint)
    checkpoint_wrapper.__dict__.setdefault("_count", 0)
    checkpoint_wrapper._count += 1
    if checkpoint_wrapper._count % ac_freq == 0:
        return checkpoint_wrapper(module, preserve_rng_state=False)
    else:
        return module


# Based on https://github.com/pytorch/pytorch/pull/40762
class AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group: dist.ProcessGroup = None,
    ) -> torch.Tensor:

        ctx.input_shape = inputs.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return inputs

        inputs = inputs.contiguous()
        out = (
            torch.empty_like(inputs)
            if output_split_sizes is None
            else inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]))
        )
        dist.all_to_all_single(
            out,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        if ctx.needs_input_grad[0]:
            # Bypass the function if we are using only 1 GPU.
            world_size = dist.get_world_size(group=ctx.group)
            if world_size == 1:
                return grad_output, None, None, None

            grad_output = grad_output.contiguous()
            out = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            dist.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None
        return None, None, None, None


def all_to_all(x, output_split_sizes=None, input_split_sizes=None, group=None):
    return AllToAll.apply(x, output_split_sizes, input_split_sizes, group)


class GroupedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        num_groups: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        assert bias is False, "Grouped FeedForward only support bias is False."
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.full_weight_shape = torch.Size((num_groups, in_features, out_features))
        self.weight = nn.Parameter(
            torch.empty((num_groups * in_features, out_features), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def forward(self, input: torch.Tensor, batch_sizes: torch.Tensor = None) -> torch.Tensor:
        return GroupedGemm.apply(
            input,
            self.weight,
            batch_sizes,
            self.full_weight_shape,
            "gmm",
        )

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return f'in_features={self.num_groups * self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        batch_sizes: torch.Tensor,
        full_weight_shape: torch.Size,
        backend: str,
    ):
        assert full_weight_shape is not None, "full_weight_shape should be provided"
        if backend == "bmm":
            assert x.dim() == 3, f"bmm only support 3d input (e, c, m), but got: {x.shape}"
        elif backend == "gmm":
            assert x.dim() == 2, f"gmm only support 2d input (s, m), but got: {x.shape}"
            assert batch_sizes is not None, "batch_sizes should be provided for gmm"
        else:
            raise NotImplementedError(f"Invalid backend: {backend}")

        input_numel = x.numel()
        if input_numel == 0:
            backend = "bmm"

        ctx.compute_weight_gradient = weight.requires_grad
        ctx.backend = backend
        ctx.full_weight_shape = full_weight_shape

        if torch.is_autocast_enabled():
            x = x.to(dtype=torch.get_autocast_gpu_dtype())
        x = x.contiguous()

        total_weight = weight
        total_weight = total_weight.reshape(full_weight_shape)

        if torch.is_autocast_enabled():
            total_weight = total_weight.to(dtype=torch.get_autocast_gpu_dtype())
        total_weight = total_weight.contiguous()

        batch_sizes = batch_sizes.to("cpu")

        if backend == "gmm":
            output = gmm(x, total_weight, batch_sizes)
        else:
            if input_numel == 0:
                total_weight = total_weight.view(x.shape[-1], -1)
            output = torch.matmul(x, total_weight)
        assert len(output.shape) == len(x.shape)

        saved_x = None if ctx.compute_weight_gradient is False else x
        ctx.save_for_backward(saved_x, weight, batch_sizes)
        
        del total_weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, batch_sizes = ctx.saved_tensors
        backend = ctx.backend
        full_weight_shape = ctx.full_weight_shape

        if grad_output.numel() == 0:
            if ctx.needs_input_grad[1]:
                total_weight_shape = torch.Size(
                    (full_weight_shape.numel() // full_weight_shape[-1], full_weight_shape[-1])
                )
                grad_weight = torch.zeros(total_weight_shape, dtype=weight.dtype, device=weight.device)
            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(x)

            return grad_input, grad_weight, None, None, None
        
        total_weight = weight
        total_weight = total_weight.reshape(full_weight_shape)
        grad_input, grad_weight = None, None
        
        grad_output = grad_output.contiguous()

        if ctx.needs_input_grad[1]:
            assert ctx.compute_weight_gradient
            if backend == "gmm":
                grad_weight = gmm(x, grad_output, batch_sizes, trans_a=True, trans_b=False)
            else:
                grad_weight = torch.matmul(x.transpose(-1, -2), grad_output)
            grad_weight = grad_weight.view(-1, grad_weight.shape[-1])

        if ctx.needs_input_grad[0]:
            if backend == "gmm":
                if grad_input is None:
                    grad_input = gmm(grad_output, total_weight, batch_sizes, trans_a=False, trans_b=True)
            else:
                grad_input = torch.matmul(grad_output, total_weight.transpose(-1, -2))

        del total_weight

        return grad_input, grad_weight, None, None, None


class DeepseekV3GroupedFFN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        num_groups: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: Optional[int] = 1,
        hidden_act: Optional[str] = "silu",
        mlp_layer_fusion: Optional[bool] = False,
    ):
        super().__init__()
        self.act_fn = ACT2FN[hidden_act]
        self.mlp_layer_fusion = mlp_layer_fusion
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        if self.mlp_layer_fusion:
            assert bias is False, "mlp_layer_fusion only support with bias=False."
            self.fused_w1_w3 = GroupedLinear(
                hidden_size,
                intermediate_size * 2,
                bias,
                num_groups,
                device,
                dtype,
            )
            self.down_proj = GroupedLinear(
                intermediate_size,
                hidden_size,
                bias,
                num_groups,
                device,
                dtype,
            )
        else:
            self.gate_proj = GroupedLinear(
                hidden_size,
                intermediate_size,
                bias,
                num_groups,
                device,
                dtype,
            )
            self.up_proj = GroupedLinear(
                hidden_size,
                intermediate_size,
                bias,
                num_groups,
                device,
                dtype,
            )
            self.down_proj = GroupedLinear(
                intermediate_size,
                hidden_size,
                bias,
                num_groups,
                device,
                dtype,
            )

    def forward(self, x, batch_sizes=None):
        if not self.mlp_layer_fusion:
            w1_o = self.gate_proj(x, batch_sizes)
            w3_o = self.up_proj(x, batch_sizes)
        else:
            w13_o = self.fused_w1_w3(x, batch_sizes)
            w1_o, w3_o = torch.split(w13_o, w13_o.shape[-1] // 2, dim=-1)
        out = self.down_proj(self.act_fn(w1_o) * w3_o, batch_sizes)
        return out


class DeepseekV3FFN(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        bias: bool, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None, 
        multiple_of: Optional[int] = 1,
        hidden_act: Optional[str] = "silu",
        mlp_layer_fusion: Optional[bool] = False,
    ):
        super().__init__()
        self.act_fn = ACT2FN[hidden_act]
        self.mlp_layer_fusion = mlp_layer_fusion
        intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

        if self.mlp_layer_fusion:
            assert bias is False, "mlp_layer_fusion only support with bias=False."
            self.fused_w1_w3 = nn.Linear(
                hidden_size, 
                intermediate_size * 2, 
                bias=bias, 
                device=device, 
                dtype=dtype
            )
            self.down_proj = nn.Linear(
                intermediate_size, 
                hidden_size, 
                bias=bias, 
                device=device, 
                dtype=dtype
            )
        else:
            self.gate_proj = nn.Linear(
                hidden_size, 
                intermediate_size, 
                bias=bias, 
                device=device, 
                dtype=dtype
            )
            self.up_proj = nn.Linear(
                hidden_size, 
                intermediate_size, 
                bias=bias, 
                device=device, 
                dtype=dtype
            )
            self.down_proj = nn.Linear(
                intermediate_size, 
                hidden_size, 
                bias=bias, 
                device=device, 
                dtype=dtype
            )

    def forward(self, x):
        if not self.mlp_layer_fusion:
            w1_o = self.gate_proj(x)
            w3_o = self.up_proj(x)
        else:
            w13_o = self.fused_w1_w3(x)
            w1_o, w3_o = torch.split(w13_o, w13_o.shape[-1] // 2, dim=-1)
        out = self.down_proj(self.act_fn(w1_o) * w3_o)
        return out


def update_e_score_correction_bias(expert_counts_list, expert_biases_list, update_rate):
    """
    Adjust biases for experts
    b_i = b_i + u + sign(e_i)
    note: this is \bar{c_i} - c_i, NOT c_i - \bar{c_i}, which will push the network to
          be maximally unbalanced. Really important to get this part right!!!
    """
    full_expert_counts = torch.stack(expert_counts_list, dim=0).float()
    full_expert_biases = torch.stack(expert_biases_list, dim=0).float()
    avg_count = full_expert_counts.mean(dim=1, keepdim=True)
    error = avg_count.expand_as(full_expert_counts) - full_expert_counts
    full_expert_biases.data += update_rate * torch.sign(error)
    expert_biases_list = full_expert_biases.unbind(dim=0)
    return expert_biases_list


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
    def reset_parameters(self):
        nn.init.ones_(self.weight)


ALL_LAYERNORM_LAYERS.append(DeepseekV3RMSNorm)


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )        
    
    def reset_parameters(self):
        device = self.inv_freq.device
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype()
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.moe_loss_type = config.moe_loss_type
        self.aux_loss_alpha = config.aux_loss_alpha
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), device=config.device, dtype=config.dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states, e_score_correction_bias=None):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )
        
        if e_score_correction_bias is not None:
            scores_for_choice = scores.view(bsz * seq_len, -1) + e_score_correction_bias.unsqueeze(0)
        else:
            scores_for_choice = scores.view(bsz * seq_len, -1)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim = -1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[
            1
        ]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
            )
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(
            tmp_scores, k=self.num_experts_per_tok, dim=-1, sorted=False
        )
        topk_weight = scores.gather(1, topk_idx)

        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        l_aux=None
        if self.moe_loss_type != "none":
            l_aux = self.deepseek_aux_loss(scores, topk_idx, bsz, seq_len)

        return topk_idx, topk_weight, l_aux
    
    def deepseek_aux_loss(self, scores, topk_idx, bsz, seq_len):
        scores_for_aux = scores
        aux_topk = self.num_experts_per_tok
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.moe_loss_type == "seq_aux":
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            scores_for_seq_aux = scores_for_seq_aux / scores_for_seq_aux.sum(dim=2, keepdim=True)
            ce = torch.zeros(bsz, self.n_routed_experts, device=scores.device)
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=scores.device),
            ).div_(seq_len * aux_topk / self.n_routed_experts) 
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.aux_loss_alpha
        else:
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
        return aux_loss


class DeepseekV3MoE(nn.Module):
    def __init__(
            self, 
            config, 
            deterministic_mode: bool = False,
        ):
        super().__init__()
        self.deterministic_mode = deterministic_mode
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.aux_free = config.aux_free
        self.device = config.device
        
        self.ep_group = gpc.get_group(ParallelMode.EXPERT)
        self.ep_size = dist.get_world_size(group=self.ep_group)
        self.ep_rank = dist.get_rank(group=self.ep_group) 
        self.num_local_experts = self.n_routed_experts // self.ep_size
        self.local_expert_indices = [self.ep_rank * self.num_local_experts + i for i in range(self.num_local_experts)]

        self.sort_input_by_local_experts = None
        self.num_out_tokens = None
        self.input_splits = None
        self.output_splits = None
        self.restore_output_by_local_experts = None

        self.experts = DeepseekV3GroupedFFN(
            config.hidden_size,
            config.moe_intermediate_size,
            bias=False,
            num_groups=self.num_local_experts,
            device=config.device,
            dtype=config.dtype,
            multiple_of=config.multiple_of,
            hidden_act=config.hidden_act,
            mlp_layer_fusion=config.mlp_layer_fusion
        )
        self.gate = MoEGate(config)
        if self.n_shared_experts is not None:
            self.shared_experts = DeepseekV3FFN(
                config.hidden_size, 
                config.moe_intermediate_size * self.n_shared_experts, 
                bias=config.mlp_bias,
                device=config.device, 
                dtype=config.dtype,
                multiple_of=config.multiple_of,
                hidden_act=config.hidden_act,
                mlp_layer_fusion=config.mlp_layer_fusion
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.aux_free:
            self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))
            self.expert_counts = torch.zeros(self.n_routed_experts, device=self.device)

    def forward(self, hidden_states):
        identity = hidden_states
        if self.aux_free:
            topk_idx, topk_weight, l_aux = self.gate(hidden_states, self.e_score_correction_bias)
        else:
            topk_idx, topk_weight, l_aux = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.deterministic_mode:
            num_local_tokens_per_expert = torch.bincount(topk_idx.view(-1), minlength=self.n_routed_experts)
        else:
            num_local_tokens_per_expert = torch.histc(topk_idx, bins=self.n_routed_experts, min=0, max=self.n_routed_experts)
        if self.aux_free:
            self.expert_counts += num_local_tokens_per_expert
        (dispatched_input, tokens_per_expert) = self.token_permutation( 
            hidden_states, topk_weight, topk_idx, num_local_tokens_per_expert
        )
        expert_output = self.experts(dispatched_input, batch_sizes=tokens_per_expert)
        y = self.token_unpermutation(expert_output, topk_weight)

        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, l_aux

    def _gather_along_first_dim_expert_parallel(self, input_, group):
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input_
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.device("cuda:{}".format(torch.cuda.current_device())))
        dist._all_gather_base(output, input_.contiguous(),group=group)
        return output

    def preprocess(self, num_local_tokens_per_expert) -> torch.Tensor:
        if self.ep_size > 1 or self.num_local_experts > 1:
            self.device_sync_point = "before_ep_alltoall"
        else:
            self.device_sync_point = "before_premute_finish"

        if self.ep_size > 1:
            self.input_splits = ( 
                num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )

            num_global_tokens_per_expert = self._gather_along_first_dim_expert_parallel(
                num_local_tokens_per_expert,
                self.ep_group
            ).reshape(self.ep_size, self.n_routed_experts)
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices]

            self.output_splits = (
                num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True).numpy()
            )
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(self.n_routed_experts)
            num_tokens_per_local_expert = num_local_tokens_per_expert

        num_tokens_per_local_expert = num_tokens_per_local_expert.to(torch.device("cpu"), non_blocking=True)

        if self.num_local_experts > 1 and self.ep_size > 1:
            self.num_global_tokens_per_local_expert_cpu = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            ).to(torch.device("cpu"), non_blocking=True)

        return num_tokens_per_local_expert

    def sort_chunks_by_idxs(self, inputs: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
        inputs = torch.split(inputs, split_sizes.tolist(), dim=0)
        output = torch.cat([inputs[i] for i in sorted_idxs], dim=0)
        return output

    def token_permutation(
        self,
        reshaped_inputs: torch.Tensor,
        expert_weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert_before_capacity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.sort_input_by_local_experts is None:
            self.sort_input_by_local_experts = torch.arange(self.n_routed_experts).reshape(-1, self.num_local_experts).T.ravel().tolist()
        
        assert expert_weights.dim() == 2, "Expected 2D tensor for expert weights"
        assert indices.dim() == 2, "Expected 2D tensor for indices"
        tokens_per_expert = self.preprocess(tokens_per_expert_before_capacity)

        self.hiddden_shape_before_permute = reshaped_inputs.shape

        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = grouped_gemm.ops.permute(
            reshaped_inputs, indices.to(torch.int32), self.num_out_tokens
        )

        if self.device_sync_point == "before_ep_alltoall":
            torch.cuda.current_stream().synchronize()
        global_input_tokens = all_to_all(permutated_local_input_tokens, self.output_splits, self.input_splits, self.ep_group)

        if self.num_local_experts > 1 and self.ep_size > 1:
            global_input_tokens = self.sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert_cpu.ravel(),
                self.sort_input_by_local_experts,
            )
        if self.device_sync_point == "before_premute_finish":
            torch.cuda.current_stream().synchronize()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.restore_output_by_local_experts is None:
            self.restore_output_by_local_experts = torch.arange(self.n_routed_experts).reshape(self.num_local_experts, -1).T.ravel().tolist()
        
        if self.num_local_experts > 1 and self.ep_size > 1:
            hidden_states = self.sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                self.restore_output_by_local_experts,
            )

        permutated_local_input_tokens = all_to_all(hidden_states, self.input_splits, self.output_splits, self.ep_group)
        
        return grouped_gemm.ops.unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            expert_weights.to(torch.float32),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->DeepseekV3
class DeepseekV3FlashAttention2(nn.Module):
    """
    DeepseekV3 flash attention module. This module inherits from `DeepseekV3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        assert config.rope_scaling is None, "Currently we only support rope_scaling=None"

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.q_head_dim, bias=False, device=config.device, dtype=config.dtype
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias, device=config.device, dtype=config.dtype
            )
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank, device=config.device, dtype=config.dtype)
            self.q_b_proj = nn.Linear( 
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            device=config.device, 
            dtype=config.dtype,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank, device=config.device, dtype=config.dtype)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
            device=config.device, 
            dtype=config.dtype,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            device=config.device, 
            dtype=config.dtype,
        )

        self.rotary_emb = DeepseekV3RotaryEmbedding(
            config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            device=config.device,
        )

        self.softmax_scale = self.q_head_dim ** (-0.5)


        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # DeepseekV3FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]

        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (DeepseekV3RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = (
                    self.q_proj.weight.dtype
                    if self.q_lora_rank is None
                    else self.q_a_proj.weight.dtype
                )

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=self.softmax_scale,
        )

        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in DeepseekV3FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV3FlashAttention2(config=config, layer_idx=layer_idx)
        self.is_moe = config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0
        if self.is_moe:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3FFN(
                config.hidden_size, 
                config.intermediate_size,
                bias=config.mlp_bias,
                device=config.device, 
                dtype=config.dtype,
                multiple_of=config.multiple_of,
                hidden_act=config.hidden_act, 
                mlp_layer_fusion=config.mlp_layer_fusion,
            )
        self.input_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=config.device, dtype=config.dtype
        )
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, device=config.device, dtype=config.dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states, moe_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
            moe_loss = None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, moe_loss


DeepseekV3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DeepseekV3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV3_START_DOCSTRING,
)
class DeepseekV3PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DeepseekV3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV3_START_DOCSTRING,
)
class DeepseekV3Model(DeepseekV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

    Args:
        config: DeepseekV3Config
    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        if config.checkpoint>0:
            for layer_id, transformer_block in self.layers.named_children():
                transformer_block = apply_ac_to_transformer_block(transformer_block, config.checkpoint)
                self.layers.register_module(layer_id, transformer_block)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=config.device, dtype=config.dtype)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            raise Exception(f"Currently we only support FA.")

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        moe_losses = []
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs, moe_loss = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            
            if moe_loss is not None:
                moe_losses.append(moe_loss)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            ), moe_losses
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekV3MTPWrapper(nn.Module):
    def __init__(self,
            hidden_size,
            decoder: DeepseekV3DecoderLayer,
            layer_norm_epsilon: float = 1e-5,
            device: Optional[torch.device] = None, 
            dtype: Optional[torch.dtype] = None,
        ):
        super().__init__()
        self.hnorm = DeepseekV3RMSNorm(hidden_size, eps=layer_norm_epsilon, device=device, dtype=dtype)
        self.enorm = DeepseekV3RMSNorm(hidden_size, eps=layer_norm_epsilon, device=device, dtype=dtype)
        self.eh_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.layer = decoder

    def forward(self, previous_hidden_states, hidden_states):
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.enorm(hidden_states)
        return self.layer(
            self.eh_proj(torch.cat([previous_hidden_states, hidden_states], dim=-1))
        )


class DeepseekV3MultiTokenPredictionLayers(nn.Module):
    def __init__(self,
            hidden_size: int,
            decoders: List[DeepseekV3DecoderLayer],
            layer_norm_epsilon: float = 1e-5,
            device: Optional[torch.device] = None, 
            dtype: Optional[torch.dtype] = None,
            pad_token_id: Optional[int] = 0,
            num_nextn_predict_layers: Optional[int] = 1,
        ):
        super().__init__()
        self.mtp_layers = nn.ModuleList(
            [DeepseekV3MTPWrapper(hidden_size, decoders[i], layer_norm_epsilon, device=device, dtype=dtype) for i in range(num_nextn_predict_layers)]
        )
        self.pad_token_id = pad_token_id

    def forward(self, input_ids, hidden_states, shared_embed, shared_norm, shared_head):
        mtp_outputs = []
        moe_losses = []
        for idx, mtp_block in enumerate(self.mtp_layers):
            input_ids = torch.cat([input_ids[:, idx+1:], torch.full((input_ids.size(0), idx+1), self.pad_token_id, dtype=input_ids.dtype, device=input_ids.device)], dim=1)
            outputs, moe_loss = mtp_block(hidden_states, shared_embed(input_ids))
            hidden_states = outputs[0]
            hidden_states = shared_norm(hidden_states)
            mtp_output = shared_head(hidden_states)
            mtp_outputs.append(mtp_output)
            moe_losses.append(moe_loss)
        return mtp_outputs, moe_losses


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Multi Token Prediction
        if config.num_nextn_predict_layers > 0:
            self.mtp = DeepseekV3MultiTokenPredictionLayers(
                config.hidden_size, 
                decoders=[DeepseekV3DecoderLayer(self.config, config.num_hidden_layers + lid) for lid in range(config.num_nextn_predict_layers)], 
                layer_norm_epsilon=config.rms_norm_eps, 
                device=config.device, 
                dtype=config.dtype,
                pad_token_id=config.pad_token_id,
                num_nextn_predict_layers=config.num_nextn_predict_layers, 
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(DeepseekV3_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

        >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, moe_losses = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Multi Token Prediction
        mtp_outputs = None
        if self.config.num_nextn_predict_layers > 0:
            mtp_outputs = self.mtp(input_ids, hidden_states, self.model.embed_tokens, self.model.norm, self.lm_head)
        
        # normal logits
        logits = logits.float()

        # Auxiliary Loss Free Load Balancing via e_score_correction_bias
        if self.config.aux_free:
            expert_counts_list = []
            expert_biases_list = []
            for layer in self.model.layers:
                if layer.is_moe:
                    expert_counts_list.append(layer.mlp.expert_counts)
                    expert_biases_list.append(layer.mlp.e_score_correction_bias)
            expert_biases_list = update_e_score_correction_bias(expert_counts_list, expert_biases_list, self.config.aux_free_update_rate)
            moe_layer_id = 0
            for layer in self.model.layers:
                if layer.is_moe:
                    layer.mlp.e_score_correction_bias = expert_biases_list[moe_layer_id]
                    torch.zero_(layer.mlp.expert_counts)
                    moe_layer_id += 1

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (loss,) + (logits,) + outputs[1:] if loss is not None else (logits,) + outputs[1:]
            # Multi Token Prediction
            if mtp_outputs is not None:
                return output, moe_losses, mtp_outputs
            else:
                return output, moe_losses

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
