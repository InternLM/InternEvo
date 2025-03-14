import torch
from torch import Tensor
import flux
import math
from typing import Optional

from internlm.model.modules.mlp import new_feed_forward
from internlm.model.moe.dropless_layer import TopKGate
from internlm.utils.utils import TensorParallelMode

from internlm.core.context import global_context as gpc
from internlm.core.context.process_group_initializer import ParallelMode
from internlm.model.moe.base_layer import BaseMoELayer


class FluxMoELayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        activation_type: str = "swiglu",
        drop_and_pad: bool = False,
        drop_policy="probs",
        capacity_factor: float = None,
        noisy_gate_policy: str = None,
        enable_fused_permute: bool = True,
        token_dispatch_policy: str = "alltoall",
        use_grouped_mlp: bool = True,
        deterministic_mode: bool = False,
    ):
        super().__init__()

        assert gpc.config.parallel["expert"]["no_tp"] is True, "no_tp in expert parallel should be true"

        seq_len = gpc.config.data["seq_len"]
        micro_bsz = gpc.config.data["micro_bsz"]
        tp_group = gpc.get_group(ParallelMode.TENSOR)
        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        etp_size = gpc.get_world_size(ParallelMode.EXPERT_TENSOR)
        world_size = gpc.get_world_size(ParallelMode.GLOBAL)
        global_rank = gpc.get_global_rank()

        assert etp_size * ep_size == tp_size == world_size, f"etp_size({etp_size}) * ep_size({ep_size}) should be equal to tp_size({tp_size}) and equal to world_size({world_size})"

        torch.cuda.set_device(global_rank)

        max_ntokens = seq_len * micro_bsz * top_k // etp_size
        initialized = False

        flux.init_flux_shm(tp_group)

        tp_env = flux.DistEnvTPWithEP(
            tp_group=tp_group,
            nnodes=1,
            ep_group=ep_group,
        )

        moe_args = flux.MoeArguments(
            max_ntokens=max_ntokens // top_k,
            hidden=in_features,
            ffn_hidden=hidden_features,
            nexperts=num_experts,
            topk=top_k,
            input_dtype=dtype,
            output_dtype=dtype,
        )

        if not initialized:
            if flux.util.get_arch() >= 90:
                self.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
                self.flux_rs_op = flux.GemmGroupedV3GatherRS(
                    num_experts,
                    max_ntokens,
                    in_features,
                    top_k,
                    global_rank,
                    world_size,
                    etp_size,
                    ep_size,
                    1,
                )
            else:
                self.flux_ag_op = flux.GemmGroupedV2AGScatterOp(tp_env=tp_env, moe_args=moe_args)
                self.flux_rs_op = flux.GemmGroupedV2GatherRSOp(
                    tp_group,
                    num_experts,
                    max_ntokens,
                    in_features,
                    top_k,
                    dtype,
                    etp_size,
                    ep_size,
                    1,
                )
            initialized = True

        torch.distributed.barrier()

        self.w1 = torch.rand((num_experts // ep_size, hidden_features // etp_size, in_features), dtype=dtype, device=device) * 0.01
        self.w2 = torch.rand((num_experts // ep_size, out_features, hidden_features // etp_size), dtype=dtype, device=device) - 0.5

        self.gate = TopKGate(
            in_features,
            num_experts,
            top_k,
            noisy_gate_policy,
        )
        self.exp_counts = None
        self.hidden_features = hidden_features
        self.num_experts = num_experts
        self.topk = top_k
        self.deterministic_mode = deterministic_mode
        self.device = device
        self.dtype = dtype
        self.drop_and_pad = drop_and_pad
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
        self.activation_type = activation_type
        self.etp_size = etp_size
        self.world_size = world_size
        self.global_rank = global_rank
        self.ep_size = ep_size
        self.tp_size = tp_size
        if self.drop_and_pad:
            assert self.capacity_factor is not None

    def forward(self, *inputs: Tensor) -> Tensor:
        d_model = inputs[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_inputs = inputs[0].reshape(-1, d_model)

        gates = self.gate(reshaped_inputs)

        indices, tokens_per_expert_before_capacity = self.topk_softmax_with_capacity(gates)
        l_aux = self.load_balancing_loss(tokens_per_expert_before_capacity, gates)

        splits_gpu = tokens_per_expert_before_capacity.to(self.device)
        splits_cpu = splits_gpu.cpu()
        nexperts_ep = self.num_experts // self.ep_size
        ep_rank = gpc.get_local_rank(ParallelMode.EXPERT)
        nrows_ep = torch.sum(splits_cpu[nexperts_ep * ep_rank : nexperts_ep * (ep_rank + 1)])
        intermediate_output = torch.zeros((nrows_ep, self.hidden_features // self.etp_size), dtype=self.dtype, device=self.device)

        gathered_indices = [
            torch.zeros_like(indices) for _ in range(self.tp_size)
        ]
        torch.distributed.all_gather(gathered_indices, indices)
        scatter_index = torch.cat(gathered_indices, dim=0)

        # MLP layer 0 (dispatch and GEMM0)
        self.flux_ag_op.forward(
            inputs_shard=reshaped_inputs,
            weights=self.w1,
            splits_gpu=splits_gpu.to(dtype=torch.int32),
            scatter_index=scatter_index.to(dtype=torch.int32),
            outputs_buf=intermediate_output,
        )
        # Activation
        if self.activation_type == "swiglu":
            ac_func = torch.nn.functional.silu
        else:
            ac_func = torch.nn.functional.gelu

        intermediate_output = ac_func(intermediate_output)

        # MLP layer 1 (GEMM1 and combine)
        mlp_output = self.flux_rs_op.forward_gather_rs(
            input=intermediate_output,
            weight=self.w2,
            splits_cpu=splits_cpu.to(dtype=torch.int32),
            routing_idx=indices.view(-1).to(dtype=torch.int32),
        )

        gathered_output = [
            torch.zeros_like(mlp_output) for _ in range(self.tp_size)
        ]
        torch.distributed.all_gather(gathered_output, mlp_output)
        final_mlp_output = torch.cat(gathered_output, dim=0)

        final_mlp_output = final_mlp_output.unsqueeze(0)

        return final_mlp_output, l_aux

    def topk_softmax_with_capacity(self, gates):
        expert_weights, indices = torch.topk(gates, self.topk, dim=1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        # we compute num_local_tokens_per_expert here. If no drop and padding, num_local_tokens_per_expert should be
        # the final value, otherwise we recompute it in self.process(.)
        # histc(.) can be faster the bincount(.), but will cause non-deterministic behavior
        if self.deterministic_mode:
            num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)
        else:
            num_local_tokens_per_expert = torch.histc(indices, bins=self.num_experts, min=0, max=self.num_experts)

        # without capacity
        if self.capacity_factor is None:
            # shape: [num_token, topk]
            return indices, num_local_tokens_per_expert

        # with capacity
        expert_capacity = self.get_capacity(
            num_tokens=gates.shape[0] * self.topk,
            num_experts=gates.shape[1],
            capacity_factor=self.capacity_factor,
        )
        # TopK selection, Maskout unused experts
        topk_masked_gates = torch.zeros_like(gates).scatter(1, indices, expert_weights)
        topk_mask = torch.zeros_like(gates).scatter(1, indices, 1)
        if self.drop_policy == "probs":
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
        elif self.drop_policy == "position":
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
        if self.drop_and_pad:
            # shape: [num_expert, capacity]
            final_indices = capacity_indices.T.contiguous()
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, indices)
            # shape: [num_token, topk]
            final_indices = indices.clone().masked_fill_(exceed_mask, torch.iinfo(torch.long).max)

        tokens_per_expert_before_capacity = topk_mask.sum(dim=0)


        return final_indices, tokens_per_expert_before_capacity
    
    def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
        capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
        if min_capacity is not None and capacity < min_capacity:
            capacity = min_capacity
        return capacity
    
    def load_balancing_loss(self, num_local_tokens_per_expert, gates):
        """Calculate the load balancing loss contribution."""
        assert len(gates.size()) == 2
        tokens, num_experts = gates.size()
        assert num_experts == self.num_experts
        assert len(num_local_tokens_per_expert.size()) == 1
        (num_experts,) = num_local_tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.topk)
        return scale * torch.dot(num_local_tokens_per_expert.to(gates.dtype), gates.mean(dim=0))

