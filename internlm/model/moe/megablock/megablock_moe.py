from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from megablocks import ops

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.moe.base_layer import BaseMoELayer
from internlm.model.moe.megablock.mlp import MegaBlockFeedForward
from internlm.model.moe.utils import all_to_all
from internlm.utils.registry import MODEL_INITIALIZER


@MODEL_INITIALIZER.register_module(module_name="MegaBlock")
class MegaBlockMoE(BaseMoELayer):
    """
    Built on the paper and library Megablocks as described in
    https://arxiv.org/abs/2211.15841. This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(
        self,
        hidden_size: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        num_experts: int,
        top_k: int = 1,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        multiple_of: int = 256,
    ) -> None:
        assert not gpc.config.parallel.sequence_parallel, "do not support sequence parallel"
        self.top_k = top_k
        self.num_experts = num_experts

        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.ffn_dim = multiple_of * ((int(hidden_size * gpc.config.model.mlp_ratio) + multiple_of - 1) // multiple_of)
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        assert self.ffn_dim % tp_size == 0
        super().__init__(
            torch.nn.Linear(hidden_size, num_experts, bias=False),
            MegaBlockFeedForward(
                hidden_size,
                self.ffn_dim // tp_size,
                num_experts // ep_size,
                device,
                dtype,
            ),
            ep_group,
            ep_size,
            1,
        )

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        self.quantize_scatter_num_bits = -1

        self.forward_fn = self._parallel_forward if gpc.expert_parallel_size > 1 else self._forward

    def expert_capacity(self, tokens, top_k):
        world_size = gpc.get_world_size(ParallelMode.EXPERT)  # mpu.get_expert_parallel_world_size(self.args)
        tokens_per_expert = top_k * tokens * world_size / self.num_experts
        return int(self.capacity_factor * tokens_per_expert)

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if len(bins.size()) == 0 else bins
        return indices, bin_ids, bins, tokens_per_expert

    def _forward(self, x, expert_weights, top_experts) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        with torch.no_grad():
            indices, _, bins, tokens_per_expert = self.indices_and_bins(top_experts)
            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, _ = x.size()
            expert_capacity = self.expert_capacity(tokens, top_k=self.top_k)
            if not self.drop_tokens:
                expert_capacity = torch.max(tokens_per_expert).item()

        out = self.permute_and_compute(x, indices, expert_weights, bins, expert_capacity, top_k=self.top_k)

        return out, tokens_per_expert.flatten()

    def _parallel_forward(self, x, expert_weights, top_experts):
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(top_experts)

            # Pass token count information to the device on which the
            # target expert resides.
            # e.g. tokens_per_expert = (1,2,1,0) in g1
            #      tokens_per_expert = (2,0,2,0) in g2
            # then:parallel_tokens_per_expert = (1,2,2,0) in g1
            #      parallel_tokens_per_expert = (1,0,2,0) in g2
            parallel_tokens_per_expert = torch.empty_like(tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert, tokens_per_expert, group=gpc.get_group(ParallelMode.EXPERT), async_op=True
            )

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        x = x.view(-1, x.shape[-1])  # TODO can be deleted
        x = ops.gather(x, indices, bin_ids, bins, self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            experts_per_rank = self.num_local_experts  # mpu.experts_per_rank(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            world_size = gpc.get_world_size(ParallelMode.EXPERT)  # mpu.get_expert_parallel_world_size(self.args)
            tokens_per_expert = tokens_per_expert.view(
                world_size, experts_per_rank
            )  # ((1,2), (1,0)) in g1, ((2,0),(2,0)) in g2
            parallel_tokens_per_expert = parallel_tokens_per_expert.view(
                world_size, experts_per_rank
            )  # ((1,2), (2,0)) in g1, ((1,0),(2,0)) in g2

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = tokens_per_expert.cpu().sum(dim=-1)
            parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
            recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = all_to_all(
            x, recv_counts, send_counts, gpc.get_group(ParallelMode.EXPERT), async_op=True
        )

        with torch.no_grad():
            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.

            replicate_bins = ops.inclusive_cumsum(parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = replicate_bins.view(1) if len(replicate_bins.size()) == 0 else replicate_bins

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(self.num_experts, dtype=torch.int32, device=indices.device),
                self.num_local_experts,  # mpu.experts_per_rank(self.args),
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0), replicate_bins, tokens_received
            ).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            _, parallel_indices = ops.sort(parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(parallel_tokens_per_expert, 0)
            parallel_bins = parallel_bins.view(1) if len(parallel_bins.size()) == 0 else parallel_bins

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, _ = x.size()
            expert_capacity = self.expert_capacity(tokens, top_k=1)
            if not self.drop_tokens:
                expert_capacity = torch.max(parallel_tokens_per_expert).item()

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_indices,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            top_k=1,
        )

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(parallel_x, send_counts, recv_counts, gpc.get_group(ParallelMode.EXPERT))

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(x, indices, bin_ids, expert_weights, bins, self.top_k, self.quantize_scatter_num_bits)
        return x, tokens_per_expert.flatten()

    def permute_and_compute(self, x, indices, expert_weights, bins, expert_capacity, top_k):  # unused  # unused
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.binned_gather(x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = self.experts(x)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(x, indices, expert_weights, bins, top_k)

    def load_balancing_loss(self, tokens_per_expert, expert_scores):
        """Calculate the load balancing loss contribution."""
        assert len(expert_scores.size()) == 2
        tokens, num_experts = expert_scores.size()
        assert num_experts == self.num_experts
        assert len(tokens_per_expert.size()) == 1
        (num_experts,) = tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.top_k)
        return scale * torch.dot(tokens_per_expert.to(expert_scores.dtype), expert_scores.mean(dim=0))

    def forward(self, *inputs) -> torch.Tensor:
        # optional reshape
        x = inputs[0]
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)

        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float)
        # weights, selected_experts: (sequence_length, top-k)
        expert_weights, top_experts = torch.topk(all_probs, self.top_k, dim=-1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()

        x, tokens_per_expert = self.forward_fn(x, expert_weights, top_experts)

        self.l_aux = self.load_balancing_loss(tokens_per_expert, all_probs)

        return x.view(*input_shape)
