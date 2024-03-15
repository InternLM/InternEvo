from typing import Optional, Tuple

import numpy as np
import stk
import torch
from megablocks import ops

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.moe.base_layer import BaseMoELayer
from internlm.model.moe.megablock.megablock_moe import MegaBlockMoE
from internlm.model.moe.megablock.mlp import MegaBlockGroupedFeedForward
from internlm.model.moe.megablock.utils import promote_scalar
from internlm.utils.registry import MODEL_INITIALIZER


@MODEL_INITIALIZER.register_module(module_name="MegaBlock-D")
class MegaBlockdMoE(MegaBlockMoE):
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

    def __init__(  # pylint: disable=W0231
        self,
        hidden_size: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        num_experts: int,
        top_k: int = 1,
        parallel_mode: str = "tensor",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        multiple_of: int = 256,
    ) -> None:
        assert gpc.expert_parallel_size == 1, "do not support expert parallel"
        self.top_k = top_k
        self.num_experts = num_experts

        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.ffn_dim = multiple_of * ((int(hidden_size * gpc.config.model.mlp_ratio) + multiple_of - 1) // multiple_of)
        assert self.ffn_dim % tp_size == 0
        if parallel_mode == "tensor":
            self.ffn_dim_per_row = self.ffn_dim // tp_size // ep_size
        else:
            self.ffn_dim_per_row = self.ffn_dim // ep_size
        BaseMoELayer.__init__(  # pylint: disable=W0233
            self,
            torch.nn.Linear(hidden_size, num_experts, bias=False),
            MegaBlockGroupedFeedForward(
                hidden_size,
                (self.ffn_dim // tp_size) * (num_experts // ep_size),
                parallel_mode,
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
        self.blocking = 128
        self.quantize_scatter_num_bits = -1

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (self.ffn_dim * (self.num_experts // ep_size)) // self.blocking
        self.transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))), 1)

        # re-init the number of experts in each device
        self.num_local_experts = num_experts // ep_size

        self.forward_fn = self._forward

    def sparse_transpose(
        self, size: int, row_indices, column_indices
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_columns = size[1] // self.blocking

        # Sort row indices by column indices to get the transposed matrix's
        # column indices.
        #
        # NOTE: Our sort operation uses the same width indices as the input
        # values. To avoid overflow when we have large activation matrices
        # we cast to 32-bit before sorting.
        _, gather_indices = ops.sort(column_indices.int(), self.transpose_sort_end_bit)

        # There are a constant number of blocks in every row of the sparse
        # matrix. A blocks offset is:
        #
        # row_index * blocks_per_row + column_index % blocks_per_row
        #
        # Once we have the block offsets ordered for transposition we can
        # divide by blocks_per_row to get the transposed column indices.
        column_indices_t = row_indices.gather(0, gather_indices.long())
        block_offsets_t = gather_indices.int()

        zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
        nnz_per_column = ops.histogram(column_indices, block_columns)
        nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
        offsets_t = torch.cat([zero, nnz_per_column])
        return column_indices_t, offsets_t, block_offsets_t

    def topology(self, x: torch.Tensor, padded_bins: torch.Tensor) -> stk.Matrix:
        padded_tokens, _ = x.size()
        assert padded_tokens % self.blocking == 0
        assert self.ffn_dim_per_row % self.blocking == 0

        # Offsets for the sparse matrix. All rows have the
        # same number of nonzero blocks dictated by the
        # dimensionality of a single expert.
        block_rows = padded_tokens // self.blocking
        blocks_per_row = self.ffn_dim_per_row // self.blocking
        offsets = torch.arange(
            0,
            block_rows * blocks_per_row + 1,
            blocks_per_row,
            dtype=torch.int32,
            device=x.device,
        )

        # Indices for the sparse matrix. The indices for
        # the intermediate matrix are dynamic depending
        # on the mapping of tokens to experts.
        column_indices = ops.topology(padded_bins, self.blocking, block_rows, blocks_per_row)

        # TODO(tgale): This is unused. Remove the need for this in stk.
        # For now, use meta init to save the device memory.
        data = torch.empty(
            column_indices.numel(),
            self.blocking,
            self.blocking,
            dtype=x.dtype,
            device="meta",
        )
        shape = (padded_tokens, self.ffn_dim_per_row * self.num_experts)
        row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
        column_indices_t, offsets_t, block_offsets_t = self.sparse_transpose(shape, row_indices, column_indices)
        return stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
        )

    def indices_and_padded_bins(
        self, selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        selected_experts = selected_experts.int()
        bin_ids, indices = ops.sort(selected_experts, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        tokens_per_expert = ops.histogram(selected_experts, self.num_experts)

        # Round the token counts up to the block size used in
        # the matrix muliplications. Caculate the starting
        # position of each bin.
        padded_tokens_per_expert = ops.round_up(tokens_per_expert, self.blocking)
        padded_bins = ops.inclusive_cumsum(padded_tokens_per_expert, 0)
        padded_bins = promote_scalar(padded_bins)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = promote_scalar(bins)
        return indices, bin_ids, bins, padded_bins, tokens_per_expert

    def _forward(self, x, expert_weights, top_experts) -> torch.Tensor:
        with torch.no_grad():
            (indices, bin_ids, bins, padded_bins, tokens_per_expert) = self.indices_and_padded_bins(top_experts)

        # Permute tokens and pad to prepare expert computation
        # (top_k * sequence_length + padding, model_dim)
        x = ops.padded_gather(x, indices, bin_ids, bins, padded_bins, self.top_k)

        # Create the sparse matrix topology
        with torch.no_grad():
            topo = self.topology(x, padded_bins)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = self.experts(x, topo=topo)

        # Permute back and remove padding
        # (top_k * sequence_length, model_dim)
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            padded_bins,
            self.top_k,
            self.quantize_scatter_num_bits,
        )

        return x, tokens_per_expert.flatten()
