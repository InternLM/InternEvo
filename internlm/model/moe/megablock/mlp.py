import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.moe.megablock.utils import (
    act_fn,
    dsd_nn,
    sdd_nt,
    tensor_parallel_bmm,
)
from internlm.model.utils import Silu


class MegaBlockFeedForward(nn.Module):
    """
    Feed forward using megablock kernel
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_local_experts: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # merged expert weights, all of size  (ffn_dim * n_experts, model_dim)
        self.w1 = nn.Parameter(torch.empty(num_local_experts, in_features, hidden_features, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(num_local_experts, in_features, hidden_features, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(num_local_experts, hidden_features, in_features, device=device, dtype=dtype))

    def forward(self, x):
        # TODO w2 and w3 should swap
        w1_o = tensor_parallel_bmm(x, self.w1, group=gpc.get_group(ParallelMode.TENSOR))
        w3_o = tensor_parallel_bmm(x, self.w3, group=gpc.get_group(ParallelMode.TENSOR))
        out = tensor_parallel_bmm(Silu(w1_o, w3_o), self.w2)
        torch.distributed.all_reduce(out, group=gpc.get_group(ParallelMode.TENSOR))

        return out


class MegaBlockGroupedFeedForward(nn.Module):
    """
    Feed forward using megablock kernel
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        parallel_mode="tensor",
        device=None,
        dtype=None,
    ):
        super().__init__()

        # merged expert weights, all of size  (ffn_dim * n_experts, model_dim)
        self.w1 = nn.Parameter(torch.empty(hidden_features, in_features, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(hidden_features, in_features, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(hidden_features, in_features, device=device, dtype=dtype))

        self.parallel_mode = parallel_mode

    def forward(self, x, topo):
        # TODO w2 and w3 should swap
        w1_o = sdd_nt(x, self.w1, topo, gpc.get_group(ParallelMode.TENSOR), self.parallel_mode)
        w3_o = sdd_nt(x, self.w3, topo, gpc.get_group(ParallelMode.TENSOR), self.parallel_mode)
        out = dsd_nn(act_fn(w1_o, w3_o, topo), self.w2, gpc.get_group(ParallelMode.TENSOR), self.parallel_mode)

        return out
