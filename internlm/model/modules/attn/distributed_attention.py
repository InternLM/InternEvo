import torch
from typing import Any, Tuple
import torch.distributed as dist
from torch import Tensor

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
        local_attention: torch.nn.Module,
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
        if gpc.is_evaluating is True:
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
