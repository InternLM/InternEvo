import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


# Adapted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/core/ \
# sequence_parallel/cross_entropy.py
class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):
    """
    Cross Entropy module for isp.
    """

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, reduction, label_smoothing=0.0):  # pylint: disable=W0613
        sp_size = gpc.get_world_size(ParallelMode.TENSOR)

        # reshape
        # vocab_seq_parallel_logits: [B * (S/P), V] -> [B, S/P, V]
        # target: [B * S/P] -> [B, S/P]
        bsz = gpc.config.data.micro_bsz if gpc.config.data.use_packed_dataset is False else 1
        vocab_seq_parallel_logits = vocab_seq_parallel_logits.view(bsz, -1, gpc.config.model.vocab_size)
        target = target.view(bsz, -1)

        # transpose
        # vocab_seq_parallel_logits: [B, S/P, V] -> [S/P, B, V]
        # target: [B, S/P] -> [S/P, B]
        # return: [S, B]
        vocab_seq_parallel_logits = vocab_seq_parallel_logits.transpose(0, 1).contiguous()
        target = target.transpose(0, 1).contiguous()

        ctx.seqlen = vocab_seq_parallel_logits.size(0) * sp_size
        batch_size = vocab_seq_parallel_logits.size(1)

        # Need softmax for backward
        softmax = torch.nn.functional.softmax(vocab_seq_parallel_logits, dim=-1)
        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        loss = torch.nn.functional.nll_loss(softmax.log().view(-1, ctx.vocab_size), target.view(-1), reduction="none")

        loss_all = torch.empty(
            ctx.seqlen, batch_size, dtype=vocab_seq_parallel_logits.dtype, device=vocab_seq_parallel_logits.device
        )

        torch.distributed.all_gather_into_tensor(loss_all, loss, group=gpc.get_group(ParallelMode.TENSOR))

        # [s b] => [b, s]
        loss_all = loss_all.transpose(0, 1).contiguous()

        ctx.save_for_backward(softmax, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        # transpose
        grad_output = grad_output.transpose(0, 1).contiguous()

        step_seqlen = ctx.seqlen // gpc.get_world_size(ParallelMode.TENSOR)
        sp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        grad_output_part = grad_output[step_seqlen * sp_rank : step_seqlen * (sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        # transpose
        grad_input = grad_input.transpose(0, 1).contiguous()
        # reshape
        grad_input = grad_input.view(-1, gpc.config.model.vocab_size)

        return grad_input, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)


class VocabSequenceParallelCrossEntropyLoss(nn.Module):
    """
    Cross Entropy module for isp.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0,
        process_group=None,
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.process_group = process_group

    def loss_mean_func(self, output_tensor):
        losses = output_tensor.float()
        loss = torch.sum(losses.view(-1)) / losses.numel()

        # TODO: allreduce loss in dp group

        return loss

    def forward(self, _input, target):
        assert _input.is_cuda and target.is_cuda

        _loss_list = vocab_sequence_parallel_cross_entropy(_input, target, self.label_smoothing)

        if self.reduction == "mean":
            loss = self.loss_mean_func(_loss_list)
            return loss

        return _loss_list.view(-1)
