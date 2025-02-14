import torch
from torch import nn

from internlm.accelerator import get_accelerator

internlm_accelerator = get_accelerator()


class CrossEntropyWriteInPython(torch.autograd.Function):
    """baseline for unit test."""

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(ctx, logits, target, ignore_idx):
        # (1) cal mask
        ignore_mask = target == ignore_idx
        target[ignore_mask] = 0

        # (2) safe softmax for logist
        logits_max = torch.max(logits, dim=-1)[0]
        logits = logits - logits_max.unsqueeze(dim=-1)

        # (3) cal predicted_logits
        vocab_size = logits.shape[-1]
        logits_2d = logits.view(-1, vocab_size)
        target = target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits = logits_2d[arange_1d, target].clone().contiguous().view_as(target)

        # (4) softmax
        exp_logits = logits
        torch.exp(logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        # (5) Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # (6) cal log
        sum_exp_logits = torch.log(sum_exp_logits)

        # (7) cal loss
        loss = sum_exp_logits - predicted_logits

        # (8) apply ignore_mask
        loss[ignore_mask] = 0.0
        ctx.save_for_backward(exp_logits, target, ignore_mask)
        return loss

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad_output):
        # The deriving of cross entropy ref:
        # https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/
        softmax, target, ignore_mask = ctx.saved_tensors

        # Add the gradient from matching classes(which is indicate by target).
        grad_input = softmax
        grad_2d = grad_input.view(-1, softmax.shape[-1])
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        grad_2d[arange_1d, target] -= 1.0

        grad_input.mul_(grad_output.unsqueeze(dim=-1))  # elementwise multiplication
        grad_input[ignore_mask] = 0.0  # set ignore token loss as 0.

        return grad_input, None, None, None


class CrossEntropyPython(nn.Module):
    """
    Baseline for unit test. Please do not use this class directly.
    """

    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        loss = CrossEntropyWriteInPython.apply(logits, target, self.ignore_index)
        if self.reduction == "mean":
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss
