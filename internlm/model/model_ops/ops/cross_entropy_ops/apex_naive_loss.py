import torch
from torch import nn

from internlm.accelerator import get_accelerator

try:
    import xentropy_cuda_lib
except (ImportError, ModuleNotFoundError):
    has_xentropy_cuda_lib = False
else:
    has_xentropy_cuda_lib = True


internlm_accelerator = get_accelerator()


class SoftmaxCrossEntropyLossFn(torch.autograd.Function):
    """
    Adapt from: https://github.com/NVIDIA/apex/blob/master/apex/contrib/xentropy/softmax_xentropy.py
    Inplace backward is supported, but loss calculation of vocab parallel is not supported.
    NOTE: it should be noted that when the pack_length exceeds 40K, the loss will not decrease.
    """

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(ctx, logits, labels, smoothing=0.0, padding_idx=0, inplace_backward=False):
        losses, max_log_sum_exp = xentropy_cuda_lib.forward(logits, labels, smoothing)
        losses.masked_fill_(labels == padding_idx, 0)
        ctx.save_for_backward(logits, max_log_sum_exp, labels)
        ctx.smoothing = smoothing
        ctx.padding_idx = padding_idx
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad_loss):
        logits, max_log_sum_exp, labels = ctx.saved_tensors
        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels == ctx.padding_idx, 0)
        grad_logits = xentropy_cuda_lib.backward(
            grad_loss, logits, max_log_sum_exp, labels, ctx.smoothing, ctx.inplace_backward
        )
        return grad_logits, None, None, None, None


class CrossEntropyLossApex(nn.Module):
    """
    Inplace backward is supported, but loss calculation of vocab parallel is not supported.
    NOTE: it should be noted that when the pack_length exceeds 40K, the loss will not decrease.
    """

    def __init__(self, ignore_index=-100, reduction="mean", label_smoothing=0.0, inplace_backward=False):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")

        assert (
            has_xentropy_cuda_lib is True
        ), "The 'xentropy_cuda_lib' package which CrossEntropyLossApex needed was not found in your environment!"
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward

    def forward(self, logits, target):
        # assert logits.is_cuda and target.is_cuda

        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = SoftmaxCrossEntropyLossFn.apply(
            logits, target, self.label_smoothing, self.ignore_index, self.inplace_backward
        )
        if self.reduction == "mean":
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss
