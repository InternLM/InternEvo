import torch
from torch import nn

from internlm.accelerator import get_accelerator
from internlm.model.ops.cross_entropy import new_cross_entropy

internlm_accelerator = get_accelerator()


class InternLoss(nn.Module):
    """We use a base class to wrap different CrossEntropy implementations
    and unify input and output parameters.

    This class is designed not to rely on gpc, making it easy to transplant.

    Different variants of CrossEntropy, with supporting parallel computation and inplace operations.

    If parallel_output is False, the output will gather head's output, only 'FlashCrossEntropyLoss' and
    'CrossEntropyApexVocabParallel' support it.
    """

    def __init__(
        self,
        parallel_output=False,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        inplace_backward=True,
        op_type="py_vocab_parallel",
    ) -> None:
        super().__init__()

        if label_smoothing is not None:
            if label_smoothing != 0:
                print(f"use label_smoothing: {label_smoothing}", flush=True)
        else:
            label_smoothing = 0

        self.label_smoothing = label_smoothing

        self.reduction = reduction
        self.ignore_index = ignore_index
        self.op_type = op_type

        assert self.reduction in [
            "mean",
            "none",
        ], f"Only support reduction is mean/none, but the passed in reduction is {self.reduction}"

        # In order to facilitate the calculation of loss for different datasets, we set reduction as 'none',
        # and do loss reduction ourselves.
        self.loss_fn = new_cross_entropy(
            op_type=op_type,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            parallel_output=parallel_output,
            inplace_backward=inplace_backward,
            reduction="none",
        )

    def forward(self, *args):
        if len(args) == 3:
            # residual is to match prenorm
            logits, _, labels = args
        elif len(args) == 2:
            # When using postnorm
            logits, labels = args
        else:
            raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
        shift_logits = logits.contiguous().view(-1, logits.size(-1))
        shift_labels = labels.contiguous().view(-1)

        with torch.autocast(device_type=internlm_accelerator.get_backend_name()):
            loss_list = self.loss_fn(
                shift_logits, shift_labels
            )  # There is no need to consider the ignore_index problem here, because the loss calculation will be
            # # calculated through the calculation range, and -100 must be outside this range, so there is no problem

        cond = shift_labels != self.ignore_index
        if self.reduction == "mean":
            # This loss is only for one dp rank.
            loss = loss_list.sum() / (cond).sum()
        else:
            loss = loss_list

        return loss
