import torch
import torch.distributed as dist
from torch import nn

from internlm.accelerator import get_accelerator

internlm_accelerator = get_accelerator()


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """Adapt from: https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/cross_entropy.py
    Supports vocab parallel loss calculation, but does not support inplace backward.
    NOTE: This class is different from the original Apex implementation. Apex will calculate the loss of
        ignore_index and flashCrossEntropy will set it to 0. InterEvo adapts the second approach.
    """

    @staticmethod
    @internlm_accelerator.amp.custom_fwd
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0, process_group=None):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        if process_group is not None and dist.get_world_size(process_group) > 1:
            torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=process_group)
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        # get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        if process_group is not None and dist.get_world_size(process_group) > 1:
            rank = dist.get_rank(process_group)
            # world_size = dist.get_world_size(process_group)
            part_len = vocab_parallel_logits.shape[-1]
            vocab_start_index, vocab_end_index = part_len * rank, part_len * (rank + 1)
        else:
            vocab_start_index, vocab_end_index = 0, vocab_parallel_logits.shape[-1]

        # vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        ignore_mask = target == -100
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        # All reduce is needed to get the chunks from other GPUs.
        if process_group is not None and dist.get_world_size(process_group) > 1:
            torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=process_group)

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        if process_group is not None and dist.get_world_size(process_group) > 1:
            torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=process_group)

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Loss = log(sum(exp(logits))) - predicted-logit.
        sum_exp_logits = torch.log(sum_exp_logits)
        loss = sum_exp_logits - predicted_logits
        loss[ignore_mask] = 0.0

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            r"""
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, ignore_mask)

        return loss

    @staticmethod
    @internlm_accelerator.amp.custom_bwd
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d, ignore_mask = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax  # s_{k}
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        grad_input[ignore_mask] = 0.0  # set ignore token loss as 0.

        return grad_input, None, None, None


class CrossEntropyApexVocabParallel(nn.Module):
    """Adapt from: https://github.com/NVIDIA/apex/blob/master/apex/transformer/tensor_parallel/cross_entropy.py
    Supports vocab parallel loss calculation, but does not support inplace backward.
    """

    def __init__(
        self, ignore_index=-100, reduction="mean", label_smoothing=0.0, process_group=None, inplace_backward=False
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        assert inplace_backward is False, "does not support inplace backward"
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.process_group = process_group

    def forward(self, vocab_parallel_logits, target):
        # assert vocab_parallel_logits.is_cuda and vocab_parallel_logits.is_cuda

        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, self.label_smoothing, self.process_group)
        if self.reduction == "mean":
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss
