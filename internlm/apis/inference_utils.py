import torch

from internlm.core.context import ParallelMode  # noqa: E402
from internlm.core.context import global_context as gpc  # noqa: E402
from internlm.core.parallel.comm.utils import _gather as gather


class InferenceParams:
    """
    Intermediate cache objects for inference
    """

    def __init__(
        self,
        max_sequence_len,
        max_batch_size,
        sequence_len_offset=0,
        batch_size_offset=0,
        key_value_memory_dict: dict = None,
        lengths_per_sample=None,
        attention_mask=None,
        window_size=None,
    ) -> None:

        self.max_sequence_len: int = max_sequence_len
        self.max_batch_size: int = max_batch_size
        self.sequence_len_offset: int = sequence_len_offset
        self.batch_size_offset: int = batch_size_offset
        if key_value_memory_dict is None:
            key_value_memory_dict = {}
        self.key_value_memory_dict: dict = key_value_memory_dict
        self.fused_ft_kernel: bool = False
        self.lengths_per_sample = lengths_per_sample
        self.attention_mask = attention_mask
        self.full_attention_mask = attention_mask
        self.window_size = window_size

    def reorder_state(self, indices):
        if self.lengths_per_sample is not None:
            self.lengths_per_sample = self.lengths_per_sample.index_select(index=indices, dim=0)
        for key, value in list(self.key_value_memory_dict.items()):
            value = value.index_select(index=indices, dim=0)
            self.key_value_memory_dict[key] = value

    def set_batch_offset(self, offset, bsz):
        """Called by `BaseScheduler._load_micro_batch`.
        when micro-batch is enabled, the working attention mask is only a view of `full_attention_mask`
        """
        self.batch_size_offset = offset
        self.attention_mask = self.full_attention_mask[offset : offset + bsz]

    def set_attention_mask(self, mask):
        """useful when generate using Engine/trainer rather than directly using model"""
        self.full_attention_mask = mask


def process_parallel_output(model_output):
    # 1. concat
    if gpc.is_last_rank(ParallelMode.PIPELINE):
        if not isinstance(model_output, torch.Tensor):
            model_output = torch.cat(model_output, dim=0)
    else:
        return None

    # gather tp parallel output
    if gpc.config.model.parallel_output and gpc.is_initialized(ParallelMode.TENSOR):
        return gather(model_output, ParallelMode.TENSOR, -1)
    else:
        return model_output
