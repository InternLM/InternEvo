import itertools
import sys

import datasets
import numpy as np
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from transformers import AutoTokenizer


class HuggingFaceStreamingDataset(Dataset):
    """
    Streaming and on-the-fly tokenized dataset for huggingface
    """

    def __init__(
        self, dataset_name, tokenizer_name, model_max_length, split="train", buffer_size=1000, subset_name=None
    ):
        self.dataset = datasets.load_dataset(dataset_name, data_dir=subset_name, split=split, streaming=True)
        self.dataset = split_dataset_by_node(
            self.dataset, rank=gpc.get_local_rank(ParallelMode.DATA), world_size=gpc.get_world_size(ParallelMode.DATA)
        )
        self.buffer_size = buffer_size
        self.senior_iterator = iter(self)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.tokenizer.model_max_length = model_max_length

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                yield from self._tokenize(buffer)
                buffer = []

        if buffer:
            yield from self._tokenize(buffer)

    def __len__(self):
        return sys.maxsize

    def _tokenize(self, samples):
        texts = [sample["text"] for sample in samples]
        tokenized_outputs = self.tokenizer(texts, truncation=True)
        for i in range(len(samples)):
            assert "input_ids" in tokenized_outputs, "huggingface tokenizer should generate input_ids"
            if len(tokenized_outputs["input_ids"][i]) > 0:
                yield {key: tokenized_outputs[key][i] for key in tokenized_outputs}

    def __getitem__(self, _):
        return next(self.senior_iterator)


class HuggingFacePackedDataset(Dataset):
    """
    Simple packed dataset for huggingface
    """

    def __init__(self, dataset, seq_len, micro_bsz, pad_token_id=0):
        self.dataset = dataset
        self.seq_len = seq_len
        self.micro_bsz = micro_bsz
        self.pad_token_id = pad_token_id
        self.senior_iterator = iter(self)

    def __iter__(self):
        input_ids = []
        cu_seqlens = [0]
        labels = []
        for sample in self.dataset:
            if len(input_ids + sample["input_ids"]) > self.micro_bsz * self.seq_len:
                assert cu_seqlens[-1] <= self.micro_bsz * self.seq_len
                input_ids = input_ids + [self.pad_token_id] * (self.micro_bsz * self.seq_len - len(input_ids))
                cu_seqlens = (
                    cu_seqlens + [self.micro_bsz * self.seq_len]
                    if cu_seqlens[-1] < self.micro_bsz * self.seq_len
                    else cu_seqlens
                )
                labels = labels + [-100] * (self.micro_bsz * self.seq_len - len(labels))
                yield {
                    "input_ids": input_ids,
                    "cu_seqlens": cu_seqlens,
                    "indexes": list(
                        itertools.chain(*[np.arange(l2 - l1) for l1, l2 in zip(cu_seqlens[:-1], cu_seqlens[1:])])
                    ),
                    "labels": labels,
                }
                input_ids = sample["input_ids"]
                cu_seqlens = [0, len(sample["input_ids"])]
                labels = [w if w > 0 else -100 for w in sample["input_ids"]][1:] + [-100]
            else:
                input_ids = input_ids + sample["input_ids"]
                cu_seqlens.append(len(sample["input_ids"]) + cu_seqlens[-1])
                labels = labels + [w if w > 0 else -100 for w in sample["input_ids"]][1:] + [-100]

        if input_ids:
            assert cu_seqlens[-1] <= self.micro_bsz * self.seq_len
            input_ids = input_ids + [self.pad_token_id] * (self.micro_bsz * self.seq_len - len(input_ids))
            cu_seqlens = (
                cu_seqlens + [self.micro_bsz * self.seq_len]
                if cu_seqlens[-1] < self.micro_bsz * self.seq_len
                else cu_seqlens
            )
            labels = labels + [-100] * (self.micro_bsz * self.seq_len - len(labels))
            yield {
                "input_ids": input_ids,
                "cu_seqlens": cu_seqlens,
                "indexes": list(
                    itertools.chain(*[np.arange(l2 - l1) for l1, l2 in zip(cu_seqlens[:-1], cu_seqlens[1:])])
                ),
                "labels": labels,
            }

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, _):
        return next(self.senior_iterator)
