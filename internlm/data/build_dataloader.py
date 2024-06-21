# Copyright (c) InternLM. All rights reserved.
from functools import partial
import sys
import datasets

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from datasets.distributed import split_dataset_by_node
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.data.streaming.batch_sampler import StreamingStaticBatchSampler
from internlm.data.tokenized.batch_sampler import (
    StaticBatchSampler,
    get_dpsampler_dataloader,
)
from internlm.data.tokenized.collaters import jsonl_ds_collate_fn, packed_collate_fn
from internlm.data.tokenized.dataset import get_dataset_dict
from internlm.data.tokenized.dummy_dataset import RandomDataset
from internlm.data.tokenized.dummy_dataset_multimodal import RandomDatasetMultimodal
from internlm.data.tokenized.packed_dataset import (
    PackedDatasetWithCut,
    PackedDatasetWithoutCuSeqlen,
    PackedDatasetWithPadForMultimodal,
    get_packed_dataset_without_short_length,
)
from internlm.data.utils import get_dataset_type_ids_map
from internlm.utils.logger import get_logger

# global llm logger
logger = get_logger(__file__)


def get_tokenized_train_loader_items(data_cfg):
    """Get the training data loader for tokenized dataset."""
    if data_cfg.get("train_folder", None) is None:
        if data_cfg.get("is_multimodal", False):
            image_token_size = int(data_cfg.image_size // data_cfg.patch_size) ** 2
            train_ds = RandomDatasetMultimodal(
                num_samples=100000,
                max_len=data_cfg.seq_len,
                image_size=data_cfg.image_size,
                image_token_size=image_token_size,
            )
            train_ds = PackedDatasetWithPadForMultimodal(
                train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
            )
        else:
            train_ds = RandomDataset(
                num_samples=1000000, max_len=data_cfg.seq_len, fixed_seqlen=data_cfg.fixed_random_dataset_seqlen
            )

            if data_cfg.pack_sample_into_one:
                train_ds = PackedDatasetWithoutCuSeqlen(
                    train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
                )
            else:
                train_ds = PackedDatasetWithCut(
                    train_ds, max_length_per_sample=data_cfg.seq_len, packed_length=data_cfg.packed_length
                )
    else:
        train_ds = get_packed_dataset_without_short_length(
            folder=data_cfg.train_folder,
            packed_length=data_cfg.packed_length,
            max_length_per_sample=data_cfg.seq_len,
            show_progress=dist.get_rank() == 0,
            min_length=data_cfg.get("min_length", 0),
            min_length_dict=data_cfg.get("min_length_dict", None),
            pack_sample_into_one=data_cfg.get("pack_sample_into_one", False),
        )

    train_sampler = StaticBatchSampler(
        train_ds.datasets if isinstance(train_ds, ConcatDataset) else [train_ds],
        batch_size=data_cfg.micro_num,
        rampup_batch_size=data_cfg.rampup_batch_size,
        micro_bsz=data_cfg.micro_bsz,
        seed=data_cfg.get("seed", 1024),
        drop_last=True,
        data_rank=gpc.get_local_rank(ParallelMode.DATA),
        data_world_size=gpc.get_world_size(ParallelMode.DATA),
    )
    train_collate_fn = partial(packed_collate_fn, packed_length=data_cfg.packed_length)

    return train_ds, train_sampler, train_collate_fn


def get_tokenized_valid_loader_items(data_cfg):
    """Get the validation data loader for tokenized dataset."""
    if not data_cfg.valid_folder:
        if data_cfg.get("is_multimodal", False):
            image_token_size = int(data_cfg.image_size // data_cfg.patch_size) ** 2
            valid_ds = RandomDatasetMultimodal(
                num_samples=gpc.get_world_size(ParallelMode.DATA) * 500,
                max_len=data_cfg.seq_len,
                image_size=data_cfg.image_size,
                image_token_size=image_token_size,
            )
        else:
            valid_ds = RandomDataset(
                num_samples=gpc.get_world_size(ParallelMode.DATA) * 500,
                max_len=data_cfg.seq_len,
                fixed_seqlen=data_cfg.fixed_random_dataset_seqlen,
            )
    else:
        valid_ds = get_dataset_dict(folder=data_cfg.valid_folder, split="")

    if not isinstance(valid_ds, dict):
        valid_ds = {"val": valid_ds}

    valid_collate_fn = partial(jsonl_ds_collate_fn, max_length_per_sample=data_cfg.seq_len)

    return valid_ds, valid_collate_fn

def create_hf_dataloader(data_cfg, split='train'):
    def hf_collate_fn(batch, micro_num, micro_bsz, seq_len):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        for b in batch:
            attention_mask = b['attention_mask']
            input_ids = b['input_ids']
            input_ids = torch.abs(input_ids * attention_mask)
            input_ids = torch.nn.functional.pad(input_ids, (0, seq_len - len(input_ids)), mode='constant', value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, seq_len - len(attention_mask)), mode='constant', value=0)
            label = torch.tensor([w if w > 0 else -100 for w in input_ids.tolist()][1:]+[-100])
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(label)
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        labels = torch.stack(labels_list)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "type_ids": torch.zeros(micro_num, micro_bsz, seq_len, dtype=torch.int64)}, labels

    train_dataset = HuggingFaceStreamingDataset(data_cfg.hf_dataset_name, data_cfg.hf_tokenizer_name, data_cfg.seq_len, split)
    train_batch_sampler = StreamingStaticBatchSampler(batch_size = data_cfg.micro_num * data_cfg.micro_bsz, rampup_batch_size = data_cfg.rampup_batch_size)
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=data_cfg.get("num_worker", 4),
        pin_memory=True,
        collate_fn=partial(hf_collate_fn, micro_num=data_cfg.micro_num, micro_bsz=data_cfg.micro_bsz, seq_len=data_cfg.seq_len),
        persistent_workers=data_cfg.get("num_worker", 4) > 0,
    )
    return train_dl

class HuggingFaceStreamingDataset(Dataset):
    def __init__(self, dataset_name, tokenizer_name, model_max_length, split='train', buffer_size=1000):
        self.dataset = datasets.load_dataset(dataset_name, split=split, streaming=True)
        self.dataset = split_dataset_by_node(self.dataset, rank=gpc.get_local_rank(ParallelMode.DATA), world_size=gpc.get_world_size(ParallelMode.DATA))
        self.buffer_size = buffer_size
        self.senior_iterator = iter(self)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
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
        texts = [sample['text'] for sample in samples]
        tokenized_outputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        for i in range(len(samples)):
            yield {key: tokenized_outputs[key][i] for key in tokenized_outputs}

    def __getitem__(self, _):
        return next(self.senior_iterator)


def build_train_loader_with_data_type():
    """
    Build and return the training data loader based on data type.

    Returns: A tuple of (train_dl, dataset_types).
    """
    data_cfg = gpc.config.data

    if data_cfg.type == "hf":
        train_dl = create_hf_dataloader(data_cfg)
        return train_dl, ["en"]

    train_folder = data_cfg.get("train_folder", None)
    dataset_types = list(get_dataset_type_ids_map(train_folder).keys()) if train_folder else ["en", "cn", "code"]

    if data_cfg.type == "tokenized":
        train_ds, train_sampler, train_collate_fn = get_tokenized_train_loader_items(data_cfg)
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    # Create the training data loader
    train_dl = DataLoader(
        dataset=train_ds,
        batch_sampler=train_sampler,
        num_workers=data_cfg.get("num_worker", 4),
        pin_memory=True,
        collate_fn=train_collate_fn,
        persistent_workers=data_cfg.get("num_worker", 4) > 0,
    )

    return train_dl, dataset_types


def build_valid_loader_with_data_type():
    """Generate and return the validation data loader based on data type."""

    data_cfg = gpc.config.data

    if data_cfg.type == "hf":
        return None

    if data_cfg.type == "tokenized":
        valid_ds, valid_collate_fn = get_tokenized_valid_loader_items(data_cfg)
    else:
        raise ValueError(f"dataset type {data_cfg.type} is not supported")

    if valid_ds is None:
        return None

    val_dls = {}
    for val_name, ds in valid_ds.items():
        # making the batch_size of validate larger can speed up the evaluation, but it should not be too large,
        # otherwise too much data may be dropped
        batch_size = min(
            data_cfg.valid_micro_num * data_cfg.micro_bsz, len(ds) // gpc.get_world_size(ParallelMode.DATA)
        )
        batch_size = batch_size // data_cfg.micro_bsz * data_cfg.micro_bsz

        if batch_size == 0 and gpc.is_rank_for_log():
            logger.info(f"skip validate {val_name}.")
            continue

        val_dls[val_name] = get_dpsampler_dataloader(
            ds,
            shuffle=False,
            num_workers=data_cfg.get("num_worker", 0),
            batch_size=batch_size,
            collate_fn=valid_collate_fn,
            drop_last=True,
        )  # drop_last=True, otherwise it may cause problems in the last batch

        if gpc.is_rank_for_log():
            logger.info(
                f"load validation dataset {val_name} with valid batch size {str(batch_size)} and "
                f"samples {str(len(val_dls[val_name]))}."
            )

    return val_dls
