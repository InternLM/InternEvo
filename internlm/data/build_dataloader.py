# Copyright (c) InternLM. All rights reserved.
from functools import partial

import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.data.tokenized.batch_sampler import (
    StaticBatchSampler,
    get_dpsampler_dataloader,
)
from internlm.data.tokenized.collaters import jsonl_ds_collate_fn, packed_collate_fn
from internlm.data.tokenized.dataset import get_dataset_dict
from internlm.data.tokenized.dummy_dataset import RandomDataset
from internlm.data.tokenized.packed_dataset import (
    PackedDatasetWithCut,
    PackedDatasetWithoutCuSeqlen,
    get_packed_dataset_without_short_length,
)
from internlm.data.utils import get_dataset_type_ids_map
from internlm.utils.logger import get_logger

# global llm logger
logger = get_logger(__file__)


def get_tokenized_train_loader_items(data_cfg):
    """Get the training data loader for tokenized dataset."""
    if data_cfg.get("train_folder", None) is None:
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


def build_train_loader_with_data_type():
    """
    Build and return the training data loader based on data type.

    Returns: A tuple of (train_dl, dataset_types).
    """
    data_cfg = gpc.config.data
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
