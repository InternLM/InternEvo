#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import random
from typing import Iterator, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

T_co = TypeVar("T_co", covariant=True)


class DataParallelSampler(Sampler):
    """A data sampler for distributed data parallelism.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The Dataset for sampling.
        shuffle (bool, optional): Whether to shuffle data, defaults to False.
        seed (int, optional): The random seed used for sampling, defaults to 0.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
    """

    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = gpc.get_world_size(ParallelMode.DATA)
        self.rank = gpc.get_local_rank(ParallelMode.DATA)
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # type: ignore[arg-type]
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas)
                / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

            # update for next epoch so that there is no need to call
            # set_epoch manually
            self.epoch += 1
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class StaticBatchSampler:
    """
    A static batch sampler that generates batches with a fixed micro-batch size.

    Args:
        num_samples (int): The total number of samples in the dataset.
        batch_size (int): The batch size for the current rank. Defaults to 192.
        rampup_batch_size (str): A string with three space-separated integers representing the
                                 starting batch size, the increment, and the number of steps between
                                 each increment. For example, "192 24 8" means that the batch size
                                 starts at 192 and increases by 24 every 8 steps. Defaults to
                                 "6 2 8", which corresponds to a batch size of 2 for the first 6 steps.
        micro_bsz (int): The micro-batch size. Defaults to 2.
        seed (int): The random seed for shuffling the indices. Defaults to 0.
        drop_last (bool): If True, drop the last incomplete batch. Currently only supports True. Defaults to True.
        data_rank (int): The rank of the current process in the data parallel group. Defaults to 0.
        data_world_size (int): The number of processes in the data parallel group. Defaults to 1.
    """

    def __init__(
        self,
        datasets,
        batch_size=192,
        rampup_batch_size="6 2 8",
        micro_bsz=2,
        seed=0,
        drop_last=True,
        data_rank=0,
        data_world_size=1,
    ):
        assert drop_last is True, "Currently only support drop last"
        if rampup_batch_size:
            # In the process increase to batch_size
            start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        else:
            start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every
        if gpc.is_initialized(ParallelMode.PIPELINE):
            assert (
                batch_size - self.start_bsz
            ) % self.bsz_incre == 0, f"{batch_size} - {self.start_bsz} should be multiple of {self.bsz_incre}"
            assert batch_size % micro_bsz == 0, f"batch_size({batch_size}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.start_bsz % micro_bsz == 0
            ), f"start_bsz({self.start_bsz}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.bsz_incre % micro_bsz == 0
            ), f"bsz_incre({self.bsz_incre}) should be multiple of micro_bsz({micro_bsz})"

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_consumed_samples_in_epoch = 0
        self.datasets = datasets
        self.num_samples = sum([len(ds) for ds in datasets])

        self.get_indices()  # get data

    def get_indices(self, old_indices=None):
        if old_indices is not None:
            assert (
                len(old_indices) <= self.num_samples
            ), f"The checkpoint has {len(old_indices)} samples, \
while the new restart use less samples ({self.num_samples})"

        else:
            old_indices = np.array([])

        # indices includes len(old_indices) but not self.num_samples
        indices = np.arange(len(old_indices), self.num_samples)
        self.rng_state = self.rng.get_state()
        self.rng.shuffle(indices)
        # Need to consider drop_last
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: \
{rampup_samples*self.data_world_size} Vs. self.num_samples: {self.num_samples}"

            num_samples = (self.num_samples - rampup_samples * self.data_world_size) // (
                self.batch_size * self.data_world_size
            )
            num_samples = num_samples * self.batch_size * self.data_world_size + rampup_samples * self.data_world_size
        else:
            num_samples = self.num_samples // (self.batch_size * self.data_world_size)
            num_samples = num_samples * self.batch_size * self.data_world_size
        indices = np.concatenate([old_indices, indices]).astype(int)  # It needs to be spliced with the previous
        indices = indices[:num_samples]
        self.indices = indices
        assert len(self.indices) >= self.batch_size, "The number of samples should be larger than batch_size"
        self.num_consumed_samples_in_epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + self.epoch)

    def __len__(self):
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: {rampup_samples*self.data_world_size} \
Vs. self.num_samples: {self.num_samples}"

            num_batches = (self.num_samples - rampup_samples * self.data_world_size) // self.batch_size
            num_batches = num_batches // self.data_world_size + self.incre_every * ramp_steps
        else:
            num_batches = self.num_samples // self.batch_size // self.data_world_size

        return num_batches

    def __iter__(self):
        indices = self.indices[self.data_rank :: self.data_world_size]
        while self.num_consumed_samples_in_epoch < len(indices):
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            batch = indices[self.num_consumed_samples_in_epoch : self.num_consumed_samples_in_epoch + cur_batch_size]
            self.num_consumed_samples_in_epoch += len(batch)  # Consider multiple processes.
            self.batch_count += 1
            yield batch

        self.get_indices()  # get a new round

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "seed": self.seed,
            "data_world_size": self.data_world_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
            # the batch may be oversent, and it needs to be overwritten by the external batch_count
            "indices": self.indices,  # The sequence used to breakpoint retraining is the same as before
        }

        return states

    def load_state_dict(self, states):
        for name in ("data_world_size", "raw_rampup_batch_size", "seed"):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.rng.set_state(states["rng_state"])
        self.get_indices(old_indices=None)  # Regenerate indices based on random state
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]

    def copy(self):
        copy_sampler = StaticBatchSampler(
            self.datasets,
            self.batch_size,
            self.raw_rampup_batch_size,
            self.micro_bsz,
            self.seed,
            drop_last=True,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
        )

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler


def get_dpsampler_dataloader(
    dataset,
    shuffle=False,
    seed=1024,
    add_sampler=True,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    **kwargs,
):
    r"""Set up a deterministic dataloader (also configure seed workers, samplers and whether shuffle or not)

    Note:
        When pipeline parallel is enabled, shuffle cannot be True as it will result in mismatch between input data
        on the 1st stage and label on the last stage.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    if add_sampler and gpc.is_using_parallel_mode(ParallelMode.DATA):
        sampler = DataParallelSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker():
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    if sampler is None:
        return DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )
    else:
        return DataLoader(
            dataset,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )


class BucketGroupBatchSampler(StaticBatchSampler):
    """
    A static batch sampler that generates batches with a fixed micro-batch size.
    Supports bucket-aware sampling to ensure micro-batches use data from the same bucket range.

    Args:
        datasets: List of datasets (can be bucket-grouped datasets)
        batch_size (int): The batch size for the current rank. Defaults to 192.
        rampup_batch_size (str): Rampup configuration string.
        micro_bsz (int): The micro-batch size. Defaults to 2.
        seed (int): The random seed for shuffling the indices. Defaults to 0.
        drop_last (bool): If True, drop the last incomplete batch. Defaults to True.
        data_rank (int): The rank of the current process in the data parallel group. Defaults to 0.
        data_world_size (int): The number of processes in the data parallel group. Defaults to 1.
        enable_bucket_balance (bool): Enable bucket-aware sampling for balanced micro-batches. Defaults to True.
    """

    def __init__(
        self,
        datasets,
        batch_size=192,
        rampup_batch_size="6 2 8",
        micro_bsz=2,
        seed=0,
        drop_last=True,
        data_rank=0,
        data_world_size=1,
        enable_bucket_balance=True,
        bucket_rotation_mode="exhaustive",  # "round_robin" or "exhaustive"
    ):
        print("---------use BucketGroupBatchSampler-----------")
        assert drop_last is True, "Currently only support drop last"
        if rampup_batch_size:
            start_bsz, bsz_incre, incre_every = map(int, rampup_batch_size.split())
        else:
            start_bsz, bsz_incre, incre_every = batch_size, batch_size, 1
        self.raw_rampup_batch_size = rampup_batch_size
        self.start_bsz = start_bsz
        self.bsz_incre = bsz_incre
        self.incre_every = incre_every
        if gpc.is_initialized(ParallelMode.PIPELINE):
            assert (
                batch_size - self.start_bsz
            ) % self.bsz_incre == 0, f"{batch_size} - {self.start_bsz} should be multiple of {self.bsz_incre}"
            assert batch_size % micro_bsz == 0, f"batch_size({batch_size}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.start_bsz % micro_bsz == 0
            ), f"start_bsz({self.start_bsz}) should be multiple of micro_bsz({micro_bsz})"
            assert (
                self.bsz_incre % micro_bsz == 0
            ), f"bsz_incre({self.bsz_incre}) should be multiple of micro_bsz({micro_bsz})"

        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.batch_count = 0
        self.micro_bsz = micro_bsz
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.num_consumed_samples_in_epoch = 0
        self.datasets = datasets
        self.enable_bucket_balance = enable_bucket_balance
        self.bucket_rotation_mode = bucket_rotation_mode
        
        # Build dataset info for bucket-aware sampling
        self._build_dataset_info()
        
        self.get_indices()  # get data

    def _build_dataset_info(self):
        """Build mapping from dataset index to bucket group"""
        self.dataset_offsets = []
        self.dataset_lengths = []
        self.dataset_to_bucket = {}  # dataset_idx -> bucket_id
        
        offset = 0
        for ds_idx, ds in enumerate(self.datasets):
            ds_len = len(ds)
            self.dataset_offsets.append(offset)
            self.dataset_lengths.append(ds_len)
            
            # Extract bucket info from dataset name if available
            if hasattr(ds, 'get_dataset_name'):
                ds_name = ds.get_dataset_name()
                # Check if this is a bucket-grouped dataset (contains "-bucket")
                if "-bucket" in ds_name:
                    # Extract bucket range from name like "xxx-bucket512-1024"
                    parts = ds_name.split("-bucket")
                    if len(parts) > 1:
                        bucket_range = parts[-1]
                        self.dataset_to_bucket[ds_idx] = bucket_range
                    else:
                        self.dataset_to_bucket[ds_idx] = f"bucket_{ds_idx}"
                else:
                    self.dataset_to_bucket[ds_idx] = "default"
            else:
                self.dataset_to_bucket[ds_idx] = f"bucket_{ds_idx}"
            
            offset += ds_len
        
        self.num_samples = offset
        
        # Group datasets by bucket
        self.bucket_to_datasets = {}
        for ds_idx, bucket_id in self.dataset_to_bucket.items():
            if bucket_id not in self.bucket_to_datasets:
                self.bucket_to_datasets[bucket_id] = []
            self.bucket_to_datasets[bucket_id].append(ds_idx)
        
        if gpc.is_rank_for_log() and self.enable_bucket_balance:
            logger.info(f"BucketGroupBatchSampler: Found {len(self.bucket_to_datasets)} bucket groups")
            for bucket_id, ds_indices in self.bucket_to_datasets.items():
                total_samples = sum(self.dataset_lengths[ds_idx] for ds_idx in ds_indices)
                logger.info(f"  Bucket '{bucket_id}': {len(ds_indices)} datasets, {total_samples} samples")

    def _generate_round_robin_indices(self, start_idx):
        """
        Round-robin mode: Each step takes data from different buckets
        Automatically skip exhausted buckets
        """
        self.rng_state = self.rng.get_state()
        
        # Get sorted bucket list (ensure consistent order)
        # bucket_ids = sorted(list(self.bucket_to_datasets.keys()))
        bucket_ids = list(self.bucket_to_datasets.keys())
        
        num_buckets = len(bucket_ids)
        
        if num_buckets == 0:
            return np.array([])
        
        # Step 1: Prepare index pool for each bucket
        bucket_indices_pool = {}
        bucket_offsets = {}  # Track consumed position for each bucket
        
        for bucket_id in bucket_ids:
            ds_indices = self.bucket_to_datasets[bucket_id]
            bucket_samples = []
            
            for ds_idx in ds_indices:
                ds_start = self.dataset_offsets[ds_idx]
                ds_end = ds_start + self.dataset_lengths[ds_idx]
                # Only include new samples (not in old_indices)
                ds_samples = np.arange(max(start_idx, ds_start), ds_end)
                bucket_samples.extend(ds_samples.tolist())
            
            # Shuffle within bucket
            bucket_samples = np.array(bucket_samples)
            self.rng.shuffle(bucket_samples)
            bucket_indices_pool[bucket_id] = bucket_samples
            bucket_offsets[bucket_id] = 0  # Initialize offset
        
        # Step 2: Round-robin allocation, automatically skip exhausted buckets
        all_indices = []
        block_size = self.batch_size * self.data_world_size
        
        # Calculate total steps that can be generated
        total_samples = sum(len(samples) for samples in bucket_indices_pool.values())
        max_steps = total_samples // block_size
        bucket_idx = 0
        steps_generated = 0
        active_buckets = set(bucket_ids)  # Buckets that still have data
        
        while steps_generated < max_steps and active_buckets:
            bucket_id = bucket_ids[bucket_idx % num_buckets]
            
            # Skip exhausted buckets
            if bucket_id not in active_buckets:
                bucket_idx += 1
                continue
            
            bucket_samples = bucket_indices_pool[bucket_id]
            offset = bucket_offsets[bucket_id]
            
            # Check if this bucket still has enough data
            if offset + block_size <= len(bucket_samples):
                # Take a complete block
                all_indices.extend(bucket_samples[offset:offset + block_size].tolist())
                bucket_offsets[bucket_id] = offset + block_size
                steps_generated += 1
            elif offset < len(bucket_samples):
                # This bucket has data but not enough for a complete block
                # Mark as exhausted to keep batch_size consistent
                active_buckets.remove(bucket_id)
                remaining = len(bucket_samples) - offset
                if gpc.is_rank_for_log():
                    logger.info(f"Bucket '{bucket_id}' exhausted, {remaining} samples dropped (incomplete block)")
            else:
                # This bucket is completely exhausted
                active_buckets.remove(bucket_id)
                if gpc.is_rank_for_log():
                    logger.info(f"Bucket '{bucket_id}' exhausted")
            
            bucket_idx += 1
            
        all_indices = np.array(all_indices)
        
        if gpc.is_rank_for_log():
            logger.info(f"Round-robin mode: Generated {len(all_indices)} indices "
                       f"across {num_buckets} buckets, {steps_generated} steps")
            for bucket_id in bucket_ids:
                used = bucket_offsets[bucket_id]
                total = len(bucket_indices_pool[bucket_id])
                logger.info(f"  Bucket '{bucket_id}': used {used}/{total} samples ({used/total*100:.1f}%)")
        
        return all_indices    
    
    
    def get_indices(self, old_indices=None):
        if old_indices is not None:
            assert (
                len(old_indices) <= self.num_samples
            ), f"The checkpoint has {len(old_indices)} samples, \
while the new restart use less samples ({self.num_samples})"
        else:
            old_indices = np.array([])

        # Generate indices with bucket-aware strategy
        if self.enable_bucket_balance and len(self.bucket_to_datasets) > 1:
            if self.bucket_rotation_mode == "round_robin":
                indices = self._generate_round_robin_indices(len(old_indices))
            elif self.bucket_rotation_mode in ["U", "U0.5"]:
                indices = self._generate_U_indices(len(old_indices))
            else:
                indices = self._generate_bucket_balanced_indices(len(old_indices))
        else:
            # Original random shuffling
            indices = np.arange(len(old_indices), self.num_samples)
            self.rng_state = self.rng.get_state()
            self.rng.shuffle(indices)
        
        # Handle ramp-up and drop_last
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: \
{rampup_samples*self.data_world_size} Vs. self.num_samples: {self.num_samples}"

            num_samples = (self.num_samples - rampup_samples * self.data_world_size) // (
                self.batch_size * self.data_world_size
            )
            num_samples = num_samples * self.batch_size * self.data_world_size + rampup_samples * self.data_world_size
        else:
            num_samples = self.num_samples // (self.batch_size * self.data_world_size)
            num_samples = num_samples * self.batch_size * self.data_world_size
        
        indices = np.concatenate([old_indices, indices]).astype(int)
        indices = indices[:num_samples]
        self.indices = indices
        assert len(self.indices) >= self.batch_size, "The number of samples should be larger than batch_size"
        self.num_consumed_samples_in_epoch = 0

    def _generate_U_indices(self, start_idx):
        """
        Generate indices to change the training micro_batches into a "U" shape distribution with the bucket-aware packed samples.
        eg. Assume we have 4 buckets (A, B, C, D) from the shortest bucket to the longest bucket, we only preserve the samples in bucket A and D, 
        and we generate the indices in the following order:
        """
        self.rng_state = self.rng.get_state()
        
        micro_num = self.batch_size 
        mciro_bsz = self.micro_bsz
        
        # Step 1: Generate indices for each bucket and shuffle within bucket
        # Only preserve the shortest and longest buckets
        bucket_ids = list(self.bucket_to_datasets.keys())
        min_bucket_id, max_bucket_id = bucket_ids[0], bucket_ids[-1]
        if gpc.is_rank_for_log():
            logger.info(f"U shape bucket sampling: preserving buckets '{min_bucket_id}' and '{max_bucket_id}'")
        U_bucket_id = [min_bucket_id, max_bucket_id]
        
        bucket_indices_map = {}
        for bucket_id in U_bucket_id:
            ds_indices = self.bucket_to_datasets[bucket_id]
            bucket_samples = []
            for ds_idx in ds_indices:
                ds_start = self.dataset_offsets[ds_idx]
                ds_end = ds_start + self.dataset_lengths[ds_idx]
                # Only include new samples (not in old_indices)
                ds_samples = np.arange(max(start_idx, ds_start), ds_end)
                bucket_samples.extend(ds_samples.tolist())
            
            # Shuffle within bucket
            bucket_samples = np.array(bucket_samples)
            self.rng.shuffle(bucket_samples)
            bucket_indices_map[bucket_id] = bucket_samples
            
        # Step 2: Arrange indices in "U" shape order
        all_indices = []
        min_bucket_samples = bucket_indices_map[min_bucket_id]
        max_bucket_samples = bucket_indices_map[max_bucket_id]
        
        block_size = micro_num * self.data_world_size
        count = 0
        
        if micro_num < 2 :
            raise ValueError("U shape sampling requires at least 2 micro-batches per batch")
        if self.bucket_rotation_mode == "U0.5": # Fixed n=m=block_size/2 
                n = block_size // 2
                m = block_size - n
                total_round = min(len(min_bucket_samples) // (n), len(max_bucket_samples) // (m))
                count = total_round
                for i in range(total_round):
                    all_indices.extend(min_bucket_samples[i*n:(i+1)*n].tolist())
                    all_indices.extend(max_bucket_samples[i*m:(i+1)*m].tolist())
        elif self.bucket_rotation_mode == "U":
            while len(min_bucket_samples) >= 0 and len(max_bucket_samples) >= 0:
                # Determine n (count from A) and m (count from B)
                # Constraints: 
                # 1. n + m = micro_num
                # 2. n >= 0, m >= 0 (to ensure mix)
                # 3. n <= len(samples_A), m <= len(samples_B)
                # Take one micro-batch from shortest bucket
                block_size = micro_num * self.data_world_size
                max_n = min(block_size, len(min_bucket_samples))
                min_n = max(0, block_size - len(max_bucket_samples))
                
                if min_n > max_n:
                    break  # Cannot form a complete batch
                
                n = self.rng.randint(min_n, max_n + 1)
                m = block_size - n
                
                batch_indices = []
                if n > 0: 
                    batch_indices.extend(min_bucket_samples[:n].tolist())
                    min_bucket_samples = min_bucket_samples[n:]
                if m > 0:
                    batch_indices.extend(max_bucket_samples[:m].tolist())
                    max_bucket_samples = max_bucket_samples[m:]
    
                self.rng.shuffle(batch_indices)
                all_indices.extend(batch_indices)
                count += 1
                
        all_indices = np.array(all_indices)
        print(all_indices.shape)
        if gpc.is_rank_for_log():
            logger.info(f"Generated {len(all_indices)} U-shape bucket-balanced indices and {count} n-m pairs from {self.num_samples} total samples")
        
        return all_indices
        

    def _generate_bucket_balanced_indices(self, start_idx):
        """
        Generate indices with bucket-aware strategy.
        Ensures that consecutive micro-batches use samples from the same bucket.
        It will generate the indice of next bucket only after all micro-batches from the current bucket are used up.
        """
        self.rng_state = self.rng.get_state()
        
        # Calculate micro_num (number of micro-batches per batch)
        micro_num = self.batch_size // self.micro_bsz
        
        # Step 1: Generate indices for each bucket and shuffle within bucket
        bucket_indices_map = {}
        for bucket_id, ds_indices in self.bucket_to_datasets.items():
            bucket_samples = []
            for ds_idx in ds_indices:
                ds_start = self.dataset_offsets[ds_idx]
                ds_end = ds_start + self.dataset_lengths[ds_idx]
                # Only include new samples (not in old_indices)
                ds_samples = np.arange(max(start_idx, ds_start), ds_end)
                bucket_samples.extend(ds_samples.tolist())
            
            # Shuffle within bucket
            bucket_samples = np.array(bucket_samples)
            self.rng.shuffle(bucket_samples)
            bucket_indices_map[bucket_id] = bucket_samples
        
        # Step 2: Arrange indices by bucket blocks
        # Each block contains enough samples for multiple micro-batches from the same bucket
        all_indices = []
        bucket_ids = list(bucket_indices_map.keys())
        self.rng.shuffle(bucket_ids)  # Randomize bucket order
        
        # Process each bucket
        for bucket_id in bucket_ids:
            bucket_samples = bucket_indices_map[bucket_id]
            
            # Split into blocks of size (micro_bsz * micro_num * data_world_size)
            # This ensures each rank gets complete micro-batches from same bucket
            block_size = self.batch_size * self.data_world_size
            
            num_complete_blocks = len(bucket_samples) // block_size
            for i in range(num_complete_blocks):
                block_start = i * block_size
                block_end = (i + 1) * block_size
                all_indices.extend(bucket_samples[block_start:block_end].tolist())
            
            # Handle remaining samples
            remaining_start = num_complete_blocks * block_size
            if remaining_start < len(bucket_samples):
                remaining = bucket_samples[remaining_start:]
                # Only add if we have at least one complete batch worth of samples
                if len(remaining) >= block_size:
                    num_complete = (len(remaining) // block_size) * block_size
                    all_indices.extend(remaining[:num_complete].tolist())
        
        all_indices = np.array(all_indices)
        
        if gpc.is_rank_for_log():
            logger.info(f"Generated {len(all_indices)} bucket-balanced indices from {self.num_samples} total samples")
        
        return all_indices

    # ...existing code... (set_epoch, __len__, __iter__, state_dict, load_state_dict, copy remain the same)
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = np.random.RandomState(self.seed + self.epoch)

    def __len__(self):
        ramp_steps = (self.batch_size - self.start_bsz) // self.bsz_incre
        if self.batch_count < ramp_steps * self.incre_every:
            rampup_samples = 0
            for i in range(ramp_steps):
                rampup_samples += (i * self.bsz_incre + self.start_bsz) * self.incre_every
            assert (
                rampup_samples * self.data_world_size <= self.num_samples
            ), f"Too much rampup samples: {rampup_samples*self.data_world_size} \
Vs. self.num_samples: {self.num_samples}"

            num_batches = (self.num_samples - rampup_samples * self.data_world_size) // self.batch_size
            num_batches = num_batches // self.data_world_size + self.incre_every * ramp_steps
        else:
            num_batches = self.num_samples // self.batch_size // self.data_world_size

        return num_batches

    def __iter__(self):
        indices = self.indices[self.data_rank :: self.data_world_size]
        while self.num_consumed_samples_in_epoch < len(indices):
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)
            batch = indices[self.num_consumed_samples_in_epoch : self.num_consumed_samples_in_epoch + cur_batch_size]
            self.num_consumed_samples_in_epoch += len(batch)  # Consider multiple processes.
            self.batch_count += 1
            yield batch

        self.get_indices()  # get a new round

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "seed": self.seed,
            "data_world_size": self.data_world_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,  # The batch_count here is due to the existence of multiple processes,
            # the batch may be oversent, and it needs to be overwritten by the external batch_count
            "indices": self.indices,  # The sequence used to breakpoint retraining is the same as before
            "bucket_rotation_mode": self.bucket_rotation_mode, 
        }

        return states

    def load_state_dict(self, states):
        for name in ("data_world_size", "raw_rampup_batch_size", "seed"):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.rng.set_state(states["rng_state"])
        
        self.bucket_rotation_mode = states.get("bucket_rotation_mode", "exhaustive")
        self.get_indices(old_indices=None)  # Regenerate indices based on random state
        self.epoch = states["epoch"]
        self.batch_count = states["batch_count"]
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]

    def copy(self):
        copy_sampler = BucketGroupBatchSampler(
            self.datasets,
            self.batch_size,
            self.raw_rampup_batch_size,
            self.micro_bsz,
            self.seed,
            drop_last=True,
            data_rank=self.data_rank,
            data_world_size=self.data_world_size,
            enable_bucket_balance=True,
            bucket_rotation_mode="round_robin" if self.bucket_rotation_mode == "round_robin" else "exhaustive",
        ) 

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler


def get_dpsampler_dataloader(
    dataset,
    shuffle=False,
    seed=1024,
    add_sampler=True,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    **kwargs,
):
    r"""Set up a deterministic dataloader (also configure seed workers, samplers and whether shuffle or not)

    Note:
        When pipeline parallel is enabled, shuffle cannot be True as it will result in mismatch between input data
        on the 1st stage and label on the last stage.

    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    if add_sampler and gpc.is_using_parallel_mode(ParallelMode.DATA):
        sampler = DataParallelSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker():
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    if sampler is None:
        return DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )
    else:
        return DataLoader(
            dataset,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )