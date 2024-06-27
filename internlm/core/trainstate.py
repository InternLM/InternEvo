#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import json
import os
from collections import deque


class TrainState:
    """
    The TrainState class is used to record the current state of training.

    Args:
        train_dl (DataLoader): The DataLoader object used for training.
    """

    def __init__(self, config, batch_sampler) -> None:
        """
        Args:
            config (Config): internlm config
            batch_sampler (torch.utils.data.Sampler): Because the dataloader loading is
            asynchronous and prefetched, the batch_sampler state maintained inside the
            dataloader are faster then the actual training progress, so we copy the
            batch_sampler as the anchor point of ckpt reload.
        """
        # The number of batches produced by the data iterator
        self.batch_count: int = 0
        # Used to store the number of samples consumed in the current epoch
        self.num_consumed_samples_in_epoch: int = 0
        # Total number of tokens consumed
        self.num_consumed_tokens: int = 0
        # Number of batches skipped due to inf or nan values
        self.inf_nan_skip_batches: int = 0
        # Records the number of updates, skipped batches and inf batches are not counted
        self.step_count: int = 0

        # Total step count
        self.total_steps: int = config.data.total_steps

        # resume tensorboard folder, need load from checkpoint or set manually.
        self.resume_tb_folder = config.resume_tb_folder

        self.tensorboard_folder = config.tensorboard_folder

        # learning rate
        self.lr = config.adam.lr

        # smapler state
        if batch_sampler:
            self.init_batch_sampler(batch_sampler)

        # tgs statistic
        self.tgs_statistic = {
            "sum_step": 0,
            "sum_tg": 0,
            "total_time": 0,
            "sum_last_tg_10": 0,
            "sum_last_time_10": 0,
            "sum_last_tg_50": 0,
            "sum_last_time_50": 0,
            "SMA_tg_50": 0,
            "SMA_time_50": 0,
            "SMA_tg_50_list": deque(),
            "SMA_time_50_list": deque(),
            "sum_tgs": 0,
            "last_tgs_10": 0,
            "last_tgs_50": 0,
        }

    def init_batch_sampler(self, batch_sampler):
        """
        Args:
            batch_sampler (torch.utils.data.Sampler): sampler.
        """
        # make a copy of batch_sampler.
        self.batch_sampler = batch_sampler.copy()
        # Iterator for the batch sampler
        self.batch_sampler_iter = iter(self.batch_sampler)

    def __str__(self) -> str:
        """Returns a string representation of the training state in JSON format."""
        info = {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
        }

        return json.dumps(info, indent=4, sort_keys=True)

    def load_state_dict(self, other_stuffs):
        """
        Resumes training from a checkpoint.

        Args:
            other_stuffs (dict): Other information needed to resume training.
        """
        self.num_consumed_samples_in_epoch = other_stuffs["num_consumed_samples_in_epoch"]
        self.num_consumed_tokens = other_stuffs["num_consumed_tokens"]
        self.inf_nan_skip_batches = other_stuffs["inf_nan_skip_batches"]

        # Because the ckpt save occurs after updating 'step_count',
        # there is no need to increment 'step_count' here (Does our step count start from 0 ?),
        # However, 'batch_count' is updating before ckpt storage, so it need to inc 1 when resume.
        self.batch_count = other_stuffs["batch_count"] + 1  # here you need to shift a batch backward
        self.step_count = other_stuffs.get("step_count", self.batch_count)

        # resume tensorboard from older tensorboard_folder
        self.resume_tb_folder = other_stuffs.get("tensorboard_folder", None)

    def state_dict(self):
        if os.environ.get("CLUSTER_NAME") == "volc" and os.environ.get("petrelfs_tb_path") is not None:
            tensorboard_folder = os.path.join(os.environ["petrelfs_tb_path"], os.environ["MLP_TASK_ID"])
        else:
            tensorboard_folder = self.tensorboard_folder
        return {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
            "tensorboard_folder": tensorboard_folder,
        }
