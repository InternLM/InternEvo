#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
from typing import Optional

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class StreamingStaticBatchSampler:
    """
    StreamingStaticBatchSampler is used for the training process.
    """

    def __init__(self, batch_size: int = 1, rampup_batch_size: Optional[str] = None, micro_bsz: int = 1):
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
        self.num_consumed_samples_in_epoch = 0
        self.batch_count = 0

    def __len__(self):
        return sys.maxsize

    def __iter__(self):
        while True:
            batch_rampup_idx = self.batch_count // self.incre_every
            cur_batch_size = batch_rampup_idx * self.bsz_incre + self.start_bsz
            cur_batch_size = min(cur_batch_size, self.batch_size)

            self.num_consumed_samples_in_epoch += cur_batch_size
            self.batch_count += 1
            yield [0] * cur_batch_size

    def state_dict(self):
        states = {
            "batch_size": self.batch_size,
            "raw_rampup_batch_size": self.raw_rampup_batch_size,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "batch_count": self.batch_count,
        }
        return states

    def load_state_dict(self, states):
        for name in ("raw_rampup_batch_size",):  # 'batch_size'
            assert states[name] == getattr(self, name), (name, states[name], getattr(self, name))  # should not change
        self.num_consumed_samples_in_epoch = states["num_consumed_samples_in_epoch"]
        self.batch_count = states["batch_count"]

    def copy(self):
        copy_sampler = StreamingStaticBatchSampler(self.batch_size, self.raw_rampup_batch_size)

        copy_sampler.load_state_dict(self.state_dict())
        return copy_sampler
