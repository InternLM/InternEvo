import copy
import math

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


class MegatronBatchSampler:
    """
    MegatronBatchSampler
    """

    def __init__(self, total_samples, consumed_samples, batch_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        self.dp_size = gpc.get_world_size(ParallelMode.DATA)

        # Sanity checks.
        assert self.total_samples > 0, "no sample to consume: {}".format(self.total_samples)
        assert self.consumed_samples < self.total_samples, "no samples left to consume: {}, {}".format(
            self.consumed_samples, self.total_samples
        )
        assert self.batch_size > 0
        assert self.dp_size > 0
        assert self.dp_rank < self.dp_size, "dp_rank should be smaller than dp_size: {}, " "{}".format(
            self.dp_rank, self.dp_size
        )

    def __len__(self):
        if self.drop_last and self.total_samples % self.dp_size != 0:
            return math.ceil(self.total_samples - self.dp_size) / self.dp_size
        else:
            return math.ceil(self.total_samples / self.dp_size)

    def get_start_end_idx(self):
        start_idx = self.dp_rank * self.batch_size
        end_idx = start_idx + self.batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.batch_size * self.dp_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]

    # TODO: implement copy method that compatible with InternEvo trainstate
    def copy(self):
        return copy.deepcopy(self)
