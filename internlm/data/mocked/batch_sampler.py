import copy


class MockedSequentialBatchSampler:
    """
    MockedSequentialBatchSampler
    """

    def __init__(self, train_ds, micro_num):
        self.train_ds = train_ds
        self.micro_num = micro_num

    def __iter__(self):
        num_samples = len(self.train_ds)
        for start in range(0, num_samples, self.micro_num):
            end = min(start + self.micro_num, num_samples)
            yield list(range(start, end))

    def __len__(self):
        return (len(self.train_ds) + self.micro_num - 1) // self.micro_num

    # TODO: implement copy method that compatible with InternEvo trainstate
    def copy(self):
        return copy.deepcopy(self)
