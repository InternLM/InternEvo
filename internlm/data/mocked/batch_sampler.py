import copy


class MockedSequentialBatchSampler:
    """
    MockedSequentialBatchSampler
    """

    def __init__(self, data_source, micro_num):
        self.data_source = data_source
        self.micro_num = micro_num

    def __iter__(self):
        num_samples = len(self.data_source)
        for start in range(0, num_samples, self.micro_num):
            end = min(start + self.micro_num, num_samples)
            yield list(range(start, end))

    def __len__(self):
        return (len(self.data_source) + self.micro_num - 1) // self.micro_num

    # TODO: implement copy method that compatible with InternEvo trainstate
    def copy(self):
        return copy.deepcopy(self)
