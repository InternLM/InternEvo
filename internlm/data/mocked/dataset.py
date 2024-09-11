import glob

import torch
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


def merge_tensors(file_pattern):
    files = sorted(glob.glob(file_pattern))
    tensors = []
    for file in files:
        tensor = torch.load(file)
        tensors.append(tensor)
    merged_tensor = torch.cat(tensors, dim=0)
    return merged_tensor


def process_raw_data(raw_data, micro_bsz):
    num_groups = len(raw_data) // micro_bsz
    result = []
    for i in range(num_groups):
        start_idx = i * micro_bsz
        end_idx = start_idx + micro_bsz
        group = raw_data[start_idx:end_idx]
        concatenated = torch.cat(group, dim=0)
        result.append(concatenated)
    return result


class MockedDataset(Dataset):
    """
    MockedDataset
    """

    def __init__(self, data_dir, micro_bsz, seq_len, mocked_steps):
        db_input_ids = []
        db_labels = []

        # load all saved data
        for i in range(mocked_steps):
            # define load pattern
            input_ids_pattern = data_dir + f"_tokens_step{i+1}_dp*"
            labels_pattern = data_dir + f"_labels_step{i+1}_dp*"
            # merge input_ids, labels, and then chunk across dp
            input_ids = torch.chunk(merge_tensors(input_ids_pattern), gpc.get_world_size(ParallelMode.DATA))[
                gpc.get_local_rank(ParallelMode.DATA)
            ]
            labels = torch.chunk(merge_tensors(labels_pattern), gpc.get_world_size(ParallelMode.DATA))[
                gpc.get_local_rank(ParallelMode.DATA)
            ]
            # load one step
            db_input_ids.append(input_ids)
            db_labels.append(labels)

        # transform db
        db_input_ids = torch.concat(db_input_ids, dim=0)
        db_labels = torch.concat(db_labels, dim=0)
        db_input_ids = [db_input_ids[i] for i in range(db_input_ids.size(0))]
        db_labels = [db_labels[i] for i in range(db_labels.size(0))]

        # gen data for internevo format
        db_input_ids = process_raw_data(db_input_ids, micro_bsz)
        db_labels = process_raw_data(db_labels, micro_bsz)
        self.db_input_ids = [item.tolist() for item in db_input_ids]
        self.db_labels = [item.tolist() for item in db_labels]

        assert len(self.db_input_ids) == len(self.db_labels)
        self.dataset_len = len(self.db_input_ids)
        self.micro_bsz = micro_bsz
        self.seq_len = seq_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        tokens = self.db_input_ids[idx]
        cu_seqlens = list(range(self.micro_bsz + 1))
        cu_seqlens = [i * self.seq_len for i in cu_seqlens]
        indexes = list(range(self.seq_len)) * self.micro_bsz
        labels = self.db_labels[idx]
        type_ids = [0] * self.micro_bsz * self.seq_len

        return {
            "tokens": tokens,
            "cu_seqlens": cu_seqlens,
            "indexes": indexes,
            "labels": labels,
            "type_ids": type_ids,
        }
