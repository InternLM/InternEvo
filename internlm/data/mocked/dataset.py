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
        self.micro_bsz = micro_bsz
        self.seq_len = seq_len

        db_tokens = []
        db_labels = []

        dp_size = gpc.get_world_size(ParallelMode.DATA)
        dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        
        for i in range(mocked_steps):
            tokens_pattern = f"{data_dir}_tokens_step{i+1}_dp*"
            labels_pattern = f"{data_dir}_labels_step{i+1}_dp*"

            # Merge and chunk
            tokens = torch.chunk(merge_tensors(tokens_pattern), dp_size)[dp_rank]
            labels = torch.chunk(merge_tensors(labels_pattern), dp_size)[dp_rank]

            db_tokens.append(tokens)
            db_labels.append(labels)

        # Concatenate all tensors at once
        db_tokens = torch.cat(db_tokens, dim=0)
        db_labels = torch.cat(db_labels, dim=0)

        # Convert to list in a more efficient way
        db_tokens = list(db_tokens)
        db_labels = list(db_labels)

        # Process data
        self.db_tokens = [item.tolist() for item in process_raw_data(db_tokens, micro_bsz)]
        self.db_labels = [item.tolist() for item in process_raw_data(db_labels, micro_bsz)]

        self.dataset_len = len(self.db_tokens)
        assert len(self.db_tokens) == len(self.db_labels), "length mismatch for tokens and labels"

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return {
            "tokens": self.db_tokens[idx],
            "cu_seqlens": [i * self.seq_len for i in range(self.micro_bsz + 1)],
            "indexes": list(range(self.seq_len)) * self.micro_bsz,
            "labels": self.db_labels[idx],
            "type_ids": [0] * (self.micro_bsz * self.seq_len),
        }
