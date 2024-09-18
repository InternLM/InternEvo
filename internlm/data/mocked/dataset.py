import glob
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


def merge_tensors(file_pattern: str) -> torch.Tensor:
    files = sorted(glob.glob(file_pattern))
    return torch.cat([torch.load(file) for file in files], dim=0)


def process_raw_data(raw_data: List[torch.Tensor], micro_bsz: int) -> List[torch.Tensor]:
    return [torch.cat(raw_data[i:i+micro_bsz], dim=0) 
            for i in range(0, len(raw_data), micro_bsz)]


class MockedDataset(Dataset):
    """
    MockedDataset
    """

    def __init__(self, data_dir: str, micro_bsz: int, seq_len: int, mocked_steps: int):
        self.micro_bsz = micro_bsz
        self.seq_len = seq_len

        dp_size = gpc.get_world_size(ParallelMode.DATA)
        dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        
        db_tokens = []
        db_labels = []

        for i in range(mocked_steps):
            tokens_pattern = f"{data_dir}_tokens_step{i+1}_dp*"
            labels_pattern = f"{data_dir}_labels_step{i+1}_dp*"

            tokens = torch.chunk(merge_tensors(tokens_pattern), dp_size)[dp_rank]
            labels = torch.chunk(merge_tensors(labels_pattern), dp_size)[dp_rank]

            db_tokens.append(tokens)
            db_labels.append(labels)

        db_tokens = torch.cat(db_tokens, dim=0)
        db_labels = torch.cat(db_labels, dim=0)

        self.db_tokens = [item.tolist() for item in process_raw_data(db_tokens, micro_bsz)]
        self.db_labels = [item.tolist() for item in process_raw_data(db_labels, micro_bsz)]

        assert len(self.db_tokens) == len(self.db_labels), "Length mismatch for tokens and labels"

    def __len__(self) -> int:
        return len(self.db_tokens)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {
            "tokens": self.db_tokens[idx],
            "cu_seqlens": [i * self.seq_len for i in range(self.micro_bsz + 1)],
            "indexes": list(range(self.seq_len)) * self.micro_bsz,
            "labels": self.db_labels[idx],
            "type_ids": [0] * (self.micro_bsz * self.seq_len),
        }
