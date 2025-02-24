import glob
import os
import re
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc


def merge_tensors(fn_pattern: str) -> torch.Tensor:
    """
    Merge per-step saved tensors into one, across all dp ranks.

    Args:
        fn_pattern: glob pattern for saved tensors, such like tokens_step{}_dp* or labels_step{}_dp*

    Returns:
        merged tensor

    """
    return torch.cat([torch.load(fn) for fn in sorted(glob.glob(fn_pattern))], dim=0)


def split_tensors(raw_data: List[torch.Tensor], micro_bsz: int) -> List[torch.Tensor]:
    """
    Split saved tensors into list of tensors where each element is micro batch with shape (micro_bsz, seq_len)

    Args:
        raw_data: list of tensors, where length is mocked_steps * micro_num * micro_bsz,
                    and each element is a tensor with shape (seq_len)
        micro_bsz: micro batch size

    Returns:
        list of tensors, where length is mocked_steps * micro_num,
                    and each element is micro batch with shape (micro_bsz, seq_len)

    """
    return [torch.cat(raw_data[i : i + micro_bsz], dim=0) for i in range(0, len(raw_data), micro_bsz)]


def get_mocked_steps(data_dir: str) -> int:
    step_pattern = r"_step(\d+)_dp"
    mocked_steps = 0

    for fn in os.listdir(data_dir):
        step_match = re.search(step_pattern, fn)
        if step_match:
            step = int(step_match.group(1))
            mocked_steps = max(mocked_steps, step)

    return mocked_steps


class MockedDataset(Dataset):
    """
    Mocked dataset for easier precision alignment.

    Suppose the saved data is with below format:
    tokens_step{}_dp{}.pt, where {} is the saved step number (start from 0) and the dp rank (start from 0).
    labels_step{}_dp{}.pt, where {} is the saved step number (start from 0) and the dp rank (start from 0).

    Each of the saved data is a micro_num accumucalted tensor, where micro batch is (micro_bsz, seq_len).
    Hence, the shape of tokens_step{}_dp{}.pt and labels_step{}_dp{}.pt is (micro_num * micro_bsz, seq_len).

    """

    def __init__(self, train_folder: str, micro_bsz: int, micro_num: int, seq_len: int):

        self.micro_bsz = micro_bsz
        self.micro_num = micro_num
        self.seq_len = seq_len

        dp_size = gpc.get_world_size(ParallelMode.DATA)
        dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        mocked_steps = get_mocked_steps(train_folder)

        tokens_list = []
        labels_list = []
        for i in range(mocked_steps):
            # define fn pattern
            tokens_fn_pattern = f"{train_folder}/tokens_step{i}_dp*"
            labels_fn_pattern = f"{train_folder}/labels_step{i}_dp*"

            # merge per-step mocked data and chunk across dp ranks
            tokens = torch.chunk(merge_tensors(tokens_fn_pattern), dp_size)[dp_rank]  # (micro_num * micro_bsz, seq_len)
            labels = torch.chunk(merge_tensors(labels_fn_pattern), dp_size)[dp_rank]  # (micro_num * micro_bsz, seq_len)

            # check and append
            assert tokens.size() == labels.size(), "Mismatch for tokens and labels"
            assert tokens.size(1) == seq_len, "Mismatch for seq_len"
            assert tokens.size(0) == micro_bsz * micro_num, "Mismatch for global_bsz"
            tokens_list.append(tokens)
            labels_list.append(labels)

        # concatenate across mocked_steps
        db_tokens = torch.cat(tokens_list, dim=0)  # (mocked_steps * micro_num * micro_bsz, seq_len)
        db_labels = torch.cat(labels_list, dim=0)  # (mocked_steps * micro_num * micro_bsz, seq_len)

        # split into (mocked_steps * micro_num, packed_length), where packed_length = micro_bsz, seq_len
        self.db_tokens = [
            item.tolist() for item in split_tensors([db_tokens[i] for i in range(db_tokens.size(0))], micro_bsz)
        ]
        self.db_labels = [
            item.tolist() for item in split_tensors([db_labels[i] for i in range(db_labels.size(0))], micro_bsz)
        ]

        # simple sanity check: ensure loaded per-step data is equivalent to saved per-step data
        self._sanity_check(tokens_list, labels_list)

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

    def _sanity_check(self, tokens_list: List[torch.Tensor], labels_list: List[torch.Tensor]):
        tokens_list_tocheck = []
        for i in range(len(self.db_tokens)):
            tokens_list_tocheck += self.db_tokens[i]
            if (i + 1) % self.micro_num == 0:
                tokens_list_ref = tokens_list[i // self.micro_num].flatten(0, 1).tolist()
                assert tokens_list_tocheck == tokens_list_ref, "loaded tokens not equivalent to saved tokens"
                tokens_list_tocheck = []

        labels_list_tocheck = []
        for i in range(len(self.db_labels)):
            labels_list_tocheck += self.db_labels[i]
            if (i + 1) % self.micro_num == 0:
                labels_list_ref = labels_list[i // self.micro_num].flatten(0, 1).tolist()
                assert labels_list_tocheck == labels_list_ref, "loaded labels not equivalent to saved labels"
                labels_list_tocheck = []
