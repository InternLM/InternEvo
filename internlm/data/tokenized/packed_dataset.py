#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import itertools as it
import operator
import os
from copy import deepcopy
from enum import Enum, unique
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.data.tokenized.single_dataset import JsonlDataset
from internlm.data.utils import get_dataset_type_id, get_dataset_type_ids_map
from internlm.utils.logger import get_logger

DEFAULT_SEED = 1024
logger = get_logger(__file__)


def get_ltor_masks_and_position_ids(data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


@unique
class PackedDatasetType(Enum):
    TorchPack = 1
    TorchUnPacked = 2
    TorchAttnMask = 3
    MindSpore = 4


packed_dataset_type = {
    "TorchPack": PackedDatasetType.TorchPack,
    "TorchUnPacked": PackedDatasetType.TorchUnPacked,
    "TorchAttnMask": PackedDatasetType.TorchAttnMask,
    "MindSpore": PackedDatasetType.MindSpore,
}


class DatasetWithPad(torch.utils.data.Dataset):
    """
    The class DatasetWithPad takes in a dataset and aggregates samples of different
    lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        max_length_per_sample: The maximum length of each original sample. Default is 2048.
        packed_length: The length of each packed sample. Default is 4096.
    """

    def __init__(
        self,
        dataset,
        max_length_per_sample: int = 2048,
        packed_length: int = 4096,
        use_flash_style_data_format: bool = True,
    ):
        assert hasattr(dataset, "lengths")
        assert len(getattr(dataset, "lengths")) == len(
            dataset
        ), "The dataset must have lengths attribute and have the same length as the dataset"
        self.dataset = dataset
        self.max_length_per_sample = max_length_per_sample
        self.micro_bsz = packed_length // max_length_per_sample
        self.lengths = getattr(self.dataset, "lengths")
        self.packed_length = packed_length
        # Force a seed to be fixed to prevent problems caused by the seed not being restored when restarting
        self.seed = DEFAULT_SEED
        self.pad_token = 2
        self.eos_token = 2
        self.sample_indices, self.len_samples_shuffled, self.acm_len_samples = self.accu_sample_len(seed=self.seed)
        self.num_tokens = sum(self.lengths)
        self.use_flash_style_data_format = use_flash_style_data_format
        assert self.pad_token == self.eos_token

        if "DEBUG_DATA_SHAPE" in os.environ:
            print(
                f"packed_length: {self.packed_length}, micro_bsz: {self.micro_bsz}, max_length_per_sample: \
{self.max_length_per_sample}, pad_token: {self.pad_token}, eos_token: {self.eos_token}",
                flush=True,
            )

    def get_dataset_name(self):
        return self.dataset.get_dataset_name()

    def accu_sample_len(self, seed=None):
        """accumulative length of samples"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(self.seed - 1)

        sample_indices = np.arange(len(self.lengths))
        rng.shuffle(sample_indices)
        len_samples_shuffled = list(map(self.lengths.__getitem__, sample_indices))
        acm_len_samples = list(it.accumulate(len_samples_shuffled, operator.add))
        return sample_indices, len_samples_shuffled, acm_len_samples

    def __len__(self):
        # Line 405 of document_to_sequence.py in metaseq is directly spliced,
        # without additional consideration of sos or eos
        n_packs = self.num_tokens // self.packed_length
        return n_packs

    def build_unpack(self, index):
        sample_indexes = list(range(index * self.micro_bsz, (index + 1) * self.micro_bsz, 1))
        pack, cu_seqlens, indexes, labels, type_ids = [], [0], [], [], []

        def append_add(x, data):
            x.append(data)

        def extend_add(x, data):
            x.extend(data)

        make_batch_fn = append_add if not self.use_flash_style_data_format else extend_add

        for idx in sample_indexes:
            if idx < len(self.dataset):
                sample = self.dataset[self.sample_indices[idx]]
                length = min(len(sample["tokens"]), self.max_length_per_sample)
                chunk = sample["tokens"][0:length]
                token_length = len(chunk)
                padding_length = self.max_length_per_sample - token_length
                if chunk[length - 1] == self.eos_token:
                    chunk[length - 1] = gpc.config.model.vocab_size + 1

                chunk = np.pad(chunk, (0, padding_length), "constant", constant_values=(self.eos_token, self.eos_token))

                label = np.array(list(chunk[1:]) + [self.eos_token])
                label[np.array(chunk[:] == self.eos_token)] = -100

                make_batch_fn(labels, label)
                make_batch_fn(pack, chunk)
                make_batch_fn(type_ids, [sample.get("type_id", 0)] * self.max_length_per_sample)
                cu_seqlens.extend([cu_seqlens[-1] + token_length])
                indexes.extend(list(range(token_length)) + [-1] * padding_length)
            else:
                # If the dataset length is not enough to fetch next sample, just drop last.
                make_batch_fn(labels, [-100] * self.max_length_per_sample)
                make_batch_fn(chunk, [self.eos_token] * self.max_length_per_sample)
                make_batch_fn(type_ids, [0] * self.max_length_per_sample)
                cu_seqlens.extend([cu_seqlens[-1] + self.max_length_per_sample])
                indexes.extend([-1] * self.max_length_per_sample)

        return {"tokens": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels, "type_ids": type_ids}

    def __getitem__(self, item: int) -> Dict:
        """Given the index, it returns a dict as
        {
         'tokens': List[int],
         'cu_seqlens': List[int],
         'labels': List[int], # corresponds to 'tokens' and shifted yet, -100 means skipping prediction
        }
        """
        return self.build_unpack(item)


class PackedDataset(torch.utils.data.Dataset):
    """
    The class PackedDataset takes in a dataset and aggregates samples of different
    lengths together based on the packed_length.

    Args:
        dataset: The original dataset to pack.
        max_length_per_sample: The maximum length of each original sample. Default is 2048.
        packed_length: The length of each packed sample. Default is 4096.
    """

    def __init__(
        self,
        dataset,
        max_length_per_sample: int = 2048,
        packed_length: int = 4096,
    ):
        assert hasattr(dataset, "lengths")
        assert len(getattr(dataset, "lengths")) == len(
            dataset
        ), "The dataset must have lengths attribute and have the same length as the dataset"
        self.dataset = dataset
        self.max_length_per_sample = max_length_per_sample
        self.lengths = getattr(self.dataset, "lengths")
        self.packed_length = packed_length
        # Force a seed to be fixed to prevent problems caused by the seed not being restored when restarting

        self.seed = DEFAULT_SEED

    def __getitem__(self, item: int) -> Dict:
        """Given the index, it returns a dict as
        {
         'tokens': List[int],
         'cu_seqlens': List[int],
         'indexes': List[int], # denotes positional vector as 'tokens'
         'labels': List[int], # corresponds to 'tokens' and shifted yet, -100 means skipping prediction
        }
        """
        return self.build_pack(item)


class DatasetPackIntoOne(torch.utils.data.Dataset):
    """
    A dataset wrapper that aggregates samples with different lengths based on packed_length.
    If a sample is shorter than max_length_per_sample, it will be merged with other samples.
    For example, given a dataset with 10 samples:
    [1, 2, 3, 4, 5]
    [6, 7]
    [8, 9, 10, 11]
    [12, ..., 100]
    ...

    Args:
        dataset: The original dataset to be wrapped.
        max_length_per_sample (int): The maximum length allowed for each sample.
        packed_length (int): The desired length for each packed sample.
        use_flash_style_data_format (bool): If False, will add a dimension for 'micro_bsz',
            like [micor_bsz, seqlen] else [seqlen], and will not return 'indexes'
            Although 'use_flash_style_data_format' is False, We still need return 'cu_seqlens', because we need to
            konw the true seq lengths when cal metrics, but we will pop 'cu_seqlens' before batch enter
            internlm runtime engine.
    """

    def __init__(
        self,
        dataset,
        max_length_per_sample: int = 2048,
        packed_length: int = 4096,
        debug=False,
        use_flash_style_data_format=True,
    ):
        assert packed_length % max_length_per_sample == 0
        assert hasattr(dataset, "lengths")
        assert len(getattr(dataset, "lengths")) == len(
            dataset
        ), "The dataset must have lengths attribute and have the same length as the dataset"
        self.dataset = dataset
        self.max_length_per_sample = max_length_per_sample
        self.lengths = getattr(self.dataset, "lengths")
        self.bsz = packed_length // max_length_per_sample
        self.packed_length = packed_length
        self.debug = debug
        # Force a seed to be fixed to prevent problems caused by the seed not being restored when restarting

        self.seed = DEFAULT_SEED
        indices = np.arange(len(self.lengths))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(indices)
        self.indices = indices
        self.cum_lens = np.cumsum(self.lengths[self.indices])
        self.num_tokens = sum(self.lengths)
        self.use_flash_style_data_format = use_flash_style_data_format

    def get_dataset_name(self):
        return self.dataset.get_dataset_name()

    def __len__(self):
        n_packs = self.num_tokens // self.packed_length
        return n_packs

    def find_offset(self, offset):
        idx = np.searchsorted(self.cum_lens, offset, side="right")
        if idx == 0:
            return idx, offset
        length = offset - self.cum_lens[idx - 1]
        return idx, length

    def pdebug(self, line):
        if self.debug:
            print(line, flush=True)

    def __getitem__(self, item: int) -> Dict:
        """Given the index, it returns a dict as
        {
         'tokens': List[int],
         'cu_seqlens': List[int],
         'indexes': List[int], # denotes positional vector as 'tokens'
         'labels': List[int], # corresponds to 'tokens' and shifted yet, -100 means skipping prediction
        }
        """

        start_idx, start_length = self.find_offset(item * self.packed_length)
        end_idx, end_length = self.find_offset((item + 1) * self.packed_length)
        pack_tokens = []
        pack_labels = []
        type_ids = []

        cu_seqlens = [i * self.max_length_per_sample for i in range(self.bsz + 1)]
        self.pdebug(f"item : {item}, start_idx:{start_idx}, start_length:{start_length} ")
        self.pdebug(f"item : {item}, end_idx:{end_idx}, end_length:{end_length} ")

        if start_idx == end_idx:
            idx = self.indices[start_idx]
            sample = self.dataset[idx]
            self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
            tokens = sample["tokens"][start_length:end_length]
            pack_tokens.extend(tokens)
            pack_labels.extend(tokens[1:] + [-100])
            type_ids.extend([sample["type_id"]] * len(tokens))
            if self.use_flash_style_data_format is False:
                return {
                    "tokens": [pack_tokens],
                    "labels": [pack_labels],
                    "type_ids": [type_ids],
                    "cu_seqlens": cu_seqlens,
                }
            else:
                return {
                    "tokens": pack_tokens,
                    "cu_seqlens": cu_seqlens,
                    "indexes": list(range(self.max_length_per_sample)) * self.bsz,
                    "labels": pack_labels,
                    "type_ids": type_ids,
                }

        idx = self.indices[start_idx]
        sample = self.dataset[idx]
        self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
        tokens = sample["tokens"][start_length:]
        pack_tokens.extend(tokens)
        pack_labels.extend(tokens[1:] + [-100])
        type_ids.extend([sample["type_id"]] * len(tokens))

        for i in range(start_idx + 1, end_idx):
            idx = self.indices[i]
            sample = self.dataset[idx]
            self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
            tokens = sample["tokens"]
            pack_tokens.extend(tokens)
            pack_labels.extend(tokens[1:] + [-100])
            type_ids.extend([sample.get("type_id")] * len(tokens))

        # corner case, the last sample is useless
        if end_length == 0:
            pass
        else:
            idx = self.indices[end_idx]
            sample = self.dataset[idx]
            self.pdebug(f"item : {item}, idx: {idx}, len : {len(sample['tokens'])}")
            tokens = sample["tokens"][:end_length]
            pack_tokens.extend(tokens)
            pack_labels.extend(tokens[1:] + [-100])
            type_ids.extend([sample.get("type_id")] * len(tokens))

        if self.use_flash_style_data_format is False:
            return {
                "tokens": [pack_tokens],
                "cu_seqlens": cu_seqlens,
                "labels": [pack_labels],
                "type_ids": [type_ids],
            }
        else:
            return {
                "tokens": pack_tokens,
                "cu_seqlens": cu_seqlens,
                "indexes": list(range(self.max_length_per_sample)) * self.bsz,
                "labels": pack_labels,
                "type_ids": type_ids,
            }


class DatasetWithCut(PackedDataset):
    """
    The class PackedDataset takes in a dataset and aggregates samples of different
    lengths together based on the packed_length using cut mode.


    max_length_per_sample = 3
    packed_length = 5
    [1, 2]
    [3, 4]
    [5, 6, 7]
    [8, 9, 10, 11, 12, 13]

    --->
    [1, 2, 3, 4, 5]
    [6, 7, 8, 9, 10]
    [11, 12, 13, 0, 0]

    Args:
        dataset: The original dataset to pack.
        max_length_per_sample: The maximum length of each original sample. Default is 2048.
        packed_length: The length of each packed sample. Default is 4096.
    """

    def __init__(
        self,
        dataset,
        max_length_per_sample: int = 2048,
        packed_length: int = 4096,
    ):
        super().__init__(dataset, max_length_per_sample, packed_length)
        self.sample_indices, self.len_samples_shuffled, self.acm_len_samples = self.accu_sample_len(seed=self.seed)
        self.num_tokens = sum(self.lengths)

    def get_dataset_name(self):
        return self.dataset.get_dataset_name()

    def accu_sample_len(self, seed=None):
        """accumulative length of samples"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(self.seed - 1)

        sample_indices = np.arange(len(self.lengths))
        rng.shuffle(sample_indices)
        len_samples_shuffled = list(map(self.lengths.__getitem__, sample_indices))
        acm_len_samples = list(it.accumulate(len_samples_shuffled, operator.add))
        return sample_indices, len_samples_shuffled, acm_len_samples

    def __len__(self):
        # Line 405 of document_to_sequence.py in metaseq is directly spliced,
        # without additional consideration of sos or eos
        n_packs = self.num_tokens // self.packed_length
        return n_packs

    def cal_map(self, carriage_idx: int = 0):
        assert carriage_idx >= 0
        length_train = (carriage_idx + 1) * self.packed_length
        post_pos = np.searchsorted(self.acm_len_samples, length_train, side="left")
        return post_pos

    def mapping(self, pack_idx: int = 0):
        # pack_idx is zero-based
        pre_pos, pre_token_id = 0, 0
        if pack_idx > 0:
            pre_pos = self.cal_map(pack_idx - 1)
            pre_token_id = self.len_samples_shuffled[pre_pos] - (
                self.acm_len_samples[pre_pos] - (pack_idx) * self.packed_length
            )
            if pre_token_id == self.len_samples_shuffled[pre_pos]:
                pre_pos += 1
                pre_token_id = 0

        pos = self.cal_map(pack_idx)
        token_id = self.len_samples_shuffled[pos] - (self.acm_len_samples[pos] - (pack_idx + 1) * self.packed_length)
        return pre_pos, pre_token_id, pos, token_id

    def build_pack(self, item):
        pre_pos, pre_token_id, pos, token_id = self.mapping(item)
        pack, cu_seqlens, indexes, labels, type_ids = [], [0], [], [], []

        while pre_pos < pos:
            sample_idx = self.sample_indices[pre_pos]
            sample = self.dataset[sample_idx]
            chunk = sample["tokens"][pre_token_id:]
            pack.extend(chunk)
            _labels = deepcopy(chunk)
            _labels = list(_labels[1:]) + [-100]
            assert len(_labels) == len(chunk), (_labels, chunk)
            labels.extend(_labels)
            type_ids.extend([sample.get("type_id", 0)] * len(chunk))
            num_new_samples, tokens_left = divmod(len(chunk), self.max_length_per_sample)
            for _ in range(num_new_samples):
                cu_seqlens.append(cu_seqlens[-1] + self.max_length_per_sample)
                indexes.extend(list(range(self.max_length_per_sample)))
            if tokens_left > 0:
                cu_seqlens.append(cu_seqlens[-1] + tokens_left)
                indexes.extend(list(range(tokens_left)))
            pre_pos = pre_pos + 1
            pre_token_id = 0

        sample_idx = self.sample_indices[pos]
        sample = self.dataset[sample_idx]
        chunk = sample["tokens"][pre_token_id:token_id]  # fragement of a sample
        pack.extend(chunk)
        _labels = deepcopy(chunk)
        if token_id == len(sample["tokens"]):
            _labels = list(_labels[1:]) + [-100]
        else:
            if token_id > len(sample["tokens"]):
                print(f"token_id {token_id}, len of sample {len(sample['tokens'])}")
            _labels = list(_labels[1:]) + [sample["tokens"][token_id]]
        assert len(_labels) == len(chunk), (_labels, chunk)
        labels.extend(_labels)
        type_ids.extend([sample.get("type_id", 0)] * len(chunk))
        num_new_samples, tokens_left = divmod(len(chunk), self.max_length_per_sample)
        for _ in range(num_new_samples):
            cu_seqlens.append(cu_seqlens[-1] + self.max_length_per_sample)
            indexes.extend(list(range(self.max_length_per_sample)))
        if tokens_left > 0:
            cu_seqlens.append(cu_seqlens[-1] + tokens_left)
            indexes.extend(list(range(tokens_left)))

        out = {"tokens": pack, "cu_seqlens": cu_seqlens, "indexes": indexes, "labels": labels, "type_ids": type_ids}
        return out


def get_packed_dataset_without_short_length(
    folder,
    max_length_per_sample=2048,
    packed_length=4096,
    show_progress=False,
    min_length=50,
    min_length_dict=None,
    pack_sample_into_one=False,
    use_flash_style_data_format=True,
    break_mode="cut",
):
    """
    Given a folder, combine all the .bin files into a single large dataset.
    And filter out short samples with length less than 'min_length'.

    Each .bin file is treated as a separate dataset.

    Args:
        folder (str): Path to the folder containing the .bin files.
        max_length_per_sample (int): Maximum length of each sample.
        packed_length (int): Length to pack samples to.
        show_progress (bool): Whether to show the progress bar.
        min_length (int): The minimum length of the sample.
        min_length_dict (dict): The minimum length of the sample for each dataset.
         The format is something like {'pile-arxiv': 50}
        dataset_backend (Optional[str]): Dataset storage location. Optional parameters are local, local-shm, kv

    Returns:
        A packed dataset containing all the data from the .bin files.
    """

    assert os.path.exists(folder), f"{folder} does not exist."
    print(f"load_data_folder: {folder}", flush=True)
    datasets = []
    delete_samples = 0

    DATASET_TYPE_IDS_MAP = get_dataset_type_ids_map(folder)

    if gpc.get_global_rank() == 0:
        triples = [list(os.walk(folder, followlinks=True))]
    else:
        triples = [None]
    dist.broadcast_object_list(triples, src=0)
    triples = triples[0]

    for root, dirs, files in triples:
        dirs.sort()  # Let the folder need to be returned in a fixed order
        if gpc.is_rank_for_log():
            logger.info(f"Reading {root}...")
        num_token_in_folder = 0

        for fn in tqdm(sorted(files), total=len(files), leave=False, disable=not show_progress):
            if fn.endswith(".bin"):
                fp = os.path.join(root, fn)
                catch_ml_keys = []
                min_length_num = min_length
                if min_length_dict is not None:
                    for k, v in min_length_dict.items():
                        if k in fp:
                            min_length_num = v
                            catch_ml_keys.append(k)
                    assert (
                        len(catch_ml_keys) < 2
                    ), f"The file name `{fp}` matched the following resample keys:{catch_ml_keys}"

                ds_type_id = get_dataset_type_id(DATASET_TYPE_IDS_MAP, path=fp)
                ds = JsonlDataset(fp, ds_type_id, min_length=min_length_num)

                if hasattr(ds, "old_length"):
                    delete_samples += ds.old_length - len(ds)
                if len(ds) == 0:
                    if gpc.is_rank_for_log():
                        logger.info(f"None of the data in `{fp}` is longer than {min_length}")
                    continue

                if pack_sample_into_one:
                    assert break_mode == "cut", "pack_sample_into_one only support cut mode"
                    ds = DatasetPackIntoOne(
                        ds,
                        max_length_per_sample,
                        packed_length,
                        use_flash_style_data_format=use_flash_style_data_format,
                    )
                else:
                    if break_mode == "cut":
                        assert (
                            use_flash_style_data_format is True
                        ), "If pack_sample_into_one is False and using cut mode, must set use_flash_style_data_format is True"
                        ds = DatasetWithCut(ds, max_length_per_sample, packed_length)
                    elif break_mode == "pad":
                        ds = DatasetWithPad(
                            ds,
                            max_length_per_sample,
                            packed_length,
                            use_flash_style_data_format=use_flash_style_data_format,
                        )
                    else:
                        raise ValueError(f"Unknow break_mode: {break_mode}")

                num_token_in_folder += len(ds) * packed_length
                datasets.append(ds)

    dataset = ConcatDataset(datasets=datasets)
    if gpc.is_rank_for_log():
        logger.info(
            f"Find `{len(datasets)}` datasets, \
            {len(dataset)} samples, \
            delete `{delete_samples}` because of short length",
        )

    return dataset


def test_npu_fa_packed_sample_into_one_batch(batch):
    """This function is used to align torch_npu_flash_attention with noFA implemention.

    Args:
        batch (tuple): the output of load_new_batch()
    """
    if gpc.config.model.use_flash_attn_npu is True:
        token_batch = batch[0]
        assert gpc.config.data.micro_num == len(
            token_batch["input_ids"]
        ), f"{gpc.config.data.micro_num} = {len(token_batch['input_ids'])}"
        from internlm.data.collaters import get_ltor_masks_and_position_ids

        attention_mask_list = []
        for micro_num_idx in range(len(token_batch["input_ids"])):
            this_micro_batch = token_batch["input_ids"][micro_num_idx]

            assert len(this_micro_batch.size()) == 1
            assert gpc.config.data.micro_bsz == 1
            this_micro_batch = torch.unsqueeze(this_micro_batch, 0)
            attention_mask, _, _ = get_ltor_masks_and_position_ids(
                data=this_micro_batch,
                eod_token=0,
                reset_position_ids=True,
                reset_attention_mask=True,
                eod_mask_loss=True,
            )
            attention_mask_list.append(attention_mask)

        token_batch["input_ids"] = torch.unsqueeze(token_batch["input_ids"], 1)  # -> [micro_num, micro_bsz, seqlen]
        token_batch["attention_mask"] = attention_mask_list
        token_batch.pop("indexes")

        batch = (token_batch, batch[1])

    return batch
