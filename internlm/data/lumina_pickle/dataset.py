import copy
import json
import os
import pickle
import traceback
import yaml

import torch
from torch.utils.data import Dataset

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from internlm.utils.logger import get_logger

logger = get_logger(__file__)

class LuminaPickleDataset(Dataset):
    def __init__(self, data_yaml: str, micro_batch_size: int, seq_len: int, base_path: str = None):
        logger.info(f"read data meta yaml from {data_yaml}, {base_path=}")
        if base_path is not None:
            self.base_path = base_path
        # TODO(zhenghuihuang): cache to disk?
        self.meta_list, self.record_list = self._load_data_yaml(data_yaml)
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        total_len = sum(meta["len"] for meta in self.meta_list)
        logger.info(f"debug {total_len=}")
        for meta in self.meta_list:
            logger.info(f"{meta['len']=}")

        return total_len

    def __getitem__(self, idx: int):

        meta_idx, idx_in_meta = self.tie_index_to_meta(idx)

        try:
            x = self.get_item_func(meta_idx, idx_in_meta)
            return self.get_item_func(meta_idx, idx_in_meta)
        except Exception as e:
            logger.info(
                f"Item {idx} errored, record:\n"
                f"{self.record_list[meta_idx][idx_in_meta]}\n"
                f"Error:\n"
                f"{traceback.format_exc()}"
            )
            if idx_in_meta != 0:
                return self[idx - 1]
            else:
                return self[idx + self.meta_list[meta_idx]["len"] - 1]

    def _load_data_yaml(self, data_yaml: str):
        meta_list = []
        record_list = []
        with open(data_yaml, "r") as yaml_fin:
            data_meta = yaml.load(yaml_fin, Loader=yaml.FullLoader)
            for meta in data_meta["META"]:
                record_json_path = meta["path"]
                if not os.path.exists(record_json_path) and hasattr(self, "base_path") and self.base_path is not None:
                    record_json_path = os.path.join(self.base_path, record_json_path)
                with open(record_json_path) as record_json_fin:
                    record_json = json.load(record_json_fin)
                    record_list.append(record_json)
                    meta["len"] = len(record_json)

                if "type" not in meta:
                    meta["type"] = "default"
                meta["item_len_list"] = [r["len"] for r in record_json]
                meta_list.append(meta)
        return meta_list, record_list

    def tie_index_to_meta(self, idx: int):
        # Initialize the starting index
        start_idx = 0

        # Iterate through the list of dictionaries
        for i, meta in enumerate(self.meta_list):
            # Calculate the ending index for the current collection
            end_idx = start_idx + meta["len"]

            # Check if the given index falls within the current collection
            if start_idx <= idx < end_idx:
                # Calculate the new index within the current collection
                new_index = idx - start_idx
                return i, new_index

            # Update the starting index for the next collection
            start_idx = end_idx

        # If the index is out of range of all collections, raise an error
        raise IndexError("Index out of range")

    def get_item_func(self, meta_idx, idx_in_meta):
        record_item = self.record_list[meta_idx][idx_in_meta]
        # Why origin code has deepcopy?
        #data_item = copy.deepcopy(record_item)

        file_path = record_item["file"]
        if not os.path.exists(file_path) and hasattr(self, "base_path") and self.base_path is not None:
            file_path = os.path.join(self.base_path, file_path)
        with open(file_path, "rb") as f:
            data_item = pickle.load(f)
        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        return {
            "tokens": self.informal_data_format(tokens),
            "cu_seqlens": [i * self.seq_len for i in range(self.micro_batch_size + 1)],
            "indexes": list(range(self.seq_len)) * self.micro_batch_size,
            "labels": self.informal_data_format(labels),
            "type_ids": self.informal_data_format([0]) 
            }

    def informal_data_format(self, data: list):
        return self.extend_data_to_packed_length(data, self.seq_len * self.micro_batch_size)
        #extend_data = self.extend_data_to_packed_length(data, self.seq_len)
        #micro_batched_data = self.data_to_micro_batch(extend_data)
        #return micro_batched_data

    def data_to_micro_batch(self, data: list):
        return [data for _ in range(self.micro_batch_size)]

    # TODO: Hardcode packed length now, fix it
    def extend_data_to_packed_length(self, data: list, packed_length):
        if len(data) > packed_length:
            return data[:packed_length]
        if len(data) == packed_length:
            return data
        # len(data) < packed_length
        return data + [0.0] * (packed_length - len(data))

    def copy(self):
        return copy.deepcopy(self)

