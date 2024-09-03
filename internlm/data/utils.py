#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import re

import torch

from internlm.core.context import global_context as gpc
from internlm.core.context.process_group_initializer import ParallelMode


def get_dataset_type_ids_map(path):
    dirlist = list(os.listdir(path))
    dirlist.sort()
    return {key: idx for idx, key in enumerate(dirlist)}


def get_dataset_type_id(dataset_type_ids_map, path):
    match_idxes = []

    for key, idx in dataset_type_ids_map.items():
        if re.search(rf"/[z_]*{key}/", path):
            match_idxes.append(idx)
    assert len(match_idxes) == 1, f"{path}, match_idxes should be 1, but got {match_idxes} from {dataset_type_ids_map}"
    return match_idxes[0]


def _unpack_data(data, cu_seqlens, padding_v: int = 0):
    bsz = data.shape[0]

    num_seq = gpc.config.data["micro_bsz"]
    seq_len_ = gpc.config.data.seq_len
    dtype_ = data.dtype

    outputs = torch.empty(bsz, num_seq, seq_len_, device=data.device, dtype=dtype_).fill_(padding_v)

    for i in range(bsz):
        output = torch.empty(num_seq, seq_len_, device=data.device, dtype=dtype_).fill_(padding_v)
        cu_seqlens_slice = cu_seqlens[i]
        for j in range(num_seq):
            length = cu_seqlens_slice[j + 1] - cu_seqlens_slice[j]
            output[j, 0:length] = data[i, cu_seqlens_slice[j] : cu_seqlens_slice[j + 1]]
        outputs[i] = output

    return outputs


def unpack_type_ids(type_ids, cu_seqlens):
    return _unpack_data(type_ids, cu_seqlens)


def unpack_data(data, label):

    data["input_ids"] = _unpack_data(data["input_ids"], data["cu_seqlens"], padding_v=0).squeeze(0)
    label = _unpack_data(label, data["cu_seqlens"], padding_v=-100).squeeze(0)

    data.pop("cu_seqlens")
    data.pop("indexes")

    return data, label


def packed_data_normalizer(data, label):
    # Should we normalize packed data in this form of this data processor
    # or let the dataset handle it? Currently inclined towards the latter.
    assert data["input_ids"].shape[0] == 1, "data should be packed with batch size 1"

    data["indexes"] = data["indexes"][0]
    data["cu_seqlens"] = data["cu_seqlens"][0].squeeze(0)
    data["max_seqlen"] = (data["cu_seqlens"][1:] - data["cu_seqlens"][:-1]).max().item()

    # If model has inject_info and data_helper is enabled, we provide cu_seqlens and max_seqlen in gpc
    if "inject_info" in gpc.config.model and gpc.config.model.inject_info.get("data_helper", False):
        gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"] = data.pop("cu_seqlens")
        gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"] = data.pop("max_seqlen")
        data["position_ids"] = data.pop("indexes")

    return data, label
