#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import re

import torch

from internlm.core.context import global_context as gpc


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


def unpack_data(input_ids, cu_seqlens, is_type_ids: bool = False, padding_v: int = 0):
    """
    input_ids: if input_ids is not type_ids, the shape is (1, packed_length)
               else the shape is (micro_num, packed_length)
    is_type_ids: whether the input_ids is type_ids

    Return:
    output: if input_ids is not type ids, the shape is (micro_bsz, max_length)
            else the shape is (micro_num, micro_bsz, max_length)
    """
    bsz = input_ids.shape[0]

    num_seq = gpc.config.data["micro_bsz"]
    seq_len_ = gpc.config.data.seq_len
    dtype_ = input_ids.dtype

    outputs = torch.empty(bsz, num_seq, seq_len_, device=input_ids.device, dtype=dtype_).fill_(padding_v)

    for i in range(bsz):
        output = torch.empty(num_seq, seq_len_, device=input_ids.device, dtype=dtype_).fill_(padding_v)
        cu_seqlens_slice = cu_seqlens[i]
        for j in range(num_seq):
            length = cu_seqlens_slice[j + 1] - cu_seqlens_slice[j]
            output[j, 0:length] = input_ids[i, cu_seqlens_slice[j] : cu_seqlens_slice[j + 1]]
        outputs[i] = output

    # if the input_ids is not type_ids, we need squeeze the first dimension if it is 1.
    if bsz == 1 and not is_type_ids:
        outputs = outputs.squeeze(0)

    return outputs
