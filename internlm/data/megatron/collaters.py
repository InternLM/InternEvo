import torch


def megatron_collate_fn(batch, micro_num, micro_bsz, seq_len):
    input_ids_list = [[] for _ in range(micro_num)]
    labels_list = [[] for _ in range(micro_num)]
    cu_seqlens_list = []
    indexes_list = []

    assert len(batch) == micro_bsz * micro_num
    for idx, b in enumerate(batch):
        tokens = b["text"]
        # The length of megatron preprocessed data samples is (seq_len + 1)
        # So we use the first seq_len tokens as input and the last seq_len tokens as shifted labels
        assert len(tokens) == seq_len + 1
        micro_bsz_index = idx % micro_bsz
        micro_num_index = idx // micro_bsz
        input_ids_list[micro_num_index].append(tokens[:-1])
        labels_list[micro_num_index].append(tokens[1:])

        if micro_bsz_index == micro_bsz - 1:
            # Since megatron data sample is numpy format, we need to convert it to tensor and concate within micro batch
            input_ids_list[micro_num_index] = torch.cat(
                [torch.from_numpy(arr) for arr in input_ids_list[micro_num_index]], dim=0
            )
            labels_list[micro_num_index] = torch.cat(
                [torch.from_numpy(arr) for arr in labels_list[micro_num_index]], dim=0
            )
            cu_seqlens_list.append(torch.IntTensor([i * seq_len for i in range(micro_bsz + 1)]))
            indexes_list.append(torch.IntTensor(list(range(seq_len)) * micro_bsz))

    return {
        "input_ids": torch.stack(input_ids_list),
        "cu_seqlens": cu_seqlens_list,
        "indexes": torch.stack(indexes_list),
        "type_ids": torch.zeros(micro_num, micro_bsz * seq_len, dtype=torch.int64),
    }, torch.stack(labels_list)
