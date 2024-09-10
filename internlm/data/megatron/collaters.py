import torch


def megatron_collate_fn(batch, micro_num, micro_bsz, seq_len):

    input_ids_result = [[] for _ in range(micro_num)]
    labels_result = [[] for _ in range(micro_num)]
    cu_seqlens = []
    cu_seqlens_list = []
    indexes = []
    indexes_list = []

    for i, item in enumerate(batch):
        assert i < micro_num * micro_bsz
        seq_len_list = item["text"]
        assert len(seq_len_list) == seq_len + 1

        micro_bsz_index = i % micro_bsz
        micro_num_index = i // micro_bsz

        input_ids_result[micro_num_index].append(seq_len_list[:-1])
        labels_result[micro_num_index].append(seq_len_list[1:])

        cu_seqlens.append(seq_len * micro_bsz_index)
        indexes = indexes + list(range(seq_len))

        if micro_bsz_index == micro_bsz - 1:
            input_ids_result[micro_num_index] = torch.cat(
                [torch.from_numpy(arr).long() for arr in input_ids_result[micro_num_index]], dim=0
            )
            labels_result[micro_num_index] = torch.cat(
                [torch.from_numpy(arr).long() for arr in labels_result[micro_num_index]], dim=0
            )
            cu_seqlens.append(seq_len * micro_bsz)
            cu_seqlens_list.append(torch.IntTensor(cu_seqlens))
            cu_seqlens = []
            indexes_list.append(torch.IntTensor(indexes))
            indexes = []

    input_ids = torch.stack(input_ids_result)
    labels = torch.stack(labels_result)
    indexes = torch.stack(indexes_list)

    return {
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens_list,
        "indexes": indexes,
        "type_ids": torch.zeros(micro_num, micro_bsz * seq_len, dtype=torch.int64),
    }, labels
