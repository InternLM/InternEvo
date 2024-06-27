import torch


def nopack_collate_fn(batch, micro_num, micro_bsz, seq_len):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for b in batch:
        attention_mask = b["attention_mask"]
        input_ids = b["input_ids"]
        input_ids = torch.abs(input_ids * attention_mask)
        input_ids = torch.nn.functional.pad(input_ids, (0, seq_len - len(input_ids)), mode="constant", value=0)
        attention_mask = torch.nn.functional.pad(
            attention_mask, (0, seq_len - len(attention_mask)), mode="constant", value=0
        )
        label = torch.tensor([w if w > 0 else -100 for w in input_ids.tolist()][1:] + [-100])
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(label)
    input_ids = torch.stack(input_ids_list)
    attention_mask = torch.stack(attention_mask_list)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "type_ids": torch.zeros(micro_num, micro_bsz, seq_len, dtype=torch.int64),
    }, labels
