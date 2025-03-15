"""
Usage:
    python tools/convert_ckpt_parallel.py \
    <origin_ckpt_path> <target_ckpt_path> \
    --origin_meta_path <origin_meta_path> --target_meta_path <target_meta_path> \
    --copy_file <True/False> --convert_optimizer <True/False>

    When meta_path is not specified, it will automatically search and load meta in the ckpt path.
    Default to convert optimizer state and copy files.
Example:
    srun -p llm_s python tools/convert_ckpt_parallel.py \
    /llm_ckpt/100 /target_ckpt/converted \
    --target_meta_path /target_ckpt/metadata.pt
"""
import argparse
import os
import shutil
import sys
from collections import OrderedDict, defaultdict

import torch
from torch._utils import _flatten_dense_tensors

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("origin_ckpt_path", type=str, default=None)
    args.add_argument("target_ckpt_path", type=str, default=None)
    args.add_argument("--origin_meta_path", type=str, default=None)
    args.add_argument("--target_meta_path", type=str, default=None)
    args.add_argument("--copy_file", type=bool, default=True, help="enable/disable copy other file.")
    args.add_argument("--convert_optimizer", type=bool, default=True, help="enable/disable optimizer converting.")
    return args.parse_args()


def get_mapping(n, m):
    if m % n != 0:
        raise ValueError("m must be a multiple of n")

    n_list = list(range(n))
    m_list = list(range(m))

    mapping = {}
    for i, n_val in enumerate(n_list):
        mapping[n_val] = m_list[i * int(m / n) : (i + 1) * int(m / n)]

    return mapping


def map_pp_lists(old_pp, new_pp):
    result = []
    old_ranks = list(range(old_pp))
    new_ranks = list(range(new_pp))

    if old_pp > new_pp:
        ratio = old_pp // new_pp
        for i in old_ranks:
            result.append([new_ranks[i // ratio]])
    elif old_pp < new_pp:
        ratio = new_pp // old_pp
        for i in old_ranks:
            result.append(new_ranks[i * ratio : (i + 1) * ratio])
    else:
        for i in old_ranks:
            result.append([new_ranks[i]])

    assert len(result) == old_pp
    return result


def sorted_state_dict(unordered_dict):
    sorted_keys = sorted(unordered_dict.keys())  # 排序键名
    sorted_dict = OrderedDict()
    for key in sorted_keys:
        sorted_dict[key] = unordered_dict[key]
    return sorted_dict


def flatten(input_):
    return _flatten_dense_tensors(input_)


def unflatten_tensor(flat_tensor, states):
    start = 0
    unflat_tensors = []

    for _, state in states.items():
        shape = state["shape"]
        size = torch.prod(torch.tensor(shape))
        tensor = flat_tensor[start : start + size].reshape(*shape)
        unflat_tensors.append(tensor)
        start += size

    length = flat_tensor.shape[0]
    assert start == length, f"{start} should equal to {length}"

    return unflat_tensors


def preprocess_optimizer_state(
    old_tp_size, old_pp_size, old_zero1_size, old_meta, folder, old_tp_mode, old_base_groups, moe_group, **kwargs
):
    # preprocess optimizer_state to unflatten format
    processed_ckpt_states = [
        [[{} for _ in range(old_zero1_size)] for _ in range(old_pp_size)] for _ in range(old_tp_size)
    ]

    if len(moe_group) > 0:
        moe_meta = [[[{} for _ in range(old_zero1_size)] for _ in range(old_pp_size)] for _ in range(old_tp_size)]
        moe_base_groups = [
            [[{} for _ in range(old_zero1_size)] for _ in range(old_pp_size)] for _ in range(old_tp_size)
        ]
        old_moe_base_groups = kwargs["old_moe_base_groups"]
        for pp_rank in range(old_pp_size):
            for ep_rank in range(kwargs["old_ep_size"]):
                for edp_rank in range(kwargs["old_edp_size"]):
                    for ewp_rank in range(kwargs["old_ewp_size"]):
                        rank_map = old_meta["moe_meta"][pp_rank][ep_rank][edp_rank][ewp_rank]["rank_map"]
                        tp_rank = rank_map[kwargs["tp_mode"]]
                        zero1_rank = rank_map["zero1"]
                        assert moe_meta[tp_rank][pp_rank][zero1_rank] == {}
                        moe_meta[tp_rank][pp_rank][zero1_rank] = old_meta["moe_meta"][pp_rank][ep_rank][edp_rank][
                            ewp_rank
                        ]["metaData"]
                        for group_id in moe_meta[tp_rank][pp_rank][zero1_rank]:
                            moe_base_groups[tp_rank][pp_rank][zero1_rank][group_id] = old_moe_base_groups[pp_rank][
                                ep_rank
                            ][edp_rank][ewp_rank][group_id]

    for old_tp_rank in range(old_tp_size):
        for old_pp_rank in range(old_pp_size):
            for old_zero1_rank in range(old_zero1_size):
                ckpt_states = torch.load(
                    os.path.join(folder, f"optimizer_{old_tp_mode}{old_tp_rank}_pp{old_pp_rank}_zo{old_zero1_rank}.pt"),
                    map_location="cpu",
                )
                base_optim_states = ckpt_states["base_optim_states"]["state"]
                flat_fp32_weights = ckpt_states["flat_fp32_weights"]
                processed_state = ckpt_states

                for group_id in list(flat_fp32_weights.keys()):
                    if group_id in moe_group:
                        metaData = moe_meta[old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
                        base_group_id = moe_base_groups[old_tp_rank][old_pp_rank][old_zero1_rank][group_id][0]
                        assert (
                            len(moe_base_groups[old_tp_rank][old_pp_rank][old_zero1_rank][group_id]) == 1
                        ), "unsupported"
                    else:
                        metaData = old_meta["metaData"][old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
                        base_group_id = old_base_groups[old_tp_rank][old_pp_rank][old_zero1_rank][group_id][0]
                        assert (
                            len(old_base_groups[old_tp_rank][old_pp_rank][old_zero1_rank][group_id]) == 1
                        ), "unsupported"

                    exp_avg = base_optim_states[base_group_id]["exp_avg"]
                    exp_avg_sq = base_optim_states[base_group_id]["exp_avg_sq"]
                    flat_tensor = flat_fp32_weights[group_id]

                    unflat_exp_avg = unflatten_tensor(exp_avg, metaData)
                    unflat_exp_avg_sq = unflatten_tensor(exp_avg_sq, metaData)
                    unflat_tensor = unflatten_tensor(flat_tensor, metaData)

                    if group_id != base_group_id:
                        processed_state["base_optim_states"]["state"][group_id] = processed_state["base_optim_states"][
                            "state"
                        ].pop(base_group_id)

                    processed_state["base_optim_states"]["state"][group_id]["exp_avg"] = unflat_exp_avg
                    processed_state["base_optim_states"]["state"][group_id]["exp_avg_sq"] = unflat_exp_avg_sq
                    processed_state["flat_fp32_weights"][group_id] = unflat_tensor

                processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank] = processed_state

    return processed_ckpt_states


def check_optimizer_convert(target_meta, target_states, group_id):
    fqn_list = list(target_meta.keys())
    index = len(target_states["global_fqn"]) - 1
    meta_fqn = fqn_list[index]
    meta_shape = target_meta[meta_fqn]["shape"]
    meta_group_id = target_meta[meta_fqn]["group_id"]
    states_fqn = target_states["global_fqn"][-1]
    states_shape = target_states["fp32_weights"][-1].shape

    assert meta_fqn == states_fqn, f"states_fqn {states_fqn} and meta_fqn {meta_fqn} are not the same."
    assert (
        meta_group_id == group_id
    ), f"For {states_fqn}: group_id {states_shape} and meta_group_id {meta_shape} are not the same."
    assert (
        meta_shape == states_shape
    ), f"For {states_fqn}: states_shape {states_shape} and meta_shape {meta_shape} are not the same."


def sort_optimizer_state(target_dict, meta_fqns):
    assert len(target_dict.keys()) == len(meta_fqns), f"fqn length error: {len(target_dict.keys())} != {len(meta_fqns)}"
    assert set(target_dict.keys()) == set(meta_fqns), f"fqns not equal: {list(target_dict.keys())} != {list(meta_fqns)}"
    sorted_exp_avg = [target_dict[key]["exp_avg"] for key in meta_fqns]
    sorted_exp_avg_sq = [target_dict[key]["exp_avg_sq"] for key in meta_fqns]
    sorted_fp32_weights = [target_dict[key]["fp32_weights"] for key in meta_fqns]

    return sorted_exp_avg, sorted_exp_avg_sq, sorted_fp32_weights


def model_tp_merge(
    old_pp_rank,
    new_states,
    old_tp_size,
    new_tp_size,
    old_tp_mode,
    ratio,
    old_meta_data,
    new_meta_data,
    old_map_local_to_global,
    new_meta,
    folder,
):
    candidate_states = [defaultdict(list) for _ in range(new_tp_size)]
    for old_tp_rank in range(old_tp_size):
        ckpt_states = torch.load(
            os.path.join(folder, f"model_{old_tp_mode}{old_tp_rank}_pp{old_pp_rank}.pt"), map_location="cpu"
        )
        for fqn, tensor in ckpt_states.items():
            new_tp_rank = old_tp_rank // ratio
            candidate_states[new_tp_rank][fqn].append(tensor)

    for new_tp_rank, states in enumerate(candidate_states):
        for fqn, tensor_list in states.items():
            global_fqn = old_map_local_to_global[old_pp_rank][fqn]
            tp_dim = new_meta_data[new_tp_rank][global_fqn]["tp_dim"]
            assert tp_dim == old_meta_data[0][global_fqn]["tp_dim"], (
                f"{global_fqn} tp_dim in old and new meta are not equal: "
                f"new={tp_dim}, old={old_meta_data[0][global_fqn]['tp_dim']}"
            )

            new_pp_rank = new_meta_data[new_tp_rank][global_fqn]["pp"]
            new_zero1_rank = new_meta_data[new_tp_rank][global_fqn]["zero1"]
            new_fqn = new_meta_data[new_tp_rank][global_fqn]["fqn"]
            group_id = new_meta_data[new_tp_rank][global_fqn]["group_id"]
            assert "bias" not in global_fqn

            if old_tp_size == new_tp_size or tp_dim == -1:
                if old_tp_size == new_tp_size:
                    assert len(tensor_list) == 1
                else:
                    if "bias" not in global_fqn:
                        assert torch.equal(tensor_list[0], tensor_list[1]), (
                            f"{global_fqn} should not be splited by tp, "
                            f"but the tensors in different checkpoints are not equal. {fqn}"
                        )
                new_states[new_tp_rank][new_pp_rank][new_fqn] = tensor_list[0].detach().clone()
            else:
                new_states[new_tp_rank][new_pp_rank][new_fqn] = torch.concat(tensor_list, dim=tp_dim).detach().clone()

            splited_shape = new_states[new_tp_rank][new_pp_rank][new_fqn].shape
            meta_shape = new_meta["metaData"][new_tp_rank][new_pp_rank][new_zero1_rank][group_id][global_fqn]["shape"]
            assert (
                splited_shape == meta_shape
            ), f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"


def model_tp_split(
    split_maps,
    old_pp_rank,
    old_tp_size,
    new_states,
    old_meta_data,
    new_meta_data,
    ratio,
    old_tp_mode,
    old_map_local_to_global,
    new_meta,
    folder,
):
    for old_tp_rank in range(old_tp_size):
        ckpt_states = torch.load(
            os.path.join(folder, f"model_{old_tp_mode}{old_tp_rank}_pp{old_pp_rank}.pt"), map_location="cpu"
        )
        for fqn, tensor in ckpt_states.items():
            global_fqn = old_map_local_to_global[old_pp_rank][fqn]
            tp_dim = old_meta_data[old_tp_rank][global_fqn]["tp_dim"]
            assert tp_dim == new_meta_data[0][global_fqn]["tp_dim"], (
                f"{global_fqn} tp_dim in old and new meta are not equal: "
                f"old={tp_dim}, new={new_meta_data[0][fqn]['tp_dim']}"
            )

            if tp_dim != -1:
                split_size = tensor.size()[tp_dim] // ratio
                new_tp_splits = torch.split(tensor, split_size, dim=tp_dim)

            for i, new_tp_rank in enumerate(split_maps[old_tp_rank]):
                # bias is not splitted and only exists in 0 rank for row tp
                if tp_dim == -1:
                    if "bias" in global_fqn and new_tp_rank > 0:
                        break

                new_pp_rank = new_meta_data[new_tp_rank][global_fqn]["pp"]
                new_zero1_rank = new_meta_data[new_tp_rank][global_fqn]["zero1"]
                new_fqn = new_meta_data[new_tp_rank][global_fqn]["fqn"]
                group_id = new_meta_data[new_tp_rank][global_fqn]["group_id"]

                if tp_dim == -1:
                    new_states[new_tp_rank][new_pp_rank][new_fqn] = tensor.detach().clone()
                    splited_shape = new_states[new_tp_rank][new_pp_rank][new_fqn].shape
                    meta_shape = new_meta["metaData"][new_tp_rank][new_pp_rank][new_zero1_rank][group_id][global_fqn][
                        "shape"
                    ]
                    assert (
                        splited_shape == meta_shape
                    ), f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"
                else:
                    new_states[new_tp_rank][new_pp_rank][new_fqn] = new_tp_splits[i].detach().clone()
                    splited_shape = new_states[new_tp_rank][new_pp_rank][new_fqn].shape
                    meta_shape = new_meta["metaData"][new_tp_rank][new_pp_rank][new_zero1_rank][group_id][global_fqn][
                        "shape"
                    ]
                    assert (
                        splited_shape == meta_shape
                    ), f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"


def optimizer_tp_merge(
    new_tp_size,
    old_tp_size,
    old_pp_rank,
    old_zero1_rank,
    old_meta,
    new_meta_data,
    processed_ckpt_states,
    new_states,
    ratio,
    moe_group,
):
    candidate_exp_avg = [defaultdict(list) for _ in range(new_tp_size)]
    candidate_exp_avg_sq = [defaultdict(list) for _ in range(new_tp_size)]
    candidate_fp32_weights = [defaultdict(list) for _ in range(new_tp_size)]
    for old_tp_rank in range(old_tp_size):
        ckpt_states = processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank]
        for group_id in ckpt_states["flat_fp32_weights"].keys():
            if group_id in moe_group:
                continue
            old_metaData = old_meta["metaData"][old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
            exp_avg_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg"]
            exp_avg_sq_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg_sq"]
            fp32_weights_list = ckpt_states["flat_fp32_weights"][group_id]
            new_tp_rank = old_tp_rank // ratio
            for i, global_fqn in enumerate(list(old_metaData.keys())):
                assert group_id == new_meta_data[0][global_fqn]["group_id"]
                candidate_exp_avg[new_tp_rank][global_fqn].append(exp_avg_list[i])
                candidate_exp_avg_sq[new_tp_rank][global_fqn].append(exp_avg_sq_list[i])
                candidate_fp32_weights[new_tp_rank][global_fqn].append(fp32_weights_list[i])

    for new_tp_rank in range(new_tp_size):
        for global_fqn in candidate_fp32_weights[new_tp_rank].keys():
            splited_exp_avg = candidate_exp_avg[new_tp_rank][global_fqn]
            splited_exp_avg_sq = candidate_exp_avg_sq[new_tp_rank][global_fqn]
            splited_fp32_weights = candidate_fp32_weights[new_tp_rank][global_fqn]

            tp_dim = new_meta_data[new_tp_rank][global_fqn]["tp_dim"]
            new_pp_rank = new_meta_data[new_tp_rank][global_fqn]["pp"]
            new_zero1_rank = new_meta_data[new_tp_rank][global_fqn]["zero1"]
            group_id = new_meta_data[new_tp_rank][global_fqn]["group_id"]
            if group_id not in new_states[new_tp_rank][new_pp_rank][new_zero1_rank]:
                new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id] = {}
            target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
            assert global_fqn not in target_new_states, f"repeated global_fqn {global_fqn}"
            target_new_states[global_fqn] = {"exp_avg": None, "exp_avg_sq": None, "fp32_weights": None}

            if old_tp_size == new_tp_size or tp_dim == -1:
                if old_tp_size == new_tp_size:
                    assert len(splited_fp32_weights) == 1
                else:
                    if "bias" not in global_fqn:
                        assert torch.equal(splited_fp32_weights[0], splited_fp32_weights[1]), (
                            f"{global_fqn} should not be splited by tp,"
                            "but the tensors in different checkpoints are not equal."
                        )
                target_new_states[global_fqn]["exp_avg"] = splited_exp_avg[0].detach().clone()
                target_new_states[global_fqn]["exp_avg_sq"] = splited_exp_avg_sq[0].detach().clone()
                target_new_states[global_fqn]["fp32_weights"] = splited_fp32_weights[0].detach().clone()
            else:
                target_new_states[global_fqn]["exp_avg"] = torch.concat(splited_exp_avg, dim=tp_dim).detach().clone()
                target_new_states[global_fqn]["exp_avg_sq"] = (
                    torch.concat(splited_exp_avg_sq, dim=tp_dim).detach().clone()
                )
                target_new_states[global_fqn]["fp32_weights"] = (
                    torch.concat(splited_fp32_weights, dim=tp_dim).detach().clone()
                )


def optimizer_tp_split(
    split_maps,
    old_tp_size,
    old_pp_rank,
    old_zero1_rank,
    old_meta,
    new_meta_data,
    processed_ckpt_states,
    new_states,
    ratio,
    moe_group,
):
    for old_tp_rank in range(old_tp_size):
        ckpt_states = processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank]
        for group_id in ckpt_states["flat_fp32_weights"].keys():
            if group_id in moe_group:
                continue
            old_metaData = old_meta["metaData"][old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
            exp_avg_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg"]
            exp_avg_sq_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg_sq"]
            fp32_weights_list = ckpt_states["flat_fp32_weights"][group_id]

            for i, global_fqn in enumerate(list(old_metaData.keys())):
                tp_dim = old_metaData[global_fqn]["tp_dim"]

                if tp_dim != -1:
                    split_size = old_metaData[global_fqn]["shape"][tp_dim] // ratio
                    new_exp_avg_splits = torch.split(exp_avg_list[i], split_size, dim=tp_dim)
                    new_exp_avg_sq_splits = torch.split(exp_avg_sq_list[i], split_size, dim=tp_dim)
                    new_fp32_weights_splits = torch.split(fp32_weights_list[i], split_size, dim=tp_dim)

                for j, new_tp_rank in enumerate(split_maps[old_tp_rank]):
                    # bias is not splitted and only exists in 0 rank for row tp
                    if tp_dim == -1:
                        if "bias" in global_fqn and new_tp_rank > 0:
                            break

                    new_pp_rank = new_meta_data[new_tp_rank][global_fqn]["pp"]
                    new_zero1_rank = new_meta_data[new_tp_rank][global_fqn]["zero1"]

                    if group_id not in new_states[new_tp_rank][new_pp_rank][new_zero1_rank]:
                        new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id] = {}

                    if tp_dim == -1:
                        target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
                        assert global_fqn not in target_new_states, f"repeated global_fqn {global_fqn}"
                        target_new_states[global_fqn] = {"exp_avg": None, "exp_avg_sq": None, "fp32_weights": None}
                        target_new_states[global_fqn]["exp_avg"] = exp_avg_list[i].detach().clone()
                        target_new_states[global_fqn]["exp_avg_sq"] = exp_avg_sq_list[i].detach().clone()
                        target_new_states[global_fqn]["fp32_weights"] = fp32_weights_list[i].detach().clone()
                    else:
                        target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
                        assert global_fqn not in target_new_states, f"repeated global_fqn {global_fqn}"
                        target_new_states[global_fqn] = {"exp_avg": None, "exp_avg_sq": None, "fp32_weights": None}
                        target_new_states[global_fqn]["exp_avg"] = new_exp_avg_splits[j].detach().clone()
                        target_new_states[global_fqn]["exp_avg_sq"] = new_exp_avg_sq_splits[j].detach().clone()
                        target_new_states[global_fqn]["fp32_weights"] = new_fp32_weights_splits[j].detach().clone()


def moe_tp_merge(
    folder,
    layer_id,
    ep_id,
    old_ewp_size,
    new_ewp_size,
    old_pp_rank,
    old_tp_mode,
    ratio,
    old_moe_map_local_to_global,
    new_states,
    old_moe_meta_data,
    new_moe_meta_data,
):
    candidate_states = [defaultdict(list) for _ in range(new_ewp_size)]
    for old_ewp_rank in range(old_ewp_size):
        ckpt_states = torch.load(
            os.path.join(folder, f"model_moe_layer{layer_id}_expert{ep_id}_{old_tp_mode}{old_ewp_rank}.pt"),
            map_location="cpu",
        )
        for fqn, tensor in ckpt_states.items():
            new_tp_rank = old_ewp_rank // ratio
            candidate_states[new_tp_rank][fqn].append(tensor)

    for new_tp_rank, states in enumerate(candidate_states):
        for fqn, tensor_list in states.items():
            global_fqn = old_moe_map_local_to_global[old_pp_rank][fqn]
            tp_dim = new_moe_meta_data[global_fqn]["tp_dim"]
            assert tp_dim == old_moe_meta_data[global_fqn]["tp_dim"], (
                f"{global_fqn} tp_dim in old and new meta are not equal: "
                f"new={tp_dim}, old={old_moe_meta_data[global_fqn]['tp_dim']}"
            )

            new_fqn = new_moe_meta_data[global_fqn]["fqn"]
            assert tp_dim != -1

            if old_ewp_size != new_ewp_size:
                new_states[new_tp_rank][new_fqn] = torch.concat(tensor_list, dim=tp_dim).detach().clone()
            else:
                assert len(tensor_list) == 1
                new_states[new_tp_rank][new_fqn] = tensor_list[0].detach().clone()

            splited_shape = new_states[new_tp_rank][new_fqn].shape
            meta_shape = new_moe_meta_data[global_fqn]["shape"]
            assert (
                splited_shape == meta_shape
            ), f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"


def moe_tp_split(
    folder,
    layer_id,
    ep_id,
    old_ewp_size,
    old_pp_rank,
    old_tp_mode,
    ratio,
    split_maps,
    old_moe_map_local_to_global,
    new_states,
    old_moe_meta_data,
    new_moe_meta_data,
):
    for old_ewp_rank in range(old_ewp_size):
        ckpt_states = torch.load(
            os.path.join(folder, f"model_moe_layer{layer_id}_expert{ep_id}_{old_tp_mode}{old_ewp_rank}.pt"),
            map_location="cpu",
        )
        for fqn, tensor in ckpt_states.items():
            global_fqn = old_moe_map_local_to_global[old_pp_rank][fqn]
            tp_dim = old_moe_meta_data[global_fqn]["tp_dim"]
            assert tp_dim == new_moe_meta_data[global_fqn]["tp_dim"], (
                f"{global_fqn} tp_dim in old and new meta are not equal: "
                f"old={tp_dim}, new={new_moe_meta_data[global_fqn]['tp_dim']}"
            )

            split_size = tensor.size()[tp_dim] // ratio
            new_tp_splits = torch.split(tensor, split_size, dim=tp_dim)
            for i, new_tp_rank in enumerate(split_maps[old_ewp_rank]):
                new_fqn = new_moe_meta_data[global_fqn]["fqn"]
                new_states[new_tp_rank][new_fqn] = new_tp_splits[i].detach().clone()
                splited_shape = new_states[new_tp_rank][new_fqn].shape
                meta_shape = new_moe_meta_data[global_fqn]["shape"]
                assert (
                    splited_shape == meta_shape
                ), f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"


def optimizer_moe_tp_merge(
    new_ewp_size,
    old_ewp_size,
    old_pp_rank,
    old_ep_rank,
    old_edp_rank,
    old_meta,
    new_meta,
    old_moe_meta_data,
    new_moe_meta_data,
    tp_mode,
    processed_ckpt_states,
    new_states,
    ratio,
    moe_group,
    moe_rank_map,
):
    candidate_exp_avg = [defaultdict(list) for _ in range(new_ewp_size)]
    candidate_exp_avg_sq = [defaultdict(list) for _ in range(new_ewp_size)]
    candidate_fp32_weights = [defaultdict(list) for _ in range(new_ewp_size)]
    for old_ewp_rank in range(old_ewp_size):
        old_rank_map = old_meta["moe_meta"][old_pp_rank][old_ep_rank][old_edp_rank][old_ewp_rank]["rank_map"]
        old_tp_rank = old_rank_map[tp_mode]
        old_zero1_rank = old_rank_map["zero1"]
        ckpt_states = processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank]

        for group_id in moe_group:
            old_metaData = old_meta["moe_meta"][old_pp_rank][old_ep_rank][old_edp_rank][old_ewp_rank]["metaData"][
                group_id
            ]
            exp_avg_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg"]
            exp_avg_sq_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg_sq"]
            fp32_weights_list = ckpt_states["flat_fp32_weights"][group_id]
            new_ewp_rank = old_ewp_rank // ratio
            for i, global_fqn in enumerate(list(old_metaData.keys())):
                assert group_id == new_moe_meta_data[global_fqn]["group_id"]
                candidate_exp_avg[new_ewp_rank][global_fqn].append(exp_avg_list[i])
                candidate_exp_avg_sq[new_ewp_rank][global_fqn].append(exp_avg_sq_list[i])
                candidate_fp32_weights[new_ewp_rank][global_fqn].append(fp32_weights_list[i])

    for new_ewp_rank in range(new_ewp_size):
        for global_fqn in candidate_fp32_weights[new_ewp_rank].keys():
            splited_exp_avg = candidate_exp_avg[new_ewp_rank][global_fqn]
            splited_exp_avg_sq = candidate_exp_avg_sq[new_ewp_rank][global_fqn]
            splited_fp32_weights = candidate_fp32_weights[new_ewp_rank][global_fqn]

            tp_dim = old_moe_meta_data[global_fqn]["tp_dim"]
            assert tp_dim == new_moe_meta_data[global_fqn]["tp_dim"]

            new_pp_rank = new_moe_meta_data[global_fqn]["pp"]
            new_ep_rank = new_moe_meta_data[global_fqn]["ep"]
            new_edp_rank = new_moe_meta_data[global_fqn]["edp"]
            group_id = new_moe_meta_data[global_fqn]["group_id"]

            new_rank_map = new_meta["moe_meta"][new_pp_rank][new_ep_rank][new_edp_rank][new_ewp_rank]["rank_map"]
            new_tp_rank = new_rank_map[tp_mode]
            new_zero1_rank = new_rank_map["zero1"]
            key = f"{new_tp_rank}_{new_pp_rank}_{new_zero1_rank}"
            if key not in moe_rank_map:
                moe_rank_map[key] = {"ep_rank": new_ep_rank, "edp_rank": new_edp_rank, "ewp_rank": new_ewp_rank}
            else:
                assert list(moe_rank_map[key].items()) == list(
                    {"ep_rank": new_ep_rank, "edp_rank": new_edp_rank, "ewp_rank": new_ewp_rank}.items()
                )

            if group_id not in new_states[new_tp_rank][new_pp_rank][new_zero1_rank]:
                new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id] = {}

            target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
            assert global_fqn not in target_new_states, f"repeated global_fqn {global_fqn}"
            target_new_states[global_fqn] = {"exp_avg": None, "exp_avg_sq": None, "fp32_weights": None}

            assert tp_dim != -1
            if old_ewp_size != new_ewp_size:
                target_new_states[global_fqn]["exp_avg"] = torch.concat(splited_exp_avg, dim=tp_dim).detach().clone()
                target_new_states[global_fqn]["exp_avg_sq"] = (
                    torch.concat(splited_exp_avg_sq, dim=tp_dim).detach().clone()
                )
                target_new_states[global_fqn]["fp32_weights"] = (
                    torch.concat(splited_fp32_weights, dim=tp_dim).detach().clone()
                )
            else:
                assert len(splited_fp32_weights) == 1
                target_new_states[global_fqn]["exp_avg"] = splited_exp_avg[0].detach().clone()
                target_new_states[global_fqn]["exp_avg_sq"] = splited_exp_avg_sq[0].detach().clone()
                target_new_states[global_fqn]["fp32_weights"] = splited_fp32_weights[0].detach().clone()


def optimizer_moe_tp_split(
    split_maps,
    old_pp_rank,
    old_ewp_size,
    old_ep_rank,
    old_edp_rank,
    old_meta,
    new_meta,
    new_moe_meta_data,
    tp_mode,
    processed_ckpt_states,
    new_states,
    ratio,
    moe_group,
    moe_rank_map,
):
    for old_ewp_rank in range(old_ewp_size):
        old_rank_map = old_meta["moe_meta"][old_pp_rank][old_ep_rank][old_edp_rank][old_ewp_rank]["rank_map"]
        old_tp_rank = old_rank_map[tp_mode]
        old_zero1_rank = old_rank_map["zero1"]
        ckpt_states = processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank]
        for group_id in moe_group:
            old_metaData = old_meta["moe_meta"][old_pp_rank][old_ep_rank][old_edp_rank][old_ewp_rank]["metaData"][
                group_id
            ]
            exp_avg_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg"]
            exp_avg_sq_list = ckpt_states["base_optim_states"]["state"][group_id]["exp_avg_sq"]
            fp32_weights_list = ckpt_states["flat_fp32_weights"][group_id]

            for i, global_fqn in enumerate(list(old_metaData.keys())):
                tp_dim = old_metaData[global_fqn]["tp_dim"]
                new_pp_rank = new_moe_meta_data[global_fqn]["pp"]
                new_ep_rank = new_moe_meta_data[global_fqn]["ep"]
                new_edp_rank = new_moe_meta_data[global_fqn]["edp"]

                if tp_dim != -1:
                    split_size = old_metaData[global_fqn]["shape"][tp_dim] // ratio
                    new_exp_avg_splits = torch.split(exp_avg_list[i], split_size, dim=tp_dim)
                    new_exp_avg_sq_splits = torch.split(exp_avg_sq_list[i], split_size, dim=tp_dim)
                    new_fp32_weights_splits = torch.split(fp32_weights_list[i], split_size, dim=tp_dim)

                for j, new_ewp_rank in enumerate(split_maps[old_ewp_rank]):
                    new_rank_map = new_meta["moe_meta"][new_pp_rank][new_ep_rank][new_edp_rank][new_ewp_rank][
                        "rank_map"
                    ]
                    new_tp_rank = new_rank_map[tp_mode]
                    new_zero1_rank = new_rank_map["zero1"]
                    key = f"{new_tp_rank}_{new_pp_rank}_{new_zero1_rank}"
                    if key in moe_rank_map:
                        assert moe_rank_map[key]["ep_rank"] == new_ep_rank, "Error: Mapping exception occurred"
                        assert moe_rank_map[key]["edp_rank"] == new_edp_rank, "Error: Mapping exception occurred"
                        assert moe_rank_map[key]["ewp_rank"] == new_ewp_rank, "Error: Mapping exception occurred"
                    moe_rank_map[key] = {"ep_rank": new_ep_rank, "edp_rank": new_edp_rank, "ewp_rank": new_ewp_rank}

                    if group_id not in new_states[new_tp_rank][new_pp_rank][new_zero1_rank]:
                        new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id] = {}
                    target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
                    assert global_fqn not in target_new_states, f"repeated global_fqn {global_fqn}"
                    target_new_states[global_fqn] = {"exp_avg": None, "exp_avg_sq": None, "fp32_weights": None}

                    if tp_dim == -1:
                        target_new_states[global_fqn]["exp_avg"] = exp_avg_list[i].detach().clone()
                        target_new_states[global_fqn]["exp_avg_sq"] = exp_avg_sq_list[i].detach().clone()
                        target_new_states[global_fqn]["fp32_weights"] = fp32_weights_list[i].detach().clone()
                    else:
                        target_new_states[global_fqn]["exp_avg"] = new_exp_avg_splits[j].detach().clone()
                        target_new_states[global_fqn]["exp_avg_sq"] = new_exp_avg_sq_splits[j].detach().clone()
                        target_new_states[global_fqn]["fp32_weights"] = new_fp32_weights_splits[j].detach().clone()


def convert_moe_layer(
    folder,
    old_pp_size,
    old_ewp_size,
    new_ewp_size,
    num_layers,
    num_experts,
    saved_folder,
    old_tp_mode,
    new_tp_mode,
    old_moe_map_local_to_global,
    old_moe_meta_data,
    new_moe_meta_data,
):
    print("Begin moe layer convert", flush=True)
    for layer_id in range(num_layers):
        old_pp_rank = layer_id // (num_layers // old_pp_size)
        for ep_id in range(num_experts):
            new_states = [{} for _ in range(new_ewp_size)]
            if old_ewp_size >= new_ewp_size:
                assert old_ewp_size % new_ewp_size == 0, f"Cannot convert {old_ewp_size} TP to {new_ewp_size} TP."
                ratio = old_ewp_size // new_ewp_size
                moe_tp_merge(
                    folder,
                    layer_id,
                    ep_id,
                    old_ewp_size,
                    new_ewp_size,
                    old_pp_rank,
                    old_tp_mode,
                    ratio,
                    old_moe_map_local_to_global,
                    new_states,
                    old_moe_meta_data,
                    new_moe_meta_data,
                )
            else:
                assert new_ewp_size % old_ewp_size == 0, f"Cannot convert {old_ewp_size} TP to {new_ewp_size} TP."
                split_maps = get_mapping(old_ewp_size, new_ewp_size)
                ratio = new_ewp_size // old_ewp_size
                moe_tp_split(
                    folder,
                    layer_id,
                    ep_id,
                    old_ewp_size,
                    old_pp_rank,
                    old_tp_mode,
                    ratio,
                    split_maps,
                    old_moe_map_local_to_global,
                    new_states,
                    old_moe_meta_data,
                    new_moe_meta_data,
                )

            for new_ewp_rank in range(new_ewp_size):
                file_name = f"model_moe_layer{layer_id}_expert{ep_id}_{new_tp_mode}{new_ewp_rank}.pt"
                torch.save(new_states[new_ewp_rank], os.path.join(saved_folder, file_name))

    print("Finish moe layer convert", flush=True)


def convert_modeling_ckpt(
    old_pp_size,
    new_pp_size,
    old_tp_size,
    new_tp_size,
    old_meta_data,
    new_meta_data,
    old_map_local_to_global,
    old_tp_mode,
    new_tp_mode,
    folder,
    saved_folder,
    new_states,
    new_meta,
):
    print("Begin model convert", flush=True)
    for old_pp_rank in range(old_pp_size):
        if old_tp_size >= new_tp_size:
            assert old_tp_size % new_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
            ratio = old_tp_size // new_tp_size
            model_tp_merge(
                old_pp_rank,
                new_states,
                old_tp_size,
                new_tp_size,
                old_tp_mode,
                ratio,
                old_meta_data,
                new_meta_data,
                old_map_local_to_global,
                new_meta,
                folder,
            )
        else:
            assert new_tp_size % old_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
            split_maps = get_mapping(old_tp_size, new_tp_size)
            ratio = new_tp_size // old_tp_size
            model_tp_split(
                split_maps,
                old_pp_rank,
                old_tp_size,
                new_states,
                old_meta_data,
                new_meta_data,
                ratio,
                old_tp_mode,
                old_map_local_to_global,
                new_meta,
                folder,
            )

    for new_tp_rank in range(new_tp_size):
        for new_pp_rank in range(new_pp_size):
            file_name = f"model_{new_tp_mode}{new_tp_rank}_pp{new_pp_rank}.pt"
            states = sorted_state_dict(new_states[new_tp_rank][new_pp_rank])
            torch.save(states, os.path.join(saved_folder, file_name))

    print("Finish model convert", flush=True)


def convert_optimizer_ckpt(
    old_meta,
    new_meta,
    old_pp_size,
    new_pp_size,
    old_tp_size,
    new_tp_size,
    old_zero1_size,
    new_zero1_size,
    new_meta_data,
    new_tp_mode,
    saved_folder,
    new_states,
    processed_ckpt_states,
    moe_group,
    new_base_groups,
    **kwargs,
):
    print("Begin optimizer convert", flush=True)
    for old_pp_rank in range(old_pp_size):
        for old_zero1_rank in range(old_zero1_size):
            if old_tp_size >= new_tp_size:
                assert old_tp_size % new_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
                ratio = old_tp_size // new_tp_size
                optimizer_tp_merge(
                    new_tp_size,
                    old_tp_size,
                    old_pp_rank,
                    old_zero1_rank,
                    old_meta,
                    new_meta_data,
                    processed_ckpt_states,
                    new_states,
                    ratio,
                    moe_group,
                )
            else:
                assert new_tp_size % old_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
                split_maps = get_mapping(old_tp_size, new_tp_size)
                ratio = new_tp_size // old_tp_size
                optimizer_tp_split(
                    split_maps,
                    old_tp_size,
                    old_pp_rank,
                    old_zero1_rank,
                    old_meta,
                    new_meta_data,
                    processed_ckpt_states,
                    new_states,
                    ratio,
                    moe_group,
                )

    if len(moe_group) > 0:
        print("Begin optimizer moe group convert", flush=True)
        old_ep_size = kwargs["old_ep_size"]
        old_edp_size = kwargs["old_edp_size"]
        old_ewp_size = kwargs["old_ewp_size"]
        new_ewp_size = kwargs["new_ewp_size"]
        tp_mode = kwargs["tp_mode"]
        old_moe_meta_data = kwargs["old_moe_meta_data"]
        new_moe_meta_data = kwargs["new_moe_meta_data"]
        moe_rank_map = {}

        for old_pp_rank in range(old_pp_size):
            for old_ep_rank in range(old_ep_size):
                for old_edp_rank in range(old_edp_size):
                    if old_ewp_size >= new_ewp_size:
                        assert (
                            old_ewp_size % new_ewp_size == 0
                        ), f"Cannot convert {old_ewp_size} ewp/tp to {new_ewp_size} ewp/tp."
                        ratio = old_ewp_size // new_ewp_size
                        optimizer_moe_tp_merge(
                            new_ewp_size,
                            old_ewp_size,
                            old_pp_rank,
                            old_ep_rank,
                            old_edp_rank,
                            old_meta,
                            new_meta,
                            old_moe_meta_data,
                            new_moe_meta_data,
                            tp_mode,
                            processed_ckpt_states,
                            new_states,
                            ratio,
                            moe_group,
                            moe_rank_map,
                        )
                    else:
                        assert (
                            new_ewp_size % old_ewp_size == 0
                        ), f"Cannot convert {old_ewp_size} ewp/tp to {new_ewp_size} ewp/tp."
                        split_maps = get_mapping(old_ewp_size, new_ewp_size)
                        ratio = new_ewp_size // old_ewp_size
                        optimizer_moe_tp_split(
                            split_maps,
                            old_pp_rank,
                            old_ewp_size,
                            old_ep_rank,
                            old_edp_rank,
                            old_meta,
                            new_meta,
                            new_moe_meta_data,
                            tp_mode,
                            processed_ckpt_states,
                            new_states,
                            ratio,
                            moe_group,
                            moe_rank_map,
                        )

    for new_tp_rank in range(new_tp_size):
        for new_pp_rank in range(new_pp_size):
            for new_zero1_rank in range(new_zero1_size):
                file_name = f"optimizer_{new_tp_mode}{new_tp_rank}_pp{new_pp_rank}_zo{new_zero1_rank}.pt"
                optimizer_state = new_states[new_tp_rank][new_pp_rank][new_zero1_rank]
                base_state = processed_ckpt_states[0][0][0]
                step = base_state["base_optim_states"]["state"][0]["step"]
                base_state["base_optim_states"]["state"] = {}
                base_state["flat_fp32_weights"] = {}
                if "zero_devide_optim_plan" in base_state:
                    base_state.pop("zero_devide_optim_plan")

                # Ensure that the order of group_id is consistent
                sorted_groups = sorted(list(optimizer_state.keys()))
                for group_id in sorted_groups:
                    if group_id in moe_group:
                        key = f"{new_tp_rank}_{new_pp_rank}_{new_zero1_rank}"
                        ep_rank = moe_rank_map[key]["ep_rank"]
                        edp_rank = moe_rank_map[key]["edp_rank"]
                        ewp_rank = moe_rank_map[key]["ewp_rank"]
                        metaData = new_meta["moe_meta"][new_pp_rank][ep_rank][edp_rank][ewp_rank]["metaData"]
                    else:
                        metaData = new_meta["metaData"][new_tp_rank][new_pp_rank][new_zero1_rank]

                    meta_fqns = metaData[group_id].keys()
                    sorted_exp_avg, sorted_exp_avg_sq, sorted_fp32_weights = sort_optimizer_state(
                        optimizer_state[group_id], meta_fqns
                    )
                    flat_exp_avg = flatten(sorted_exp_avg)
                    flat_exp_avg_sq = flatten(sorted_exp_avg_sq)
                    flat_fp32_weights = flatten(sorted_fp32_weights)
                    state = {"step": step, "exp_avg": flat_exp_avg, "exp_avg_sq": flat_exp_avg_sq}
                    base_state["base_optim_states"]["state"][group_id] = state
                    base_state["flat_fp32_weights"][group_id] = flat_fp32_weights

                # overwrite base states group_id
                base_groups = new_base_groups[new_tp_rank][new_pp_rank][new_zero1_rank]
                for group_id in base_groups:
                    if group_id not in base_state["base_optim_states"]["state"]:
                        continue
                    base_group_id = base_groups[group_id][0]
                    if group_id in base_state["base_optim_states"]["state"]:
                        assert len(base_groups[group_id]) == 1
                        if group_id != base_group_id:
                            base_state["base_optim_states"]["state"][base_group_id] = base_state["base_optim_states"][
                                "state"
                            ].pop(group_id)
                    base_state["base_optim_states"]["param_groups"][group_id]["params"] = base_groups[group_id]

                if len(moe_group) > 0:
                    base_groups = kwargs["new_moe_base_groups"][new_pp_rank][ep_rank][edp_rank][ewp_rank]
                    for group_id in base_groups:
                        base_group_id = base_groups[group_id][0]
                        if group_id in base_state["base_optim_states"]["state"]:
                            assert len(base_groups[group_id]) == 1
                            if group_id != base_group_id:
                                base_state["base_optim_states"]["state"][base_group_id] = base_state[
                                    "base_optim_states"
                                ]["state"].pop(group_id)
                        base_state["base_optim_states"]["param_groups"][group_id]["params"] = base_groups[group_id]
                torch.save(base_state, os.path.join(saved_folder, file_name))

    print("Finish optimizer convert", flush=True)


if __name__ == "__main__":

    args = parse_args()
    folder = args.origin_ckpt_path
    saved_folder = args.target_ckpt_path

    if args.origin_meta_path is not None:
        old_meta_path = args.origin_meta_path
    else:
        old_meta_path = os.path.join(folder, "metadata.pt")

    if args.target_meta_path is not None:
        new_meta_path = args.target_meta_path
    else:
        new_meta_path = os.path.join(saved_folder, "metadata.pt")

    assert os.path.exists(
        old_meta_path
    ), "old meta file does not exist, plese generate it before converting checkpoint."
    assert os.path.exists(
        new_meta_path
    ), "new meta file does not exist, plese generate it before converting checkpoint."

    # read and process metaData for original ckpt
    old_meta = torch.load(old_meta_path, map_location="cpu")
    old_pp_size = old_meta["parallel_setting"]["pp_size"]
    old_zero1_size = old_meta["parallel_setting"]["zero1_size"]

    if "tp_size" in old_meta["parallel_setting"]:
        old_tp_mode = "tp"
    elif "wp_size" in old_meta["parallel_setting"]:
        old_tp_mode = "wp"
    else:
        assert False, "tp or wp should be in parallel setting."
    old_tp_size = old_meta["parallel_setting"][f"{old_tp_mode}_size"]

    # preprocess base group_id
    # old_base_groups[tp_pp_zero1_groupId] = [base_group_id]
    old_base_groups = [[[{} for _ in range(old_zero1_size)] for _ in range(old_pp_size)] for _ in range(old_tp_size)]
    for tp_rank in range(old_tp_size):
        for pp_rank in range(old_pp_size):
            for zero_rank in range(old_zero1_size):
                old_base_groups[tp_rank][pp_rank][zero_rank] = old_meta["metaData"][tp_rank][pp_rank][zero_rank].pop(
                    "base_groups"
                )

    # To facilitate key query, aggregate meta_data.
    old_meta_data = [{} for _ in range(old_tp_size)]
    for tp_rank in range(old_tp_size):
        for pp_rank in range(old_pp_size):
            for zero_rank in range(old_zero1_size):
                assert "base_groups" not in old_meta["metaData"][tp_rank][pp_rank][zero_rank]
                for states in old_meta["metaData"][tp_rank][pp_rank][zero_rank].values():
                    old_meta_data[tp_rank].update(states)

    # map local fqn to global fqn
    old_map_local_to_global = [{} for _ in range(old_pp_size)]
    for tp_rank in range(old_tp_size):
        for global_fqn, states in old_meta_data[tp_rank].items():
            if states["fqn"] not in old_map_local_to_global[states["pp"]]:
                old_map_local_to_global[states["pp"]][states["fqn"]] = global_fqn
            else:
                assert global_fqn == old_map_local_to_global[states["pp"]][states["fqn"]]

    # read and process metaData for target ckpt
    new_meta = torch.load(new_meta_path, map_location="cpu")
    new_pp_size = new_meta["parallel_setting"]["pp_size"]
    new_zero1_size = new_meta["parallel_setting"]["zero1_size"]

    if "tp_size" in new_meta["parallel_setting"]:
        new_tp_mode = "tp"
    elif "wp_size" in new_meta["parallel_setting"]:
        new_tp_mode = "wp"
    else:
        assert False, "tp or wp should be in parallel setting."
    # TODO: support converting between tp and wp
    assert old_tp_mode == new_tp_mode, "Do not support converting between tp and wp currently."
    new_tp_size = new_meta["parallel_setting"][f"{new_tp_mode}_size"]

    # preprocess base group_id
    # new_base_groups[tp_pp_zero1_groupId] = [base_group_id]
    new_base_groups = [[[{} for _ in range(new_zero1_size)] for _ in range(new_pp_size)] for _ in range(new_tp_size)]
    for tp_rank in range(new_tp_size):
        for pp_rank in range(new_pp_size):
            for zero_rank in range(new_zero1_size):
                new_base_groups[tp_rank][pp_rank][zero_rank] = new_meta["metaData"][tp_rank][pp_rank][zero_rank].pop(
                    "base_groups"
                )

    assert set(new_meta["metaData"][0][0][0].keys()) == set(
        old_meta["metaData"][0][0][0].keys()
    ), "Error: old meta and new meta have diffent group_id lists."
    group_id_list = list(new_meta["metaData"][0][0][0].keys())

    # To facilitate key query, aggregate meta_data.
    new_meta_data = [{} for _ in range(new_tp_size)]
    for tp_rank in range(new_tp_size):
        for pp_rank in range(new_pp_size):
            for zero_rank in range(new_zero1_size):
                assert "base_groups" not in new_meta["metaData"][tp_rank][pp_rank][zero_rank]
                for states in new_meta["metaData"][tp_rank][pp_rank][zero_rank].values():
                    new_meta_data[tp_rank].update(states)

    # moe
    num_layers = old_meta["parallel_setting"]["num_layers"]
    num_experts = old_meta["parallel_setting"]["num_experts"]
    assert num_layers == new_meta["parallel_setting"]["num_layers"]
    assert num_experts == new_meta["parallel_setting"]["num_experts"]
    if num_experts > 1:
        old_ep_size = old_meta["parallel_setting"]["ep_size"]
        old_edp_size = old_meta["parallel_setting"]["edp_size"]
        old_ewp_size = old_meta["parallel_setting"]["ewp_size"]
        new_ep_size = new_meta["parallel_setting"]["ep_size"]
        new_edp_size = new_meta["parallel_setting"]["edp_size"]
        new_ewp_size = new_meta["parallel_setting"]["ewp_size"]

        assert len(old_meta["moe_group"]) > 0, "moe group should not be empty"
        assert (
            old_meta["moe_group"] == new_meta["moe_group"]
        ), "Error: old meta and new meta have diffent moe grou lists."

        # preprocess base group_id
        # old_moe_base_groups[pp_ep_edp_ewp_groupId] = [base_group_id]
        old_moe_base_groups = [
            [[[{} for _ in range(old_ewp_size)] for _ in range(old_edp_size)] for _ in range(old_ep_size)]
            for _ in range(old_pp_size)
        ]
        for pp_rank in range(old_pp_size):
            for ep_rank in range(old_ep_size):
                for edp_rank in range(old_edp_size):
                    for ewp_rank in range(old_ewp_size):
                        old_moe_base_groups[pp_rank][ep_rank][edp_rank][ewp_rank] = old_meta["moe_meta"][pp_rank][
                            ep_rank
                        ][edp_rank][ewp_rank]["metaData"].pop("base_groups")

        old_moe_meta_data = {}
        for pp_rank in range(old_pp_size):
            for ep_rank in range(old_ep_size):
                for edp_rank in range(old_edp_size):
                    assert "base_groups" not in old_meta["moe_meta"][pp_rank][ep_rank][edp_rank][0]["metaData"]
                    for states in old_meta["moe_meta"][pp_rank][ep_rank][edp_rank][0]["metaData"].values():
                        old_moe_meta_data.update(states)

        old_moe_map_local_to_global = [{} for _ in range(old_pp_size)]
        for global_fqn, states in old_moe_meta_data.items():
            old_moe_map_local_to_global[states["pp"]][states["fqn"]] = global_fqn

        # preprocess base group_id
        # new_moe_base_groups[pp_ep_edp_ewp_groupId] = [base_group_id]
        new_moe_base_groups = [
            [[[{} for _ in range(new_ewp_size)] for _ in range(new_edp_size)] for _ in range(new_ep_size)]
            for _ in range(new_pp_size)
        ]
        for pp_rank in range(new_pp_size):
            for ep_rank in range(new_ep_size):
                for edp_rank in range(new_edp_size):
                    for ewp_rank in range(new_ewp_size):
                        new_moe_base_groups[pp_rank][ep_rank][edp_rank][ewp_rank] = new_meta["moe_meta"][pp_rank][
                            ep_rank
                        ][edp_rank][ewp_rank]["metaData"].pop("base_groups")

        new_moe_meta_data = {}
        for pp_rank in range(new_pp_size):
            for ep_rank in range(new_ep_size):
                for edp_rank in range(new_edp_size):
                    assert "base_groups" not in new_meta["moe_meta"][pp_rank][ep_rank][edp_rank][0]["metaData"]
                    for states in new_meta["moe_meta"][pp_rank][ep_rank][edp_rank][0]["metaData"].values():
                        new_moe_meta_data.update(states)

    new_states = [[{} for _ in range(new_pp_size)] for _ in range(new_tp_size)]
    convert_modeling_ckpt(
        old_pp_size=old_pp_size,
        new_pp_size=new_pp_size,
        old_tp_size=old_tp_size,
        new_tp_size=new_tp_size,
        old_meta_data=old_meta_data,
        new_meta_data=new_meta_data,
        old_map_local_to_global=old_map_local_to_global,
        old_tp_mode=old_tp_mode,
        new_tp_mode=new_tp_mode,
        folder=folder,
        saved_folder=saved_folder,
        new_states=new_states,
        new_meta=new_meta,
    )

    if num_experts > 1:
        convert_moe_layer(
            folder=folder,
            old_pp_size=old_pp_size,
            old_ewp_size=old_ewp_size,
            new_ewp_size=new_ewp_size,
            num_layers=num_layers,
            num_experts=num_experts,
            saved_folder=saved_folder,
            old_tp_mode=old_tp_mode,
            new_tp_mode=new_tp_mode,
            old_moe_map_local_to_global=old_moe_map_local_to_global,
            old_moe_meta_data=old_moe_meta_data,
            new_moe_meta_data=new_moe_meta_data,
        )

    if args.convert_optimizer:
        if num_experts > 1:
            kwargs = {
                "old_ep_size": old_ep_size,
                "old_edp_size": old_edp_size,
                "old_ewp_size": old_ewp_size,
                "new_ewp_size": new_ewp_size,
                "tp_mode": old_tp_mode,
                "old_moe_meta_data": old_moe_meta_data,
                "new_moe_meta_data": new_moe_meta_data,
                "old_moe_base_groups": old_moe_base_groups,
                "new_moe_base_groups": new_moe_base_groups,
            }
        else:
            kwargs = {}

        processed_ckpt_states = preprocess_optimizer_state(
            old_tp_size,
            old_pp_size,
            old_zero1_size,
            old_meta,
            folder,
            old_tp_mode,
            old_base_groups,
            old_meta["moe_group"],
            **kwargs,
        )
        new_states = [[[{} for _ in range(new_zero1_size)] for _ in range(new_pp_size)] for _ in range(new_tp_size)]
        convert_optimizer_ckpt(
            old_meta=old_meta,
            new_meta=new_meta,
            old_pp_size=old_pp_size,
            new_pp_size=new_pp_size,
            old_tp_size=old_tp_size,
            new_tp_size=new_tp_size,
            old_zero1_size=old_zero1_size,
            new_zero1_size=new_zero1_size,
            new_meta_data=new_meta_data,
            new_tp_mode=new_tp_mode,
            saved_folder=saved_folder,
            new_states=new_states,
            processed_ckpt_states=processed_ckpt_states,
            moe_group=old_meta["moe_group"],
            new_base_groups=new_base_groups,
            **kwargs,
        )

    if args.copy_file:
        file_list = ["context.pt", "sampler.pt", "schedulder.pt"]
        for file_name in file_list:
            src = os.path.join(folder, file_name)
            dst = os.path.join(saved_folder, file_name)
            shutil.copy(src, dst)
        print(f"Finish copy: {file_list}", flush=True)
