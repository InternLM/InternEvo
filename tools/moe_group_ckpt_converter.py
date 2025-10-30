import argparse
import os
import re
import shutil
import sys

import torch
from tqdm import tqdm

sys.path.append(".")
import internlm  # noqa: E402,F401 # pylint: disable=W0611,C0413

moe_str_prefix = None
weight_key_suffix = ".weight"


def load(fp):
    with open(fp, "rb") as f:
        pt_data = torch.load(f, map_location="cpu")
    return pt_data


def list_to_group_ckpt_tp(src, tgt, ep_size, num_layer, num_local_experts, max_tp):
    print("Converting checkpoints from sequence module list to group mlp...")

    for layer_id in tqdm(range(num_layer)):
        for tp_rank in range(max_tp):
            for expp_rank in range(ep_size):
                # merge local experts into grouped mlp
                expert_state_dict = dict()
                # expert_w_state[key][expert] = weight
                expert_w_state = {"w1": [], "w2": [], "w3": []}
                expert_ids = range(num_local_experts * expp_rank, num_local_experts * (expp_rank + 1))
                for global_expert_id in expert_ids:
                    fn = f"model_moe_layer{layer_id}_expert{global_expert_id}_tp{tp_rank}.pt"
                    fp = os.path.join(src, fn)
                    origin_state = load(fp)
                    pattern = r"(.*?\.moe_layer\.experts\.wrapped_experts)\.\d+\.(w\d+)(?:\.weight)?"
                    for key, weight in origin_state.items():
                        moe_str_prefix, w_i = re.search(pattern, key).group(1), re.search(pattern, key).group(2)
                        # [d2, d1] -> [d1, d2]
                        expert_w_state[w_i].append(weight.T)
                # k*[d1, d2] -> [k, d1, d2]
                for key, weights in expert_w_state.items():
                    local_key = f"{moe_str_prefix}.{expp_rank}.{key}{weight_key_suffix}"
                    expert_state_dict[local_key] = torch.stack(weights, dim=0)

                torch.save(
                    expert_state_dict, os.path.join(tgt, f"model_moe_layer{layer_id}_expert{expp_rank}_tp{tp_rank}.pt")
                )


def group_to_list_ckpt_tp(src, tgt, ep_size, num_layer, num_local_experts, max_tp):
    print("Converting checkpoints from group mlp list to sequence module...")

    for layer_id in tqdm(range(num_layer)):
        for tp_rank in range(max_tp):
            for expp_rank in range(ep_size):
                # split group mlp to local experts, expert_w_state[key][expert] = weight
                expert_w_state = {"w1": [], "w2": [], "w3": []}
                fn = f"model_moe_layer{layer_id}_expert{expp_rank}_tp{tp_rank}.pt"
                fp = os.path.join(src, fn)
                origin_state = load(fp)
                pattern = r"(.*?\.moe_layer\.experts\.wrapped_experts)\.\d+\.(w\d+)(?:\.weight)?"
                for local_expert_id in range(num_local_experts):
                    expert_state_dict = dict()
                    global_expert_id = expp_rank * num_local_experts + local_expert_id
                    for key, weight in origin_state.items():
                        moe_str_prefix, w_i = re.search(pattern, key).group(1), re.search(pattern, key).group(2)
                        # [k, d1, d2] -> k * [d1, d2]
                        expert_w_state[w_i] = weight.chunk(num_local_experts)
                        local_key = key.replace(f"{moe_str_prefix}.{expp_rank}", f"{moe_str_prefix}.{global_expert_id}")
                        # [d2, d1] -> [d1, d2]
                        value = expert_w_state[w_i][local_expert_id].squeeze().T
                        expert_state_dict[local_key] = value
                    torch.save(
                        expert_state_dict,
                        os.path.join(tgt, f"model_moe_layer{layer_id}_expert{global_expert_id}_tp{tp_rank}.pt"),
                    )


def list_to_group_ckpt_wp(src, tgt, ep_size, num_layer, num_local_experts, max_wp):
    print("Converting checkpoints from sequence module list to group mlp...")

    for layer_id in tqdm(range(num_layer)):
        for expp_rank in range(ep_size):
            # expert_w_state[key][expert][wp]=weight
            expert_w_state = {
                "w1": [[] for _ in range(num_local_experts)],
                "w2": [[] for _ in range(num_local_experts)],
                "w3": [[] for _ in range(num_local_experts)],
            }
            expert_ids = range(num_local_experts * expp_rank, num_local_experts * (expp_rank + 1))
            for local_expert_id, global_expert_id in enumerate(expert_ids):
                for wp_rank in range(max_wp):
                    fn = f"model_moe_layer{layer_id}_expert{global_expert_id}_wp{wp_rank}.pt"
                    fp = os.path.join(src, fn)
                    origin_state = load(fp)
                    pattern = r"(.*?\.moe_layer\.experts\.wrapped_experts)\.\d+\.(w\d+)(?:\.weight)?"
                    for key, weight in origin_state.items():
                        moe_str_prefix, w_i = re.search(pattern, key).group(1), re.search(pattern, key).group(2)
                        # [d2/2, d1] -> [d1, d2/w]
                        expert_w_state[w_i][local_expert_id].append(weight.T)
            # expert_state_dict[wp][key] = value
            expert_state_dict = [{} for _ in range(max_wp)]
            # k*w*[d1,d2/w] -> k*[d1, d2] -> [k*d1, d2] -> w*[k/w*d1, w*d2]
            for key, weights in expert_w_state.items():
                flat_weights = [torch.cat(row, dim=1) for row in weights]
                full_weights = torch.cat(flat_weights, dim=0).chunk(max_wp, dim=0)
                local_key = f"{moe_str_prefix}.{expp_rank}.{key}{weight_key_suffix}"
                for wp_rank in range(max_wp):
                    expert_state_dict[wp_rank][local_key] = full_weights[wp_rank]

            for wp_rank in range(max_wp):
                torch.save(
                    expert_state_dict[wp_rank],
                    os.path.join(tgt, f"model_moe_layer{layer_id}_expert{expp_rank}_wp{wp_rank}.pt"),
                )


def group_to_list_ckpt_wp(src, tgt, ep_size, num_layer, num_local_experts, max_wp):
    print("Converting checkpoints from group mlp list to sequence module...")

    for layer_id in tqdm(range(num_layer)):
        for expp_rank in range(ep_size):
            # expert_w_state[key][wp]=weight
            expert_w_state = {
                "w1": [None for _ in range(max_wp)],
                "w2": [None for _ in range(max_wp)],
                "w3": [None for _ in range(max_wp)],
            }
            for wp_rank in range(max_wp):
                fn = f"model_moe_layer{layer_id}_expert{expp_rank}_wp{wp_rank}.pt"
                fp = os.path.join(src, fn)
                origin_state = load(fp)
                pattern = r"(.*?\.moe_layer\.experts\.wrapped_experts)\.\d+\.(w\d+)(?:\.weight)?"
                for key, weight in origin_state.items():
                    moe_str_prefix, w_i = re.search(pattern, key).group(1), re.search(pattern, key).group(2)
                    expert_w_state[w_i][wp_rank] = weight

            # expert_state_dict[expert][wp][key] = value
            expert_state_dict = [[{} for _ in range(max_wp)] for _ in range(num_local_experts)]
            for key, weight in expert_w_state.items():
                # w*[k*d1/w, d2] -> [k*d1, d2] -> k*[d1, d2/w]
                full_weight = torch.cat(weight, dim=0).chunk(num_local_experts, dim=0)
                flat_weight = [row.chunk(max_wp, dim=1) for row in full_weight]
                for local_expert_id in range(num_local_experts):
                    for wp_rank in range(max_wp):
                        global_expert_id = expp_rank * num_local_experts + local_expert_id
                        local_key = f"{moe_str_prefix}.{global_expert_id}.{key}{weight_key_suffix}"
                        value = flat_weight[local_expert_id][wp_rank].T
                        expert_state_dict[local_expert_id][wp_rank][local_key] = value

            for local_expert_id in range(num_local_experts):
                global_expert_id = expp_rank * num_local_experts + local_expert_id
                for wp_rank in range(max_wp):
                    torch.save(
                        expert_state_dict[local_expert_id][wp_rank],
                        os.path.join(tgt, f"model_moe_layer{layer_id}_expert{global_expert_id}_wp{wp_rank}.pt"),
                    )


def print_args(args):
    print("-------------- Arguments --------------")
    print(f"Source Path: {args.src}")
    print(f"Target Path: {args.tgt}")
    print(f"Expert Number: {args.num_experts}")
    print(f"EP Size: {args.ep_size}")
    print(f"Convert Mode: {'list to group' if args.convert_mode == 0 else 'group to list'}")
    print("---------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", type=str, help="Input folder")
    parser.add_argument("--tgt", type=str, help="Output folder")
    parser.add_argument("--num-experts", type=int, help="Number of experts")
    parser.add_argument("--ep-size", type=int, help="expert parallel size")
    parser.add_argument("--convert-mode", type=int, help="parallel mode: 0. list to group, 1.group to list")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    fns = list(os.listdir(args.src))
    moe_fns = []
    for fn in fns:
        if fn.startswith("model_moe") and not fn.endswith("md5"):
            moe_fns.append(fn)
        elif (fn.startswith("model_t") or fn.startswith("model_w")) and not fn.endswith("md5"):
            shutil.copyfile(os.path.join(args.src, fn), os.path.join(args.tgt, fn))
    num_layer, max_mp = -1, -1
    mode = None
    for fn in moe_fns:
        _, _, layer_info, _, mp_info = os.path.splitext(fn)[0].split("_")
        num_layer = max(num_layer, int(layer_info[5:]) + 1)
        max_mp = max(max_mp, int(mp_info[2:]) + 1)
        mode = mp_info[:2]
    num_local_experts = args.num_experts // args.ep_size

    if mode == "tp" and args.convert_mode == 0:
        list_to_group_ckpt_tp(args.src, args.tgt, args.ep_size, num_layer, num_local_experts, max_mp)
    elif mode == "tp" and args.convert_mode == 1:
        group_to_list_ckpt_tp(args.src, args.tgt, args.ep_size, num_layer, num_local_experts, max_mp)
    elif mode == "wp" and args.convert_mode == 0:
        list_to_group_ckpt_wp(args.src, args.tgt, args.ep_size, num_layer, num_local_experts, max_mp)
    elif mode == "wp" and args.convert_mode == 1:
        group_to_list_ckpt_wp(args.src, args.tgt, args.ep_size, num_layer, num_local_experts, max_mp)
    else:
        assert False, "unsupport convert mode"
