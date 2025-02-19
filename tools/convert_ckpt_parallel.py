import argparse
import os
import shutil
import sys
from collections import defaultdict

import torch
from torch._utils import _flatten_dense_tensors
from itertools import cycle
from collections import OrderedDict


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../"))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("origin_model_path", type=str, default=None)
    args.add_argument("target_model_path", type=str, default=None)
    args.add_argument("--target_tp_size", type=int, default=0)
    args.add_argument("--target_pp_size", type=int, default=0)
    args.add_argument("--model_size", type=str, default="7B", choices=["7B", "20B", "70B"])
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
            result.append(new_ranks[i * ratio:(i + 1) * ratio])
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
    """
    根据目标形状，将扁平化的张量拆分为多个子张量。

    :param flat_tensor: 扁平化的张量
    :param shapes: 每个子张量的目标形状(list of tuples)
    :return: 切分后的多个子张量列表
    """
    start = 0
    unflat_tensors = []
    
    for _, state in states.items():
        shape = state['shape']
        size = torch.prod(torch.tensor(shape))  # 计算每个子张量的大小
        tensor = flat_tensor[start:start + size].reshape(*shape)  # 切分并恢复形状
        unflat_tensors.append(tensor)
        start += size  # 更新起始位置
    
    return unflat_tensors


def preprocess_optimizer_state(old_tp_size, old_pp_size, old_zero1_size, old_meta, folder):
    processed_ckpt_states = [[[{} for _ in range(old_zero1_size)] for _ in range(old_pp_size)] for _ in range(old_tp_size)]
    for old_tp_rank in range(old_tp_size):
        for old_pp_rank in range(old_pp_size):
            for old_zero1_rank in range(old_zero1_size):
                ckpt_states = torch.load(os.path.join(folder, f"optimizer_tp{old_tp_rank}_pp{old_pp_rank}_zo{old_zero1_rank}.pt"), map_location="cpu")
                base_optim_states = ckpt_states['base_optim_states']['state']
                flat_fp32_weights = ckpt_states['flat_fp32_weights']
                processed_state = ckpt_states
                for group_id in list(base_optim_states.keys()):
                    exp_avg = base_optim_states[group_id]['exp_avg']
                    exp_avg_sq = base_optim_states[group_id]['exp_avg_sq']
                    flat_tensor = flat_fp32_weights[group_id]
                    metaData = old_meta['metaData'][old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
                    
                    unflat_exp_avg = unflatten_tensor(exp_avg, metaData)
                    unflat_exp_avg_sq = unflatten_tensor(exp_avg_sq, metaData)
                    unflat_tensor = unflatten_tensor(flat_tensor, metaData)
                    
                    processed_state['base_optim_states']['state'][group_id]['exp_avg'] = unflat_exp_avg
                    processed_state['base_optim_states']['state'][group_id]['exp_avg_sq'] = unflat_exp_avg_sq
                    processed_state['flat_fp32_weights'][group_id] = unflat_tensor
                
                processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank] = processed_state
    
    return processed_ckpt_states


def check_optimizer_convert(target_meta, target_states, group_id):
    fqn_list = list(target_meta.keys())
    index = len(target_states['global_fqn']) - 1
    meta_fqn = fqn_list[index]
    meta_shape = target_meta[meta_fqn]['shape']
    meta_group_id = target_meta[meta_fqn]['group_id']
    states_fqn = target_states['global_fqn'][-1]
    states_shape = target_states['fp32_weights'][-1].shape
    
    print(fqn_list)
    print(target_states['global_fqn'])
    assert meta_fqn == states_fqn, f"states_fqn {states_fqn} and meta_fqn {meta_fqn} are not the same."
    assert meta_group_id == group_id, f"For {states_fqn}: group_id {states_shape} and meta_group_id {meta_shape} are not the same."
    assert meta_shape == states_shape, f"For {states_fqn}: states_shape {states_shape} and meta_shape {meta_shape} are not the same."
    

def model_tp_split(split_maps, old_pp_rank, old_tp_size, new_states, old_meta_data, new_meta_data, ratio, tp_mode, old_map_local_to_global, new_meta, folder):
    for old_tp_rank in range(old_tp_size):
        ckpt_states = torch.load(os.path.join(folder, f"model_{tp_mode}{old_tp_rank}_pp{old_pp_rank}.pt"), map_location='cpu')
        for fqn, tensor in ckpt_states.items():
            assert len(tensor.size()) < 3, "Only support 2D or 1D tensors."
            global_fqn = old_map_local_to_global[old_pp_rank][fqn]
            tp_dim = old_meta_data[global_fqn]['tp_dim']
            assert tp_dim == new_meta_data[global_fqn]['tp_dim'], f"{global_fqn} tp_dim in old and new meta are not equal: old={tp_dim}, new={new_meta_data[fqn]['tp_dim']}"
            new_pp_rank = new_meta_data[global_fqn]['pp']
            new_zero1_rank = new_meta_data[global_fqn]['zero1']
            new_fqn = new_meta_data[global_fqn]['fqn']
            group_id = new_meta_data[global_fqn]['group_id']
            
            if tp_dim == -1:
                for _, new_tp_rank in enumerate(split_maps[old_tp_rank]):
                    new_states[new_tp_rank][new_pp_rank][new_fqn] = tensor.detach().clone()
                    splited_shape = new_states[new_tp_rank][new_pp_rank][new_fqn].shape
                    meta_shape = new_meta['metaData'][new_tp_rank][new_pp_rank][new_zero1_rank][group_id][global_fqn]['shape']
                    assert splited_shape == meta_shape, f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"
            else:
                split_size = tensor.size()[tp_dim] // ratio
                new_tp_splits = torch.split(tensor, split_size, dim=tp_dim)
                for i, new_tp_rank in enumerate(split_maps[old_tp_rank]):
                    new_states[new_tp_rank][new_pp_rank][new_fqn] = new_tp_splits[i].detach().clone()
                    splited_shape = new_states[new_tp_rank][new_pp_rank][new_fqn].shape
                    meta_shape = new_meta['metaData'][new_tp_rank][new_pp_rank][new_zero1_rank][group_id][global_fqn]['shape']       
                    assert splited_shape == meta_shape, f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"
            

def optimizer_tp_split(split_maps, old_tp_size, old_pp_rank, old_zero1_rank, old_meta, new_meta, new_meta_data, processed_ckpt_states, new_states, ratio):
    for old_tp_rank in range(old_tp_size):
        ckpt_states = processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank]
        for group_id in ckpt_states['flat_fp32_weights'].keys():
            old_metaData = old_meta['metaData'][old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
            exp_avg_list = ckpt_states['base_optim_states']['state'][group_id]['exp_avg']
            exp_avg_sq_list = ckpt_states['base_optim_states']['state'][group_id]['exp_avg_sq']
            fp32_weights_list = ckpt_states['flat_fp32_weights'][group_id]
            
            for i, global_fqn in enumerate(list(old_metaData.keys())):
                tp_dim = old_metaData[global_fqn]['tp_dim']
                new_pp_rank = new_meta_data[global_fqn]['pp']
                new_zero1_rank = new_meta_data[global_fqn]['zero1']
                
                if tp_dim == -1:
                    for _, new_tp_rank in enumerate(split_maps[old_tp_rank]):
                        target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
                        target_new_states['global_fqn'].append(global_fqn)
                        target_new_states['exp_avg'].append(exp_avg_list[i].detach().clone())
                        target_new_states['exp_avg_sq'].append(exp_avg_sq_list[i].detach().clone())
                        target_new_states['fp32_weights'].append(fp32_weights_list[i].detach().clone())
                        
                        check_optimizer_convert(new_meta['metaData'][new_tp_rank][new_pp_rank][new_zero1_rank][group_id], target_new_states, group_id)  
                else:
                    split_size = old_metaData[global_fqn]['shape'][tp_dim] // ratio
                    new_exp_avg_splits = torch.split(exp_avg_list[i], split_size, dim=tp_dim)
                    new_exp_avg_sq_splits = torch.split(exp_avg_sq_list[i], split_size, dim=tp_dim)
                    new_fp32_weights_splits = torch.split(fp32_weights_list[i], split_size, dim=tp_dim)
                    for j, new_tp_rank in enumerate(split_maps[old_tp_rank]):
                        target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
                        target_new_states['global_fqn'].append(global_fqn)
                        target_new_states['exp_avg'].append(new_exp_avg_splits[j].detach().clone())
                        target_new_states['exp_avg_sq'].append(new_exp_avg_sq_splits[j].detach().clone())
                        target_new_states['fp32_weights'].append(new_fp32_weights_splits[j].detach().clone())
                        
                        print(new_tp_rank, new_pp_rank, new_zero1_rank, group_id)
                        check_optimizer_convert(new_meta['metaData'][new_tp_rank][new_pp_rank][new_zero1_rank][group_id], target_new_states, group_id)                     
                

def model_tp_merge(old_pp_rank, new_states, old_tp_size, new_tp_size, tp_mode, ratio, old_meta_data, new_meta_data, old_map_local_to_global, new_meta, folder):
    candidate_states = [defaultdict(list) for _ in range(new_tp_size)]
    for old_tp_rank in range(old_tp_size):
        ckpt_states = torch.load(os.path.join(folder, f"model_{tp_mode}{old_tp_rank}_pp{old_pp_rank}.pt"), map_location="cpu")
        for fqn, tensor in ckpt_states.items():
            assert len(tensor.size()) < 3, "Only support 2D or 1D tensors."
            new_tp_rank = old_tp_rank // ratio
            candidate_states[new_tp_rank][fqn].append(tensor)

    for new_tp_rank, states in enumerate(candidate_states):
        for fqn, tensor_list in states.items():
            global_fqn = old_map_local_to_global[old_pp_rank][fqn]
            tp_dim = old_meta_data[global_fqn]['tp_dim']
            assert tp_dim == new_meta_data[global_fqn]['tp_dim'], f"{global_fqn} tp_dim in old and new meta are not equal: old={tp_dim}, new={new_meta_data[fqn]['tp_dim']}"
            new_pp_rank = new_meta_data[global_fqn]['pp']
            new_zero1_rank = new_meta_data[global_fqn]['zero1']
            new_fqn = new_meta_data[global_fqn]['fqn']
            group_id = new_meta_data[global_fqn]['group_id']
            
            if tp_dim == -1:
                assert torch.equal(tensor_list[0], tensor_list[1]), f"{global_fqn} should not be splited by tp, but the tensors in different checkpoints are not equal."
                new_states[new_tp_rank][new_pp_rank][new_fqn] = tensor_list[0].detach().clone()
            else:
                new_states[new_tp_rank][new_pp_rank][new_fqn] = torch.concat(tensor_list, dim=tp_dim).detach().clone()
            
            splited_shape = new_states[new_tp_rank][new_pp_rank][new_fqn].shape
            meta_shape = new_meta['metaData'][new_tp_rank][new_pp_rank][new_zero1_rank][group_id][global_fqn]['shape']       
            assert splited_shape == meta_shape, f"{new_fqn}: splited shape {splited_shape} is not euqal to metaData {meta_shape}"
            
            
def optimizer_tp_merge(new_tp_size, old_tp_size, old_pp_rank, old_zero1_rank, old_meta, new_meta, old_meta_data, new_meta_data, processed_ckpt_states, new_states, ratio):
    candidate_exp_avg = [defaultdict(list) for _ in range(new_tp_size)]
    candidate_exp_avg_sq = [defaultdict(list) for _ in range(new_tp_size)]
    candidate_fp32_weights = [defaultdict(list) for _ in range(new_tp_size)]
    for old_tp_rank in range(old_tp_size):
        ckpt_states = processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank]
        for group_id in ckpt_states['flat_fp32_weights'].keys():
            old_metaData = old_meta['metaData'][old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
            exp_avg_list = ckpt_states['base_optim_states']['state'][group_id]['exp_avg']
            exp_avg_sq_list = ckpt_states['base_optim_states']['state'][group_id]['exp_avg_sq']
            fp32_weights_list = ckpt_states['flat_fp32_weights'][group_id]
            new_tp_rank = old_tp_rank // ratio
            for i, global_fqn in enumerate(list(old_metaData.keys())):
                assert group_id == new_meta_data[global_fqn]['group_id']
                candidate_exp_avg[new_tp_rank][global_fqn].append(exp_avg_list[i])
                candidate_exp_avg_sq[new_tp_rank][global_fqn].append(exp_avg_sq_list[i])
                candidate_fp32_weights[new_tp_rank][global_fqn].append(fp32_weights_list[i])
    
    for new_tp_rank in len(candidate_fp32_weights):
        for global_fqn in candidate_fp32_weights.keys():
            splited_exp_avg = candidate_exp_avg[new_tp_rank][global_fqn]
            splited_exp_avg_sq = candidate_exp_avg_sq[new_tp_rank][global_fqn]
            splited_fp32_weights = candidate_fp32_weights[new_tp_rank][global_fqn]
            
            tp_dim = old_meta_data[global_fqn]['tp_dim']
            new_pp_rank = new_meta_data[global_fqn]['pp']
            new_zero1_rank = new_meta_data[global_fqn]['zero1']
            group_id = new_meta_data[global_fqn]['group_id']
            target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
            
            if tp_dim == -1:
                assert torch.equal(splited_fp32_weights[0], splited_fp32_weights[1]), f"{global_fqn} should not be splited by tp, but the tensors in different checkpoints are not equal."
                target_new_states['global_fqn'].append(global_fqn)
                target_new_states['exp_avg'].append(splited_exp_avg[0].detach().clone())
                target_new_states['exp_avg_sq'].append(splited_exp_avg_sq[0].detach().clone())
                target_new_states['fp32_weights'].append(splited_fp32_weights[0].detach().clone())
            else:
                target_new_states['global_fqn'].append(global_fqn)
                target_new_states['exp_avg'].append(torch.concat(splited_exp_avg, dim=tp_dim).detach().clone())
                target_new_states['exp_avg_sq'].append(torch.concat(splited_exp_avg_sq, dim=tp_dim).detach().clone())
                target_new_states['fp32_weights'].append(torch.concat(splited_fp32_weights, dim=tp_dim).detach().clone())
            
            check_optimizer_convert(new_meta['metaData'][new_tp_rank][new_pp_rank][new_zero1_rank][group_id], target_new_states, group_id)  
            

def convert_modeling_ckpt(old_pp_size, new_pp_size, old_tp_size, new_tp_size, old_meta_data, new_meta_data, old_map_local_to_global, tp_mode, folder, saved_folder, new_states):
    for old_pp_rank in range(old_pp_size):
        if old_tp_size != new_tp_size:
            if old_tp_size > new_tp_size:
                assert old_tp_size % new_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
                ratio = old_tp_size // new_tp_size
                model_tp_merge(old_pp_rank, new_states, old_tp_size, new_tp_size, tp_mode, ratio, old_meta_data, new_meta_data, old_map_local_to_global, new_meta, folder)
            else:
                assert new_tp_size % old_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
                split_maps = get_mapping(old_tp_size, new_tp_size)
                ratio = new_tp_size // old_tp_size
                model_tp_split(split_maps, old_pp_rank, old_tp_size, new_states, old_meta_data, new_meta_data, ratio, tp_mode, old_map_local_to_global, new_meta, folder)
        else:
            print(f"New tp and old tp are equal. Directly copy.")
            for old_tp_rank in range(old_tp_size):
                ckpt_states = torch.load(os.path.join(folder, f"model_{tp_mode}{old_tp_rank}_pp{old_pp_rank}.pt"), map_location='cpu')
                for fqn, tensor in ckpt_states.items():
                    new_pp_rank = new_meta_data[fqn]['pp']
                    new_states[new_tp_rank][new_pp_rank][fqn] = tensor.detach().clone()
    for new_tp_rank in range(new_tp_size):
        for new_pp_rank in range(new_pp_size):
            # print(f"pp={new_pp_rank}, tp={new_tp_rank}: {new_states[new_tp_rank][new_pp_rank].keys()}")
            file_name = f"model_{tp_mode}{new_tp_rank}_pp{new_pp_rank}.pt"
            states = sorted_state_dict(new_states[new_tp_rank][new_pp_rank])
            # torch.save(os.path.join(saved_folder, file_name), states)
    
    print(f"Finish Modeling")


def convert_optimizer_ckpt(old_meta, new_meta, old_pp_size, new_pp_size, old_tp_size, new_tp_size, old_zero1_size, new_zero1_size, old_meta_data, new_meta_data, saved_folder, new_states, processed_ckpt_states):
    for old_pp_rank in range(old_pp_size):
        for old_zero1_rank in range(old_zero1_size):
            if old_tp_size != new_tp_size:
                if old_tp_size > new_tp_size:
                    assert old_tp_size % new_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
                    ratio = old_tp_size // new_tp_size
                    optimizer_tp_merge(new_tp_size, old_tp_size, old_pp_rank, old_zero1_rank, old_meta, new_meta, old_meta_data, new_meta_data, processed_ckpt_states, new_states, ratio)
                else:
                    assert new_tp_size % old_tp_size == 0, f"Cannot convert {old_tp_size} TP to {new_tp_size} TP."
                    split_maps = get_mapping(old_tp_size, new_tp_size)
                    ratio = new_tp_size // old_tp_size
                    optimizer_tp_split(split_maps, old_tp_size, old_pp_rank, old_zero1_rank, old_meta, new_meta, new_meta_data, processed_ckpt_states, new_states, ratio)
            else:
                print(f"New tp and old tp are equal. Directly copy.")
                for old_tp_rank in range(old_tp_size):
                    ckpt_states = processed_ckpt_states[old_tp_rank][old_pp_rank][old_zero1_rank]
                    for group_id in ckpt_states['flat_fp32_weights'].keys():
                        old_metaData = old_meta['metaData'][old_tp_rank][old_pp_rank][old_zero1_rank][group_id]
                        exp_avg_list = ckpt_states['base_optim_states']['state'][group_id]['exp_avg']
                        exp_avg_sq_list = ckpt_states['base_optim_states']['state'][group_id]['exp_avg_sq']
                        fp32_weights_list = ckpt_states['flat_fp32_weights'][group_id]
                        for i, global_fqn in enumerate(list(old_metaData.keys())):
                            new_pp_rank = new_meta_data[global_fqn]['pp']
                            new_zero1_rank = new_meta_data[global_fqn]['zero1']
                            target_new_states = new_states[new_tp_rank][new_pp_rank][new_zero1_rank][group_id]
                            target_new_states['global_fqn'].append(global_fqn)
                            target_new_states['exp_avg'].append(exp_avg_list[i].detach().clone())
                            target_new_states['exp_avg_sq'].append(exp_avg_sq_list[i].detach().clone())
                            target_new_states['fp32_weights'].append(fp32_weights_list[i].detach().clone())             
                          
    for new_tp_rank in range(new_tp_size):
        for new_pp_rank in range(new_pp_size):
            for new_zero1_rank in range(new_zero1_size):
                file_name = f"optimizer_tp{new_tp_rank}_pp{new_pp_rank}_zo{new_zero1_rank}.pt"
                optimizer_state = new_states[new_tp_rank][new_pp_rank][new_zero1_rank]
                base_state = processed_ckpt_states[0][0][0]
                step = base_state['base_optim_states']['state'][0]['step']
                base_state['base_optim_states']['state'] = {}
                base_state['flat_fp32_weights'] = {}
                if 'zero_devide_optim_plan' in base_state:
                    base_state.pop('zero_devide_optim_plan')
                    
                for group_id in optimizer_state.keys():
                    if len(optimizer_state[group_id]['global_fqn']) == 0:
                        print(f"Warning: tp{new_tp_rank}_pp{new_pp_rank}_zo{new_zero1_rank}_groupId{group_id} has no param.")
                        continue
                    
                    assert optimizer_state[group_id]['global_fqn'] == list(new_meta['metaData'][new_tp_rank][new_pp_rank][new_zero1_rank][group_id].keys())
                    flat_exp_avg = flatten(optimizer_state[group_id]['exp_avg'])
                    flat_exp_avg_sq = flatten(optimizer_state[group_id]['exp_avg_sq'])
                    flat_fp32_weights = flatten(optimizer_state[group_id]['fp32_weights'])
                    state = {'step': step, 'exp_avg': flat_exp_avg, 'exp_avg_sq': flat_exp_avg_sq}
                    base_state['base_optim_states']['state'][group_id] = state
                    base_state['flat_fp32_weights'][group_id] = flat_fp32_weights
                    
                print(f"optimizer tp{new_tp_rank}_pp{new_pp_rank}_zo{new_zero1_rank}: {base_state}")
                # torch.save(os.path.join(saved_folder, file_name), base_state)
    
    print(f"Finish optimizer")
                
    
if __name__ == "__main__":
    folder = "/mnt/petrelfs/lijiaxing/InternEvo/llm_ckpts_3/5"
    saved_folder = "/mnt/petrelfs/lijiaxing/InternEvo/llm_ckpts_3_new"
    meta_path = "/mnt/petrelfs/lijiaxing/InternEvo/llm_ckpts_3_new/metadata.pt"
    
    
    old_meta = os.path.join(folder, f"metadata.pt")
    assert os.path.exists(old_meta), 'old meta file does not exist, plese generate it before converting checkpoint.'
    
    old_meta = torch.load(old_meta, map_location='cpu')
    old_pp_size = old_meta['parallel_setting']['pp_size']
    old_zero1_size = old_meta['parallel_setting']['zero1_size']
    if 'tp_size' in old_meta['parallel_setting']:
        tp_mode = 'tp'
    elif 'wp_size' in old_meta['parallel_setting']:
        tp_mode = 'wp'
    else:
        assert False, "tp or wp should be in parallel setting."
    old_tp_size = old_meta['parallel_setting'][f"{tp_mode}_size"]

    old_meta_data = {}
    for pp_rank in range(old_pp_size):
        for zero_rank in range(old_zero1_size):
            for states in old_meta['metaData'][0][pp_rank][zero_rank].values():
                old_meta_data.update(states)

    old_map_local_to_global = [{} for _ in range(old_pp_size)]
    for global_fqn, states in old_meta_data.items():
        old_map_local_to_global[states['pp']][states['fqn']] = global_fqn
    

    new_meta = torch.load(meta_path, map_location='cpu')
    new_pp_size = new_meta['parallel_setting']['pp_size']
    new_zero1_size = new_meta['parallel_setting']['zero1_size']
    new_tp_size = new_meta['parallel_setting'][f"{tp_mode}_size"]
    assert set(new_meta['metaData'][0][0][0].keys()) == set(old_meta['metaData'][0][0][0].keys()), "Error: old meta and new meta have diffent group_id lists."
    group_id_list = list(new_meta['metaData'][0][0][0].keys())

    new_meta_data = {}
    for pp_rank in range(new_pp_size):
        for zero_rank in range(new_zero1_size):
            for states in new_meta['metaData'][0][pp_rank][zero_rank].values():
                new_meta_data.update(states)
    
    
    # new_states = [[{} for _ in range(new_pp_size)] for _ in range(new_tp_size)]
    # convert_modeling_ckpt(old_pp_size, new_pp_size, old_tp_size, new_tp_size, old_meta_data, new_meta_data, old_map_local_to_global, tp_mode, folder, saved_folder, new_states)
    
    processed_ckpt_states = preprocess_optimizer_state(old_tp_size, old_pp_size, old_zero1_size, old_meta, folder)
    group_dict = {group_id: {'global_fqn': [], 'exp_avg': [], 'exp_avg_sq': [], 'fp32_weights': []} for group_id in group_id_list}
    new_states = [[[group_dict for _ in range(new_zero1_size)] for _ in range(new_pp_size)] for _ in range(new_tp_size)]
    convert_optimizer_ckpt(old_meta, new_meta, old_pp_size, new_pp_size, old_tp_size, new_tp_size, old_zero1_size, new_zero1_size, old_meta_data, new_meta_data, saved_folder, new_states, processed_ckpt_states)
    
    
    # srun -p llm_s -x HOST-10-140-60-16 --gres=gpu:8 python tools/convert_ckpt_parallel.py > debug1.log 2>&1
    # srun -p llm_s python tools/convert_ckpt_parallel.py 2>&1|tee debug1.log
    
    
    
    