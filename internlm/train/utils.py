from typing import Dict, Tuple

import torch

from internlm.core.context.parallel_context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.model.utils import is_moe_param
from internlm.utils.parallel import is_tensor_data_parallel_parameter, is_using_isp


def split_params_into_different_groups_for_optimizer(
    param_groups: Tuple[Dict],
) -> Tuple[Dict]:
    """Split parameters into different groups for optimizer

    Args:
        param_groups (Tuple[Dict]): The list of parameter groups to split
        Input Example:
        >>> (
        >>>     {'name': 'default', 'params': [tensor], 'weight_decay' :xxx},
        >>> )

    Returns:
        Tuple[Dict]: list of params groups for optimizer
        Output Example:
        >>> (
        >>>     {'name': 'default', 'params': [tensor], 'weight_decay' :xxx},
        >>>     {'name': 'embed_head', 'params': [tensor], 'weight_decay' :xxx},
        >>>     {'name': 'fp32', 'params': [tensor], 'weight_decay' :xxx},
        >>> )
    """

    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # create new groups for IS_TENSOR_DATA_PARALLEL parameter group
    new_groups = {}
    if is_using_isp():
        new_groups["embed_head"] = {"name": "embed_head", "params": [], "optimizer_mode": ParallelMode.DATA}
    # create new groups for fp32 parameter group
    new_groups["fp32"] = {"name": "fp32", "params": [], "optimizer_mode": ParallelMode.ZERO1}

    if gpc.config.model.get("num_experts", 1) > 1:
        for key in gpc.expert_parallel_group_names:
            new_groups[key] = {"name": key, "moe": True, "params": [], "optimizer_mode": ParallelMode.EXPERT_DATA}

    for pgroup in param_groups:
        # copy attribute from origin group, we assume the input param_groups only
        # have one group, so the attribute will not be copyed multiple times.
        for ori_key in pgroup.keys():
            if ori_key not in ("name", "params"):
                for _, group in new_groups.items():
                    group[ori_key] = pgroup[ori_key]
        # assign param
        origin_params = []
        for param in pgroup["params"]:
            if is_tensor_data_parallel_parameter(param):
                # should not be here if not isp mode
                new_groups["embed_head"]["params"].append(param)
            elif param.dtype == torch.float32:
                new_groups["fp32"]["params"].append(param)
            # moe param means MoE is enabled
            elif is_moe_param(param):
                new_groups[param.group_name]["params"].append(param)
            else:
                origin_params.append(param)

        # default param group, which is the first group in the param groups
        pgroup["params"] = origin_params
        pgroup["optimizer_mode"] = ParallelMode.ZERO1

    # param groups may contain empty groups, such as fp32
    param_groups.extend(new_groups.values())

    return tuple(param_groups)


def create_param_groups(model, weight_decay):
    parameters = {"params": list(model.parameters()), "name": "default", "weight_decay": weight_decay}
    return split_params_into_different_groups_for_optimizer(parameters)
