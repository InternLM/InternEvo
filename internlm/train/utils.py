from typing import Dict, Tuple

import torch
from torch import nn

from internlm.core.context.parallel_context import ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.core.naive_amp import unwrap_naive_amp
from internlm.model.modules.utils import is_moe_param
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


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

    new_groups = {}
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
            # moe param means MoE is enabled
            if is_moe_param(param):
                new_groups[param.group_name]["params"].append(param)
            elif param.dtype == torch.float32 and gpc.config.model.dtype != torch.float32:
                new_groups["fp32"]["params"].append(param)
            else:
                origin_params.append(param)

        # default param group, which is the first group in the param groups
        pgroup["params"] = origin_params
        pgroup["optimizer_mode"] = ParallelMode.ZERO1

    # param groups may contain empty groups, such as fp32
    param_groups.extend(new_groups.values())

    return tuple(param_groups)


def create_param_groups(model, weight_decay):
    parameters = {
        "params": [param for param in model.parameters() if param.requires_grad],
        "name": "default",
        "weight_decay": weight_decay,
    }
    return split_params_into_different_groups_for_optimizer(parameters)


def map_param_block(model):
    for _chunk in unwrap_naive_amp(model):
        for name, children in _chunk.named_children():
            if isinstance(children, nn.ModuleList):
                for idx, block in enumerate(children):
                    block_name = name + f"_{idx}"
                    for param in block.parameters():
                        setattr(param, "block_name", block_name)
            else:
                for param in children.parameters():
                    setattr(param, "block_name", name)


def timeout_input(printout, default, timeout=None, interactive=True):
    if not interactive:
        return default
    import select
    import sys

    if gpc.is_rank_for_log():
        logger.info(printout)

    i, _, _ = select.select([sys.stdin], [], [], timeout)
    if i:
        msg = sys.stdin.readline().strip()
        return default if len(msg) == 0 else msg
    else:
        return default
