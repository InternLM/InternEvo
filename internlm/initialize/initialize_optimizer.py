from typing import Dict, Tuple, Union

import torch
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import unwrap_naive_amp
from internlm.core.parallel.comm import ISPCommunicatorWrapper, ParamAsyncBcastHandler
from internlm.model.model_ops.modules.utils import is_moe_param
from internlm.solver.optimizer import (
    FSDPadaptOptimizer,
    HybridZeroOptimizer,
    HybridZeroOptimizer_v2,
)
from internlm.solver.optimizer.compatible_adamw import new_compatible_adamw
from internlm.solver.schedulers import Beta2Scheduler, FineTuneCosineAnnealingWarmupLR
from internlm.utils.parallel import is_using_fsdp
from internlm.utils.timeout import llm_timeout


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


@llm_timeout(func_name="initialize_optimizer")
def initialize_optimizer(model: Union[nn.Module, nn.ModuleList], isp_communicator: ISPCommunicatorWrapper = None):
    """
    Initialize optimizer.

    Args:
        model (:class:`torch.nn.Module`): Your model instance to be trained or evaluated.

    Returns:
        A tuple of (optimizer, beta2_scheduler, lr_scheduler).
    """

    adam_cfg = gpc.config.adam
    zero_cfg = gpc.config.hybrid_zero_optimizer
    grad_scal_cfg = gpc.config.grad_scaler
    use_apex_adam = getattr(gpc.config, "use_apex_adam", False)

    if "use_split_tensor_optim" in zero_cfg and zero_cfg.use_split_tensor_optim:
        map_param_block(model)

    params = create_param_groups(model, adam_cfg.weight_decay)

    naive_optimizer = new_compatible_adamw(
        params=params,
        lr=adam_cfg.lr,
        betas=(adam_cfg.adam_beta1, adam_cfg.adam_beta2),
        eps=adam_cfg.adam_eps,
        use_apex_adam=use_apex_adam,
    )

    if (
        zero_cfg.overlap_sync_grad
        and gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
        and gpc.is_pipeline_first_stage() is False
    ):
        # When pipeline parallelism is enabled, we prefer to only enable optimizer
        # gradient communication overlap in the first stage, to avoid amplifying
        # the communication overhead stage by stage in cases where the optimizer
        # communication overhead is greater than the compute overhead.
        # For pipeline stages except the first, even if overlap is not enabled,
        # their gradient synchronization overhead can be well hidden by
        # the inherent bubbles of pipeline parallelism.
        zero_cfg.overlap_sync_grad = False

    if zero_cfg.overlap_sync_param:
        param_bcast_sync_handler = ParamAsyncBcastHandler(ParallelMode.ZERO1, model, isp_communicator)
    else:
        param_bcast_sync_handler = None

    if not is_using_fsdp():
        if (
            "use_split_tensor_optim" not in gpc.config.hybrid_zero_optimizer
            or not gpc.config.hybrid_zero_optimizer.use_split_tensor_optim
        ):
            optimizer = HybridZeroOptimizer(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
        else:
            optimizer = HybridZeroOptimizer_v2(
                naive_optimizer,
                grad_scal_cfg=grad_scal_cfg,
                zero_cfg=zero_cfg,
                param_bcast_sync_handler=param_bcast_sync_handler,
                isp_communicator=isp_communicator,
            )
    else:
        optimizer = FSDPadaptOptimizer(
            naive_optimizer,
            grad_scal_cfg=grad_scal_cfg,
            zero_cfg=zero_cfg,
        )

    beta2_scheduler = Beta2Scheduler(optimizer=naive_optimizer, **gpc.config.beta2_scheduler)

    lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, **gpc.config.lr_scheduler)

    return optimizer, beta2_scheduler, lr_scheduler
