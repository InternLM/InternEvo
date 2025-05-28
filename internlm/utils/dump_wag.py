"""
Tensor Statistics Collection Utility (WAG - Weights, Activations, Gradients)

This module provides functionality to:
- Collect and analyze tensor statistics during model training
- Monitor weights, activations, and gradients at specified layers/steps
- Support distributed training with various parallelism strategies
- Efficiently manage memory with non-blocking transfers
- Save data in statistical summary or raw tensor format

Usage: Register hooks via register_dump_hooks() and configure sampling
frequency in your training configuration.

config example:
JOB_NAME = "your job name"
dump_profiling = dict(
    save_statistics=True,
    dump_activation=True,
    dump_weight=True,
    dump_gradient=True,
    dump_activation_gradient=True,
    dump_activation_path=f"llm_profiling/{JOB_NAME}/activation",
    dump_weight_path=f"llm_profiling/{JOB_NAME}/weight",
    dump_gradient_path=f"llm_profiling/{JOB_NAME}/gradient",
    dump_activation_gradient_path=f"llm_profiling/{JOB_NAME}/activation_gradient",
    step_internval=5,
    layer_interval=4,
)

"""
import json
import os
from typing import Tuple

import torch
from torch import nn

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.naive_amp import unwrap_naive_amp
from internlm.model.modules.linear import ParallelLinearWithCommExt
from internlm.model.ops.norm import RMSNorm
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_isp

internlm_accelerator = get_accelerator()
logger = get_logger(__file__)


def quick_quantile(x: torch.Tensor, q: float):
    """
    Calculate the quantile of a tensor.
    """
    if x is None:
        return None

    n = x.numel()
    k = int(q * n)
    x_q, _ = torch.kthvalue(x, k)
    return x_q


def get_statistics(x: torch.Tensor):
    """
    Get the statistics of the tensor.
    """
    if x is None:
        return {}

    data_type = x.dtype
    x_float = x.flatten().float()  # Convert to float for calculations to avoid overflow/underflow with some dtypes

    base_stats = {
        "mean": x_float.mean().item(),
        "std": x_float.std().item(),
        "min": x_float.min().item(),
        "max": x_float.max().item(),
        "abs_mean": x_float.abs().mean().item(),
        "abs_max": x_float.abs().max().item(),
    }

    shape_stats = {
        "numel": x.numel(),
        "shape": list(x.shape),
        "dtype": str(data_type),
    }

    quantile_stats = {
        "p01": quick_quantile(x_float, 0.01).item(),
        "p05": quick_quantile(x_float, 0.05).item(),
        # "p10": quick_quantile(x_float, 0.10).item(),
        # "p25": quick_quantile(x_float, 0.25).item(),
        "p50": quick_quantile(x_float, 0.50).item(),  # median
        # "p75": quick_quantile(x_float, 0.75).item(),
        # "p90": quick_quantile(x_float, 0.90).item(),
        "p95": quick_quantile(x_float, 0.95).item(),
        "p99": quick_quantile(x_float, 0.99).item(),
    }

    enhanced_stats = {
        "l2_norm": x_float.norm(p=2).item(),
        "outlier_ratio": ((x_float < quantile_stats["p01"]) | (x_float > quantile_stats["p99"])).float().mean().item(),
        "nan_inf_ratio": (torch.isnan(x_float) | torch.isinf(x_float)).float().mean().item(),
        "zero_ratio": (x_float == 0).float().mean().item(),
    }

    return {
        **base_stats,
        **quantile_stats,
        **enhanced_stats,
        **shape_stats,
    }


def get_layerid_from_name(name: str):
    layerid = -1
    if name.startswith("layers."):
        layerid = int(name.split(".")[1])
    elif name.startswith("output.") or name.startswith("norm."):
        layerid = gpc.config.model.num_layers - 1
    elif name.startswith("tok_embeddings."):
        layerid = 0

    return layerid


def dump_activation_and_weight(model: nn.Module, args, output: torch.Tensor):  # pylint: disable=W0613
    """
    Dumps the activation and weight of the model.
    """

    cur_step = gpc.config.batch_count
    dump_cfg = gpc.config.dump_profiling
    step_internval = dump_cfg.get("step_internval", 1)
    if is_using_isp():
        should_profiling = (
            gpc.get_local_rank(ParallelMode.WEIGHT_DATA) == 0 and cur_step % step_internval == 0
        )  # only dp 0
    else:
        should_profiling = gpc.get_local_rank(ParallelMode.DATA) == 0 and cur_step % step_internval == 0  # only dp 0

    if should_profiling:
        weight_name = getattr(model.weight, "global_name", None)
        activation_name = weight_name.replace(".weight", ".activation")
        layer_internval = dump_cfg.get("layer_interval", 1)
        layerid = get_layerid_from_name(weight_name)

        if gpc.precision_profiling.get(cur_step, None) is None:
            gpc.precision_profiling[cur_step] = {}
        if layerid == -1:
            logger.warning(f"(precision profiling): Layer id not found for {weight_name}")
            return
        if (layerid + 1) % layer_internval == 0 or layerid == 0 or layerid == gpc.config.model.num_layers - 1:
            if gpc.precision_profiling[cur_step].get(layerid, None) is None:
                gpc.precision_profiling[cur_step][layerid] = {"activation": {}, "weight": {}}
            curlayer_profiling_data = gpc.precision_profiling[cur_step][layerid]

            if dump_cfg.dump_activation:
                if activation_name not in curlayer_profiling_data["activation"]:
                    curlayer_profiling_data["activation"][activation_name] = []
            if dump_cfg.dump_weight:
                if weight_name not in curlayer_profiling_data["weight"]:
                    curlayer_profiling_data["weight"][weight_name] = []

            if dump_cfg.save_statistics:
                if dump_cfg.dump_activation:
                    curlayer_profiling_data["activation"][activation_name].append(get_statistics(output))
                if dump_cfg.dump_weight:
                    curlayer_profiling_data["weight"][weight_name].append(get_statistics(model.weight))
            else:
                # 先转移至cpu，然后保存，no blocking
                if dump_cfg.dump_activation:
                    curlayer_profiling_data["activation"][activation_name].append(
                        output.to(torch.device("cpu"), non_blocking=True)
                    )
                if dump_cfg.dump_weight:
                    curlayer_profiling_data["weight"][weight_name].append(
                        model.weight.to(torch.device("cpu"), non_blocking=True)
                    )


def dump_gradient(
    model: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]
):  # pylint: disable=W0613
    """
    Dumps the gradient of the model.
    """

    cur_step = gpc.config.batch_count
    dump_cfg = gpc.config.dump_profiling
    step_internval = dump_cfg.get("step_internval", 1)
    if is_using_isp():
        should_profiling = (
            gpc.get_local_rank(ParallelMode.WEIGHT_DATA) == 0 and cur_step % step_internval == 0
        )  # only dp 0
    else:
        should_profiling = gpc.get_local_rank(ParallelMode.DATA) == 0 and cur_step % step_internval == 0  # only dp 0

    if should_profiling:
        weight_name = getattr(model.weight, "global_name", None)
        grad_name = weight_name.replace(".weight", ".grad")
        grad_output_name = weight_name.replace(".weight", ".grad_output")
        layer_internval = dump_cfg.get("layer_interval", 1)
        layerid = get_layerid_from_name(weight_name)

        if gpc.precision_profiling.get(cur_step, None) is None:
            gpc.precision_profiling[cur_step] = {}
        if layerid == -1:
            logger.warning(f"(precision profiling): Layer id not found for {weight_name}")
            return
        if (layerid + 1) % layer_internval == 0 or layerid == 0 or layerid == gpc.config.model.num_layers - 1:
            if gpc.precision_profiling[cur_step].get(layerid, None) is None:
                gpc.precision_profiling[cur_step][layerid] = {"grad": {}, "grad_output": {}}
            else:
                if gpc.precision_profiling[cur_step][layerid].get("grad", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["grad"] = {}
                if gpc.precision_profiling[cur_step][layerid].get("grad_output", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["grad_output"] = {}
            curlayer_profiling_data = gpc.precision_profiling[cur_step][layerid]

            if dump_cfg.dump_gradient:
                if grad_name not in curlayer_profiling_data["grad"]:
                    curlayer_profiling_data["grad"][grad_name] = []
            if dump_cfg.dump_activation_gradient:
                if grad_output_name not in curlayer_profiling_data["grad_output"]:
                    curlayer_profiling_data["grad_output"][grad_output_name] = []

            if dump_cfg.save_statistics:
                if dump_cfg.dump_gradient:
                    curlayer_profiling_data["grad"][grad_name].append(get_statistics(model.weight.grad))
                if dump_cfg.dump_activation_gradient:
                    curlayer_profiling_data["grad_output"][grad_output_name].append(get_statistics(grad_output[0]))
            else:
                # 先转移至cpu，然后保存，no blocking
                if dump_cfg.dump_gradient:
                    curlayer_profiling_data["grad"][grad_name].append(
                        model.weight.grad.to(torch.device("cpu"), non_blocking=True)
                    )
                if dump_cfg.dump_activation_gradient:
                    curlayer_profiling_data["grad_output"][grad_output_name].append(
                        grad_output[0].to(torch.device("cpu"), non_blocking=True)
                    )


def register_dump_hooks(model: torch.nn.Module):
    """
    Register hooks to dump activation, weight and gradient.
    """
    # config
    if gpc.config.get("dump_profiling", None) is None:
        return
    setattr(gpc, "precision_profiling", {})

    should_register_forward = False
    should_register_backward = False
    dump_cfg = gpc.config.dump_profiling
    if dump_cfg.get("dump_activation", False) or dump_cfg.get("dump_weight", False):
        should_register_forward = True

    if dump_cfg.get("dump_gradient", False) or dump_cfg.get("dump_activation_gradient", False):
        should_register_backward = True

    os.makedirs(dump_cfg.get("dump_activation_path", "llm_profiling/activation"), exist_ok=True)
    os.makedirs(dump_cfg.get("dump_gradient_path", "llm_profiling/gradient"), exist_ok=True)
    os.makedirs(dump_cfg.get("dump_weight_path", "llm_profiling/weight"), exist_ok=True)
    os.makedirs(dump_cfg.get("dump_activation_gradient_path", "llm_profiling/activation_gradient"), exist_ok=True)

    for chunk_id, chunk in enumerate(unwrap_naive_amp(model)):
        # Important: only works for llama-class models
        childrens = chunk.named_children()
        for children_name, children in childrens:
            if isinstance(children, nn.ModuleList):
                for idx, block in enumerate(children):
                    for name, child in block.named_modules():
                        if name == "":
                            continue

                        full_name = f"{chunk_id}.{idx}.{name}"
                        name_parts = f"{full_name}.weight".split(".", 2)
                        # global_id for pipeline parallel case
                        global_id = model.first_layer + idx
                        global_fqn = f"{children_name}.{global_id}." + ".".join(name_parts[2:])

                        if isinstance(child, (ParallelLinearWithCommExt)):
                            setattr(
                                child.weight,
                                "global_name",
                                f"{global_fqn}",
                            )
                            if child.bias is not None:
                                setattr(
                                    child.bias,
                                    "global_name",
                                    f"{global_fqn.replace('weight', 'bias')}",
                                )
                            if should_register_forward:
                                child.register_forward_hook(dump_activation_and_weight)
                            if should_register_backward:
                                child.register_full_backward_hook(dump_gradient)
                        elif isinstance(child, (RMSNorm)):
                            setattr(
                                child.weight,
                                "global_name",
                                f"{global_fqn}",
                            )
                            if should_register_forward:
                                child.register_forward_hook(dump_activation_and_weight)
                            if should_register_backward:
                                child.register_full_backward_hook(dump_gradient)
            # else:
            #     full_name = f"{chunk_id}.{children_name}"
            #     local_fqn = f"{children_name}.weight"
            #     setattr(
            #         children.weight,
            #         "global_name",
            #         f"{local_fqn}",
            #     )
            #     if getattr(children, "bias", None) is not None:
            #         if children.bias is not None:
            #             setattr(
            #                 children.bias,
            #                 "global_name",
            #                 f"{local_fqn.replace('weight', 'bias')}",
            #             )
            #     if should_register_forward:
            #         children.register_forward_hook(dump_activation_and_weight)
            #     if should_register_backward:
            #         children.register_full_backward_hook(dump_gradient)


def save_profiling():
    if getattr(gpc, "precision_profiling", None) is None:
        return

    # 判断是否为空字典
    if gpc.precision_profiling == {}:
        return

    if is_using_isp():
        suffix = f"wdp0_wp{gpc.get_local_rank(ParallelMode.WEIGHT)}_pp{gpc.get_local_rank(ParallelMode.PIPELINE)}"
    else:
        suffix = f"dp0_tp{gpc.get_local_rank(ParallelMode.TENSOR)}_pp{gpc.get_local_rank(ParallelMode.PIPELINE)}"
    cur_step = gpc.config.batch_count

    assert cur_step in gpc.precision_profiling, f"cur_step {cur_step} not in gpc.precision_profiling"

    dump_cfg = gpc.config.dump_profiling
    dump_activation_path = dump_cfg.get("dump_activation_path", "llm_profiling/activation")
    dump_gradient_path = dump_cfg.get("dump_gradient_path", "llm_profiling/gradient")
    dump_weight_path = dump_cfg.get("dump_weight_path", "llm_profiling/weight")
    dump_activation_gradient_path = dump_cfg.get("dump_activation_gradient_path", "llm_profiling/activation_gradient")
    dump_activation_path = os.path.join(dump_activation_path, f"step_{cur_step}")
    dump_gradient_path = os.path.join(dump_gradient_path, f"step_{cur_step}")
    dump_weight_path = os.path.join(dump_weight_path, f"step_{cur_step}")
    dump_activation_gradient_path = os.path.join(dump_activation_gradient_path, f"step_{cur_step}")

    # 确保所有dump的tensor已经copy到cpu
    if not dump_cfg.save_statistics:
        internlm_accelerator.synchronize()

    if dump_cfg.dump_activation:
        os.makedirs(dump_activation_path, exist_ok=True)
        for layerid, layer_profiling_data in gpc.precision_profiling[cur_step].items():
            act_profiling_data = layer_profiling_data["activation"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_activation_path, f"layer{layerid}_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(act_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_activation_path, f"layer{layerid}_{suffix}.pt")
                torch.save(act_profiling_data, save_path)

    if dump_cfg.dump_gradient:
        os.makedirs(dump_gradient_path, exist_ok=True)
        for layerid, layer_profiling_data in gpc.precision_profiling[cur_step].items():
            grad_profiling_data = layer_profiling_data["grad"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_gradient_path, f"layer{layerid}_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(grad_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_gradient_path, f"layer{layerid}_{suffix}.pt")
                torch.save(grad_profiling_data, save_path)

    if dump_cfg.dump_weight:
        os.makedirs(dump_weight_path, exist_ok=True)
        for layerid, layer_profiling_data in gpc.precision_profiling[cur_step].items():
            weight_profiling_data = layer_profiling_data["weight"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_weight_path, f"layer{layerid}_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(weight_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_weight_path, f"layer{layerid}_{suffix}.pt")
                torch.save(weight_profiling_data, save_path)

    if dump_cfg.dump_activation_gradient:
        os.makedirs(dump_activation_gradient_path, exist_ok=True)
        for layerid, layer_profiling_data in gpc.precision_profiling[cur_step].items():
            grad_output_profiling_data = layer_profiling_data["grad_output"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_activation_gradient_path, f"layer{layerid}_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(grad_output_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_activation_gradient_path, f"layer{layerid}_{suffix}.pt")
                torch.save(grad_output_profiling_data, save_path)

    # clear profiling
    gpc.precision_profiling = {}
