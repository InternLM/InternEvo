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
    dump_quantize=True,  # only for int8 training
    dump_activation=True,
    dump_weight=True,
    dump_gradient=True,
    dump_activation_gradient=True,
    dump_activation_path=f"llm_profiling/{JOB_NAME}/activation",
    dump_weight_path=f"llm_profiling/{JOB_NAME}/weight",
    dump_gradient_path=f"llm_profiling/{JOB_NAME}/gradient",
    dump_activation_gradient_path=f"llm_profiling/{JOB_NAME}/activation_gradient",
    dump_quantize_path=f"llm_profiling/{JOB_NAME}/quantize",
    step_internval=5,
    layer_interval=2,
)

"""
import functools
import json
import os
from typing import Tuple

import torch
from torch import nn

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
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


def quick_quantiles(x: torch.Tensor, qs: list):
    """
    Calculate multiple quantiles of a tensor using one sorting step.
    """
    if x is None or len(qs) == 0:
        return {}

    # Ensure the tensor is flattened for consistent quantile calculation
    n = x.numel()
    x_flat = x.flatten()

    # Sort the tensor once
    sorted_x, _ = torch.sort(x_flat)

    # Calculate the indices for each quantile
    indices = [int(q * n) for q in qs]

    # Extract the quantiles using the pre-sorted tensor
    # quantiles = {q: sorted_x[i].item() for q, i in zip(qs, indices)}
    quantiles = torch.tensor([sorted_x[i] for i in indices])

    return quantiles


def get_statistics(x: torch.Tensor):
    """
    Get the statistics of the tensor.
    """
    if x is None:
        return {}

    data_type = x.dtype
    x_flatten = x.flatten()
    if data_type != torch.float32:
        x_float = x_flatten.float()  # Convert to float for calculations to avoid overflow/underflow with some dtypes

    base_list = torch.tensor(
        [x_float.mean(), x_float.std(), x_float.min(), x_float.max(), x_float.abs().mean(), x_float.abs().max()]
    ).tolist()
    base_stats = {
        "mean": base_list[0],
        "std": base_list[1],
        "min": base_list[2],
        "max": base_list[3],
        "abs_mean": base_list[4],
        "abs_max": base_list[5],
    }

    numel = x.numel()
    shape_stats = {
        "numel": numel,
        "shape": list(x.shape),
        "dtype": str(data_type),
    }

    quantiles_list = quick_quantiles(x_float, [0.01, 0.05, 0.50, 0.95, 0.99])  # Pre-calculate quantiles for efficiency
    quantiles_list = quantiles_list.tolist()
    quantile_stats = {
        "p01": quantiles_list[0],
        "p05": quantiles_list[1],
        # "p10": quick_quantile(x_float, 0.10).item(),
        # "p25": quick_quantile(x_float, 0.25).item(),
        "p50": quantiles_list[2],  # median
        # "p75": quick_quantile(x_float, 0.75).item(),
        # "p90": quick_quantile(x_float, 0.90).item(),
        "p95": quantiles_list[3],
        "p99": quantiles_list[4],
    }

    outlier_mask = (x_float < quantile_stats["p01"]) | (x_float > quantile_stats["p99"])
    nan_inf_mask = torch.isnan(x_float) | torch.isinf(x_float)
    zero_mask = x_float == 0
    enhanced_counts = (
        torch.tensor([outlier_mask.sum(), nan_inf_mask.sum(), zero_mask.sum()], dtype=torch.float32) / numel
    )
    enhanced_counts = enhanced_counts.tolist()

    # Calculate enhanced statistics
    enhanced_stats = {
        "l2_norm": x_float.norm(p=2).item(),  # L2 norm,
        "outlier_ratio": enhanced_counts[0],
        "nan_inf_ratio": enhanced_counts[1],
        "zero_ratio": enhanced_counts[2],
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


def dump_quantize_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper function to dump quantized tensor statistics.
        """
        if gpc.config.get("dump_profiling", None) is None:
            return func(*args, **kwargs)

        cur_step = gpc.config.batch_count
        dump_cfg = gpc.config.dump_profiling
        step_internval = dump_cfg.get("step_internval", 1)
        if is_using_isp():
            should_profiling = (
                gpc.get_local_rank(ParallelMode.WEIGHT_DATA) == 0 and cur_step % step_internval == 0
            )  # only dp 0
        else:
            should_profiling = (
                gpc.get_local_rank(ParallelMode.DATA) == 0 and cur_step % step_internval == 0
            )  # only dp 0

        if not should_profiling:
            return func(*args, **kwargs)
        if not dump_cfg.get("dump_quantize", False):
            return func(*args, **kwargs)

        # Call the original function
        output = func(*args, **kwargs)

        a_global_name = getattr(args[0], "global_name", None)
        b_global_name = getattr(args[1], "global_name", None)
        quantize_a, quantize_b = output[0], output[1]

        dump_a, dump_b = False, False
        if a_global_name is not None and a_global_name.endswith(".input"):
            dump_a = True
            if b_global_name is not None and b_global_name.endswith(".weight"):
                dump_b = True
        if a_global_name is not None and a_global_name.endswith(".grad_output"):
            dump_a = True

        if not dump_a:
            return output

        layerid = get_layerid_from_name(a_global_name)
        layer_internval = dump_cfg.get("layer_interval", 1)
        if gpc.precision_profiling.get(cur_step, None) is None:
            gpc.precision_profiling[cur_step] = {}
        if layerid == -1:
            logger.warning(f"(precision profiling): Layer id not found for {a_global_name}")
        if (layerid + 1) % layer_internval == 0 or layerid == 0 or layerid == gpc.config.model.num_layers - 1:
            if gpc.precision_profiling[cur_step].get(layerid, None) is None:
                gpc.precision_profiling[cur_step][layerid] = {
                    "quantize_input": {},
                    "quantize_gradoutput": {},
                    "quantize_weight": {},
                }
            else:
                if gpc.precision_profiling[cur_step][layerid].get("quantize_input", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["quantize_input"] = {}
                if gpc.precision_profiling[cur_step][layerid].get("quantize_gradoutput", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["quantize_gradoutput"] = {}
                if gpc.precision_profiling[cur_step][layerid].get("quantize_weight", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["quantize_weight"] = {}

            curlayer_profiling_data = gpc.precision_profiling[cur_step][layerid]
            if a_global_name.endswith(".input"):
                if a_global_name not in curlayer_profiling_data["quantize_input"]:
                    curlayer_profiling_data["quantize_input"][a_global_name] = []
                if dump_cfg.save_statistics:
                    curlayer_profiling_data["quantize_input"][a_global_name].append(get_statistics(quantize_a))
                else:
                    # 先转移至cpu，然后保存，no blocking
                    curlayer_profiling_data["quantize_input"][a_global_name].append(
                        quantize_a.to(torch.device("cpu"), non_blocking=True)
                    )
            elif a_global_name.endswith(".grad_output"):
                if a_global_name not in curlayer_profiling_data["quantize_gradoutput"]:
                    curlayer_profiling_data["quantize_gradoutput"][a_global_name] = []
                if dump_cfg.save_statistics:
                    curlayer_profiling_data["quantize_gradoutput"][a_global_name].append(get_statistics(quantize_a))
                else:
                    # 先转移至cpu，然后保存，no blocking
                    curlayer_profiling_data["quantize_gradoutput"][a_global_name].append(
                        quantize_a.to(torch.device("cpu"), non_blocking=True)
                    )

            if dump_b:
                if b_global_name not in curlayer_profiling_data["quantize_weight"]:
                    curlayer_profiling_data["quantize_weight"][b_global_name] = []
                if dump_cfg.save_statistics:
                    curlayer_profiling_data["quantize_weight"][b_global_name].append(get_statistics(quantize_b))
                else:
                    # 先转移至cpu，然后保存，no blocking
                    curlayer_profiling_data["quantize_weight"][b_global_name].append(
                        quantize_b.to(torch.device("cpu"), non_blocking=True)
                    )

        return output

    return wrapper


def dump_activation_and_weight(model: nn.Module, input, output: torch.Tensor):  # pylint: disable=W0613
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
        input_name = weight_name.replace(".weight", ".input")
        output_name = weight_name.replace(".weight", ".output")
        layer_internval = dump_cfg.get("layer_interval", 1)
        layerid = get_layerid_from_name(weight_name)

        if gpc.precision_profiling.get(cur_step, None) is None:
            gpc.precision_profiling[cur_step] = {}
        if layerid == -1:
            logger.warning(f"(precision profiling): Layer id not found for {weight_name}")
            return
        if (layerid + 1) % layer_internval == 0 or layerid == 0 or layerid == gpc.config.model.num_layers - 1:
            if gpc.precision_profiling[cur_step].get(layerid, None) is None:
                gpc.precision_profiling[cur_step][layerid] = {"input": {}, "output": {}, "weight": {}}
            else:
                if gpc.precision_profiling[cur_step][layerid].get("input", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["input"] = {}
                if gpc.precision_profiling[cur_step][layerid].get("output", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["output"] = {}
                if gpc.precision_profiling[cur_step][layerid].get("weight", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["weight"] = {}
            curlayer_profiling_data = gpc.precision_profiling[cur_step][layerid]

            if dump_cfg.dump_activation:
                if input_name not in curlayer_profiling_data["input"]:
                    curlayer_profiling_data["input"][input_name] = []
                if output_name not in curlayer_profiling_data["output"]:
                    curlayer_profiling_data["output"][output_name] = []
            if dump_cfg.dump_weight:
                if weight_name not in curlayer_profiling_data["weight"]:
                    curlayer_profiling_data["weight"][weight_name] = []

            if dump_cfg.save_statistics:
                if dump_cfg.dump_activation:
                    curlayer_profiling_data["input"][input_name].append(get_statistics(input[0]))
                    curlayer_profiling_data["output"][output_name].append(get_statistics(output))
                if dump_cfg.dump_weight:
                    curlayer_profiling_data["weight"][weight_name].append(get_statistics(model.weight))
            else:
                # 先转移至cpu，然后保存，no blocking
                if dump_cfg.dump_activation:
                    curlayer_profiling_data["input"][input_name].append(
                        input[0].to(torch.device("cpu"), non_blocking=True)
                    )
                    curlayer_profiling_data["output"][output_name].append(
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
        grad_input_name = weight_name.replace(".weight", ".grad_input")
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
                gpc.precision_profiling[cur_step][layerid] = {"grad": {}, "grad_input": {}, "grad_output": {}}
            else:
                if gpc.precision_profiling[cur_step][layerid].get("grad", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["grad"] = {}
                if gpc.precision_profiling[cur_step][layerid].get("grad_input", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["grad_input"] = {}
                if gpc.precision_profiling[cur_step][layerid].get("grad_output", None) is None:
                    gpc.precision_profiling[cur_step][layerid]["grad_output"] = {}
            curlayer_profiling_data = gpc.precision_profiling[cur_step][layerid]

            if dump_cfg.dump_gradient:
                if grad_name not in curlayer_profiling_data["grad"]:
                    curlayer_profiling_data["grad"][grad_name] = []
            if dump_cfg.dump_activation_gradient:
                if grad_input_name not in curlayer_profiling_data["grad_input"]:
                    curlayer_profiling_data["grad_input"][grad_input_name] = []
                if grad_output_name not in curlayer_profiling_data["grad_output"]:
                    curlayer_profiling_data["grad_output"][grad_output_name] = []

            if dump_cfg.save_statistics:
                if dump_cfg.dump_gradient:
                    curlayer_profiling_data["grad"][grad_name].append(get_statistics(model.weight.grad))
                if dump_cfg.dump_activation_gradient:
                    curlayer_profiling_data["grad_input"][grad_input_name].append(get_statistics(grad_input[0]))
                    curlayer_profiling_data["grad_output"][grad_output_name].append(get_statistics(grad_output[0]))
            else:
                # 先转移至cpu，然后保存，no blocking
                if dump_cfg.dump_gradient:
                    curlayer_profiling_data["grad"][grad_name].append(
                        model.weight.grad.to(torch.device("cpu"), non_blocking=True)
                    )
                if dump_cfg.dump_activation_gradient:
                    curlayer_profiling_data["grad_input"][grad_input_name].append(
                        grad_input[0].to(torch.device("cpu"), non_blocking=True)
                    )
                    curlayer_profiling_data["grad_output"][grad_output_name].append(
                        grad_output[0].to(torch.device("cpu"), non_blocking=True)
                    )


def register_dump_hooks(model: torch.nn.Module):
    """
    Register hooks to dump activation, weight and gradient.
    """
    from internlm.core.naive_amp import unwrap_naive_amp
    from internlm.model.modules.linear import ParallelLinearWithCommExt
    from internlm.model.ops.norm import RMSNorm

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
    os.makedirs(dump_cfg.get("dump_quantize_path", "llm_profiling/quantize"), exist_ok=True)

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
                        # elif isinstance(child, (RMSNorm)):
                        #     setattr(
                        #         child.weight,
                        #         "global_name",
                        #         f"{global_fqn}",
                        #     )
                        #     if should_register_forward:
                        #         child.register_forward_hook(dump_activation_and_weight)
                        #     if should_register_backward:
                        #         child.register_full_backward_hook(dump_gradient)
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
    dump_quantize_path = dump_cfg.get("dump_quantize_path", "llm_profiling/quantize")
    dump_activation_path = os.path.join(dump_activation_path, f"step_{cur_step}")
    dump_gradient_path = os.path.join(dump_gradient_path, f"step_{cur_step}")
    dump_weight_path = os.path.join(dump_weight_path, f"step_{cur_step}")
    dump_activation_gradient_path = os.path.join(dump_activation_gradient_path, f"step_{cur_step}")
    dump_quantize_path = os.path.join(dump_quantize_path, f"step_{cur_step}")

    # 确保所有dump的tensor已经copy到cpu
    if not dump_cfg.save_statistics:
        internlm_accelerator.synchronize()

    if dump_cfg.dump_activation:
        os.makedirs(dump_activation_path, exist_ok=True)
        for layerid, layer_profiling_data in gpc.precision_profiling[cur_step].items():
            input_profiling_data = layer_profiling_data["input"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_activation_path, f"layer{layerid}_input_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(input_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_activation_path, f"layer{layerid}_input_{suffix}.pt")
                torch.save(input_profiling_data, save_path)
            output_profiling_data = layer_profiling_data["output"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_activation_path, f"layer{layerid}_output_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(output_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_activation_path, f"layer{layerid}_output_{suffix}.pt")
                torch.save(output_profiling_data, save_path)

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
            grad_input_profiling_data = layer_profiling_data["grad_input"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_activation_gradient_path, f"layer{layerid}_gradinput_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(grad_input_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_activation_gradient_path, f"layer{layerid}_gradinput_{suffix}.pt")
                torch.save(grad_input_profiling_data, save_path)
            grad_output_profiling_data = layer_profiling_data["grad_output"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_activation_gradient_path, f"layer{layerid}_gradoutput_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(grad_output_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_activation_gradient_path, f"layer{layerid}_gradoutput_{suffix}.pt")
                torch.save(grad_output_profiling_data, save_path)

    if dump_cfg.dump_quantize:
        os.makedirs(dump_quantize_path, exist_ok=True)
        for layerid, layer_profiling_data in gpc.precision_profiling[cur_step].items():
            quantize_input_profiling_data = layer_profiling_data["quantize_input"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_quantize_path, f"layer{layerid}_quantize_input_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(quantize_input_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_quantize_path, f"layer{layerid}_quantize_input_{suffix}.pt")
                torch.save(quantize_input_profiling_data, save_path)

            quantize_gradoutput_profiling_data = layer_profiling_data["quantize_gradoutput"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_quantize_path, f"layer{layerid}_quantize_gradoutput_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(quantize_gradoutput_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_quantize_path, f"layer{layerid}_quantize_gradoutput_{suffix}.pt")
                torch.save(quantize_gradoutput_profiling_data, save_path)

            quantize_weight_profiling_data = layer_profiling_data["quantize_weight"]
            if dump_cfg.save_statistics:
                save_path = os.path.join(dump_quantize_path, f"layer{layerid}_quantize_weight_{suffix}.json")
                with open(save_path, "w") as f:
                    json.dump(quantize_weight_profiling_data, f, indent=2)
            else:
                save_path = os.path.join(dump_quantize_path, f"layer{layerid}_quantize_weight_{suffix}.pt")
                torch.save(quantize_weight_profiling_data, save_path)

    # clear profiling
    gpc.precision_profiling = {}
