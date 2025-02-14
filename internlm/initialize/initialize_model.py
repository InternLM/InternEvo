from typing import Optional, Union

import torch
from torch import nn

from internlm.core.context import (
    IS_REPLICA_EXPERT_DATA_PARALLEL,
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_EXPERT_DATA_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from internlm.core.context import global_context as gpc
from internlm.core.context import set_mode
from internlm.core.fsdp import wrap_FSDP_model
from internlm.core.naive_amp import (
    NaiveAMPModel,
    set_fp32_attr_to_module,
    unwrap_naive_amp,
)
from internlm.initialize.initialize_communicator import initialize_parallel_communicator
from internlm.model.model_implementations.builder import create_model
from internlm.model.model_implementations.registry import register_model_initializer
from internlm.model.model_ops.modules.embedding import Embedding1D
from internlm.model.model_ops.modules.linear import (
    ParallelLinearWithCommExt,
    ScaleColumnParallelLinear,
)
from internlm.model.model_ops.moe import Experts, MoE
from internlm.model.model_ops.moe.moe import Qwen2MoE
from internlm.model.model_ops.ops.norm import RMSNorm
from internlm.utils.parallel import (
    is_replica_expert_data_parallel_parameter,
    is_replica_zero_parallel_parameter,
    is_tensor_expert_data_parallel_parameter,
    is_tensor_zero_parallel_parameter,
    is_using_fsdp,
    is_using_hf,
    is_using_isp,
    is_weight_expert_data_parallel_parameter,
    is_weight_zero_parallel_parameter,
    sync_model_param,
    sync_model_replica_param_group,
)
from internlm.utils.timeout import llm_timeout


def set_param_unique_tracking_name(model):
    for chunk_id, chunk in enumerate(unwrap_naive_amp(model)):
        # Important: only works for llama-class models
        childrens = chunk.named_children()
        for _, children in childrens:
            if isinstance(children, nn.ModuleList):
                for idx, block in enumerate(children):
                    for name, child in block.named_modules():
                        if isinstance(child, (ParallelLinearWithCommExt)):
                            full_name = f"{chunk_id}.{idx}.{name}"
                            setattr(
                                child.weight,
                                "tracking_name",
                                f"{full_name}.weight",
                            )
                            if child.bias is not None:
                                setattr(
                                    child.bias,
                                    "tracking_name",
                                    f"{full_name}.bias",
                                )
            else:
                if isinstance(children, Embedding1D):
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{chunk_id}_embedding.weight",
                    )
                else:
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{chunk_id}_head.weight",
                    )


def set_fp32_attr_for_model(model: Union[nn.Module, nn.ModuleList]):
    if not isinstance(model, nn.ModuleList):
        model = [model]

    for _chunk in model:
        for _, module in _chunk.named_modules():
            if isinstance(module, (RMSNorm, nn.LayerNorm)) and gpc.config.get("use_fp32_norm", False):
                set_fp32_attr_to_module(module)


def set_parallel_attr_for_param_groups(model: Union[nn.Module, nn.ModuleList]):
    def _check_module(name, module):
        # layer_norm
        if isinstance(module, (RMSNorm, nn.LayerNorm)):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, (MoE, Qwen2MoE)):
            for param in module.moe_layer.gate.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)
            if hasattr(module, "coefficient"):
                for param in module.coefficient.parameters():
                    setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        # embedding and head
        if isinstance(module, (Embedding1D, ScaleColumnParallelLinear)):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)

        # for moe linear module
        if isinstance(module, nn.Linear) and not isinstance(module, ParallelLinearWithCommExt):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, Experts):
            for param in module.parameters():
                if (
                    gpc.is_initialized(ParallelMode.TENSOR)
                    and not is_using_isp()
                    and getattr(gpc.config.parallel.expert, "no_tp", False)
                ):
                    setattr(param, IS_REPLICA_EXPERT_DATA_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_EXPERT_DATA_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_EXPERT_DATA_PARALLEL, True)
        # for non-moe linear module
        elif isinstance(module, ParallelLinearWithCommExt):
            for param in module.parameters():
                if gpc.is_initialized(ParallelMode.TENSOR) and not is_using_isp():
                    setattr(param, IS_TENSOR_ZERO_PARALLEL, True)
                elif gpc.is_initialized(ParallelMode.WEIGHT) and is_using_isp():
                    setattr(param, IS_WEIGHT_ZERO_PARALLEL, True)

        # for vit and vit project
        if "vision_tower" in name.lower() or "vision_proj" in name.lower():
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

    for _chunk in unwrap_naive_amp(model):
        if not is_using_fsdp():
            # set param parallel attribute
            for name, module in _chunk.named_modules():
                _check_module(name, module)

            for name, param in _chunk.named_parameters():
                assert (
                    is_replica_zero_parallel_parameter(param)
                    or is_tensor_zero_parallel_parameter(param)
                    or is_weight_zero_parallel_parameter(param)
                    or is_tensor_expert_data_parallel_parameter(param)
                    or is_weight_expert_data_parallel_parameter(param)
                    or is_replica_expert_data_parallel_parameter(param)
                ), f"parameter with name: {name} has no parallel attribution."


@llm_timeout(func_name="initialize_model_and_parallel_communicator")
def initialize_model_and_parallel_communicator(model: Optional[Union[nn.Module, nn.ModuleList]] = None):
    """
    initialize model with Automatic Mixed Precision.

    Returns:
        torch.nn.Module:
            The neural network model to be trained or evaluated.
        An isp communicator for managing comp/comm overlap.
    """
    if model is None:
        register_model_initializer()
        model = create_model()

    # For non-HF cases, set tracking name for parameters
    if not is_using_hf():
        set_param_unique_tracking_name(model)

    # should be set before NaiveAMPModel
    set_fp32_attr_for_model(model)

    if isinstance(model, nn.ModuleList):
        model = nn.ModuleList(
            [
                NaiveAMPModel(
                    model=_m,
                    output_to_fp32=False,  # manually controlled by interleaved pipleline scheduler
                    dtype=gpc.config.model.get("dtype", torch.half),
                    sync_buffer=False,
                )
                for _m in model
            ]
        )
    else:
        model = NaiveAMPModel(
            model=model,
            output_to_fp32=gpc.is_no_pp_or_last_stage(),
            dtype=gpc.config.model.get("dtype", torch.half),
            sync_buffer=False,
        )

    set_parallel_attr_for_param_groups(model)

    # This sync is very important, cause the model weights kept in optimizer are copied
    # from the origin parameters in the memory, so we should make sure the dp sync
    # does not influence the model weights in optimizer be different with the origin parameters.
    if not is_using_fsdp() or gpc.config.parallel.fsdp.get("init_method", "cuda") == "cuda":
        sync_model_param(model)

    # This function is needed to make sure parameters that are not splitted by tensor parallelism are
    # the same across tensor parallelism.
    sync_model_replica_param_group(model)

    # Change random state mode to ParallelMode.DATA after model is built, guaranteeing the random
    # state in the same dp group are all the same.
    random_mode = ParallelMode.WEIGHT_DATA if is_using_isp() else ParallelMode.DATA
    set_mode(random_mode)

    # initialize isp communicator
    isp_communicator = initialize_parallel_communicator(model)

    model = wrap_FSDP_model(model)

    return model, isp_communicator
