import os
from typing import Optional, Union

import torch
import torch.distributed as dist
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
from internlm.model.model_ops.ops.norm import RMSNorm
from internlm.utils.logger import get_logger
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

logger = get_logger(__file__)


# For universal checkpoint
# record offset and complete_size of param in each layer
map_layer_attr = {}
map_fqn_local_to_global = {}
map_fqn_global_to_local = {}


def set_param_unique_tracking_name(model):
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
                        local_fqn = f"{children_name}." + ".".join(name_parts[1:])
                        global_fqn = f"{children_name}.{global_id}." + ".".join(name_parts[2:])

                        if isinstance(child, (ParallelLinearWithCommExt)):
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

                            setattr(
                                child.weight,
                                "fqn",
                                f"{local_fqn}",
                            )
                            if child.bias is not None:
                                setattr(
                                    child.bias,
                                    "fqn",
                                    f"{local_fqn}",
                                )

                            assert hasattr(child, "offset"), f"{child}"
                            map_fqn_local_to_global[local_fqn] = global_fqn
                            map_fqn_global_to_local[global_fqn] = local_fqn

                            assert global_fqn not in map_layer_attr, f"{map_layer_attr} exists"
                            map_layer_attr[global_fqn] = {
                                "offset": getattr(child, "offset", [0] * len(child.weight.size())),
                                "complete_size": getattr(child, "complete_size", list(child.weight.size())),
                            }

                        elif isinstance(child, (RMSNorm)):
                            map_fqn_local_to_global[local_fqn] = global_fqn
                            map_fqn_global_to_local[global_fqn] = local_fqn
                            setattr(
                                child.weight,
                                "fqn",
                                f"{local_fqn}",
                            )
                            map_layer_attr[global_fqn] = {
                                "offset": getattr(child, "offset", [0] * len(child.weight.size())),
                                "complete_size": getattr(child, "complete_size", list(child.weight.size())),
                            }

            else:
                full_name = f"{chunk_id}.{children_name}"
                local_fqn = f"{children_name}.weight"
                assert getattr(children, "bias", None) is None
                if isinstance(children, Embedding1D):
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{chunk_id}_embeddings.weight",
                    )
                    assert local_fqn not in map_layer_attr, f"{map_layer_attr} exists"
                else:
                    setattr(
                        children.weight,
                        "tracking_name",
                        f"{full_name}.weight",
                    )
                    assert local_fqn not in map_layer_attr, f"{map_layer_attr} exists"

                setattr(
                    children.weight,
                    "fqn",
                    f"{local_fqn}",
                )
                if getattr(children, "bias", None) is not None:
                    if children.bias is not None:
                        setattr(
                            children.bias,
                            "fqn",
                            f"{local_fqn}",
                        )

                map_layer_attr[local_fqn] = {
                    "offset": getattr(children, "offset", [0] * len(children.weight.size())),
                    "complete_size": getattr(children, "complete_size", list(children.weight.size())),
                }


def generate_meta_data(optimizer):
    if not gpc.config.ckpt.need_metadata:
        return

    if gpc.get_world_size(ParallelMode.PIPELINE) > 1:
        assert optimizer.meta_for_zero is not None
        dst = gpc.get_ranks_in_group(ParallelMode.PIPELINE)[0]
        if gpc.get_global_rank() == dst:
            output = [None for _ in range(gpc.get_world_size(ParallelMode.PIPELINE))]
        else:
            output = None

        dist.gather_object(optimizer.meta_for_zero, output, dst=dst, group=gpc.get_group(ParallelMode.PIPELINE))
        pp_gather_output = output

    else:
        pp_gather_output = [optimizer.meta_for_zero]

    tp_parallel = ParallelMode.WEIGHT if is_using_isp() else ParallelMode.TENSOR
    if gpc.get_world_size(tp_parallel) > 1:
        dst = gpc.get_ranks_in_group(tp_parallel)[0]
        if gpc.get_global_rank() == dst:
            output = [None for _ in range(gpc.get_world_size(tp_parallel))]
        else:
            output = None

        dist.gather_object(pp_gather_output, output, dst=dst, group=gpc.get_group(tp_parallel))
        final_output = output
    else:
        final_output = [pp_gather_output]

    if gpc.get_global_rank() == 0:
        assert len(final_output) == gpc.get_world_size(tp_parallel)
        assert len(final_output[0]) == gpc.get_world_size(ParallelMode.PIPELINE)
        assert len(final_output[0][0]) == gpc.get_world_size(ParallelMode.ZERO1)
        tp_mode = "wp_size" if is_using_isp() else "tp_size"
        final_meta = {
            "parallel_setting": {
                tp_mode: gpc.get_world_size(tp_parallel),
                "pp_size": gpc.get_world_size(ParallelMode.PIPELINE),
                "zero1_size": gpc.get_world_size(ParallelMode.ZERO1),
            },
            "metaData": final_output,
        }

        if gpc.config.ckpt.generate_meta_data.enable:
            save_path = os.path.join(gpc.config.ckpt.generate_meta_data.path, "metadata.pt")
            torch.save(final_meta, save_path)
            logger.info(f"Successfully generate metadata.pt in {gpc.config.ckpt.generate_meta_data.path}")

        return final_meta
    return None


def set_fp32_attr_for_model(model: Union[nn.Module, nn.ModuleList]):
    if not isinstance(model, nn.ModuleList):
        model = [model]

    for _chunk in model:
        for _, module in _chunk.named_modules():
            if isinstance(module, (RMSNorm, nn.LayerNorm)) and gpc.config.get("use_fp32_norm", False):
                set_fp32_attr_to_module(module)


def set_parallel_attr_for_param_groups(model: Union[nn.Module, nn.ModuleList]):
    def _check_module(module):
        # layer_norm
        if isinstance(module, (RMSNorm, nn.LayerNorm)):
            for param in module.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

        if isinstance(module, MoE):
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

    for _chunk in unwrap_naive_amp(model):
        if not is_using_fsdp():
            # set param parallel attribute
            for _, module in _chunk.named_modules():
                _check_module(module)

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

    # For non-HF or non-FSDP cases, set tracking name for parameters
    if not is_using_hf() and not is_using_fsdp():
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

    if gpc.is_rank_for_log():
        logger.info(f"show model: {model}")
        logger.info(f"model params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, isp_communicator
