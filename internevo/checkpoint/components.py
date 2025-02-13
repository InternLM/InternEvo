import copy
import os
import re
from collections import defaultdict

import torch

from internevo.accelerator import get_accelerator
from internevo.core.context import ParallelMode
from internevo.core.context import global_context as gpc
from internevo.core.trainer import TrainState
from internevo.model.moe import MoE
from internevo.solver.optimizer import HybridZeroOptimizer, HybridZeroOptimizer_v2
from internevo.utils.common import get_current_device
from internevo.utils.lazy import LazyObject
from internevo.utils.logger import get_logger
from internevo.utils.parallel import is_using_fsdp, is_using_hf, is_using_isp
from internevo.utils.storage_manager import get_fns, llm_load, llm_save

from .utils import get_model_topology, get_non_moe_state_dict

try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    DCP_SUPPORTED = True
except (ImportError, ModuleNotFoundError):
    DCP_SUPPORTED = False

RESUME_HF_FORMAT = True

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


# only support auto resume
def try_load_moe_checkpoint(folder, model, state_dict, expert_mp_rank, pp_rank):
    """Load MoE layer parameters from separate files if the model has MoE layers."""
    # Calculate the stage size and rank within the pipeline parallelism
    pp_stage_size = gpc.config.model.num_layers // gpc.get_world_size(ParallelMode.PIPELINE)
    moe_layer_id = pp_rank * pp_stage_size
    mode = "wp" if is_using_isp() else "tp"

    # Iterate over all modules in the model to find MoE layers
    for _, module in model.named_modules():
        if isinstance(module, MoE):
            num_local_wrapped_experts = len(module.moe_layer.experts.wrapped_experts)
            expp_rank = gpc.get_local_rank(ParallelMode.EXPERT)
            # loop all local_experts
            for local_expert_id in range(num_local_wrapped_experts):
                global_expert_id = expp_rank * num_local_wrapped_experts + local_expert_id
                fn = f"model_moe_layer{moe_layer_id}_expert{global_expert_id}_{mode}{expert_mp_rank}.pt"
                fp = os.path.join(folder, fn)
                expert_state_dict = llm_load(fp, map_location=get_current_device())
                # Updating global -> local expert ids
                moe_str_prefix = ".moe_layer.experts.wrapped_experts."
                for key in list(expert_state_dict.keys()):
                    local_key = key.replace(f"{moe_str_prefix}{global_expert_id}", f"{moe_str_prefix}{local_expert_id}")
                    expert_state_dict[local_key] = expert_state_dict.pop(key)
                state_dict.update(expert_state_dict)
            moe_layer_id += 1


def try_save_moe_checkpoint(folder, model, expert_mp_rank, pp_rank):
    # Using layer_#_expert_# to save the model's expert state_dict，a hack.
    pipeline_stage_size = gpc.config.model.num_layers // gpc.get_world_size(ParallelMode.PIPELINE)
    moe_layer_id = pp_rank * pipeline_stage_size
    mode = "wp" if is_using_isp() else "tp"
    for n_module, module in model.named_modules():
        if isinstance(module, MoE):
            num_local_wrapped_experts = len(module.moe_layer.experts.wrapped_experts)
            expp_rank = gpc.get_local_rank(ParallelMode.EXPERT)

            # get all moe parameters
            moe_state_dict = {}
            for n, p in module.state_dict().items():
                if "expert" in n and "moe_layer.gate" not in n:
                    moe_state_dict[n_module + "." + n] = p
            moe_str_prefix = ".moe_layer.experts.wrapped_experts."
            # Reorder the moe name rank, so that each checkpoint only has one expert
            experts_state_dict = defaultdict(dict)
            for key in list(moe_state_dict.keys()):
                m = re.match(f".*{moe_str_prefix}([0-9]+).*", key)

                local_expert_id = None
                if not m:
                    logger.warning(f"No expert found in key {key}.")
                else:
                    local_expert_id = m.group(1)

                global_expert_id = expp_rank * num_local_wrapped_experts + int(local_expert_id)
                expert_key = key.replace(f"{moe_str_prefix}{local_expert_id}", f"{moe_str_prefix}{global_expert_id}")

                # truncating extra tensor (shared) storage
                truncated = moe_state_dict.pop(key).clone().detach()
                experts_state_dict[str(global_expert_id)][expert_key] = truncated

            # let save the moe parameters
            for global_expert_id, expert_state_dict in experts_state_dict.items():
                # save the moe parameters
                fn = f"model_moe_layer{moe_layer_id}_expert{global_expert_id}_{mode}{expert_mp_rank}.pt"
                fp = os.path.join(folder, fn)
                llm_save(fp, saved_obj=expert_state_dict)
            moe_layer_id += 1


def load_fsdp_model_checkpoint(folder, model):
    if DCP_SUPPORTED:
        assert folder.startswith("local:"), "Currently we only support DCP load and save locally."
        local_folder = folder[6:]

        if is_using_hf() and RESUME_HF_FORMAT:
            hf = gpc.config.hf
            mod = LazyObject(hf.mod, hf.mod_cls)
            mod = mod.build()
            state_dict = mod.from_pretrained(
                pretrained_model_name_or_path=os.path.join(local_folder, "hf"), use_safetensors=True
            ).state_dict()
            state_dict = {f"model.{key}": state_dict[key].clone().detach() for key in state_dict}
            set_model_state_dict(
                model=model, model_state_dict=state_dict, options=StateDictOptions(full_state_dict=True)
            )
        else:
            state_dict = get_model_state_dict(model=model)
            state_dict = {key: state_dict[key].clone().detach() for key in state_dict}
            dcp.load(state_dict=state_dict, checkpoint_id=local_folder)
            set_model_state_dict(model=model, model_state_dict=state_dict)

        del state_dict
        internlm_accelerator.empty_cache()
    else:
        raise RuntimeError("DCP is not supported in this version of PyTorch.")


def load_model_checkpoint(folder, model):
    """
    There should be weights with names similar to the following under the folder.
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If tensor parallel mode is isp, the saved weight is named:
    - folder
        - model_wp{wp_rank}_pp{pp_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.
    """

    if is_using_fsdp():
        return load_fsdp_model_checkpoint(folder, model)

    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)

    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    fns = get_fns(folder)

    _start_with = "model_w" if is_using_isp() else "model_t"

    max_pp, max_wp, max_tp = 0, 0, 0
    for fn in fns:
        if fn.startswith(_start_with) and not fn.endswith(".md5"):
            segements = os.path.splitext(fn)[0].split("_")
            if is_using_isp():
                max_pp = max(max_pp, int(segements[-1][2:]))
                max_wp = max(max_wp, int(segements[-2][2:]))
            else:
                max_pp = max(max_pp, int(segements[-1][2:]))
                max_tp = max(max_tp, int(segements[-2][2:]))

    assert (
        pp_size == max_pp + 1
    ), f"The weights are save for {max_pp+1} pipelines, while current has {pp_size} pipelines"
    assert (
        wp_size == max_wp + 1
    ), f"The weights are save for {max_wp+1} parallelism, while current has {wp_size} weight parallelism"
    if not is_using_isp():
        assert (
            tp_size == max_tp + 1
        ), f"The weights are save for {max_tp+1} parallelism, while current has {tp_size} tensor parallelism"
    if is_using_isp():
        should_load_name = f"model_wp{wp_rank}_pp{pp_rank}.pt"
    else:
        should_load_name = f"model_tp{tp_rank}_pp{pp_rank}.pt"
    fp = os.path.join(folder, should_load_name)

    states = llm_load(fp, map_location=get_current_device())
    """
    # need convert the gate parameters to float32 (to fit deepspeed style mechanism), it may cause round-off in
    # gate.weight. The conversion will also be done when doing forward. so we can just comment it out. this make
    # the gate parameters to be float16 before forward.
    for key in list(states.keys()):
        if 'moe_layer.gate.wg.weight' in key:
            states[key] = states[key].float()
            print("load: ", states[key].float(),flush=True)
    """
    if is_using_isp():
        expert_wp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)
        try_load_moe_checkpoint(folder, model, states, expert_wp_rank, pp_rank)
    else:
        expert_tp_rank = 0 if gpc.config.parallel.expert.no_tp else tp_rank
        try_load_moe_checkpoint(folder, model, states, expert_tp_rank, pp_rank)

    missing_k, unexpected_keys = model.load_state_dict(states, strict=False)
    if len(missing_k) != 0:
        logger.warning(f"Warning: missing keys {missing_k}")
    if len(unexpected_keys) != 0:
        logger.warning(f"Warning: unexpected keys {unexpected_keys}")

    # avoid to cuda oom, Ref: https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/11
    del states
    internlm_accelerator.empty_cache()


def save_fsdp_model_checkpoint(folder, model):
    def remove_model_prefix(state_dict):
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("model.", "", 1)
            new_state_dict[new_key] = state_dict[key].clone().detach()
        return new_state_dict

    if DCP_SUPPORTED:
        assert folder.startswith("local:"), "Currently we only support DCP load and save locally."
        local_folder = folder[6:]

        if is_using_hf() and RESUME_HF_FORMAT:
            state_dict = remove_model_prefix(
                get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            )
            if state_dict:
                hf = gpc.config.hf
                cfg = LazyObject(hf.cfg, hf.cfg_cls)
                cfg = cfg.build()
                mod = LazyObject(hf.mod, hf.mod_cls)
                mod = mod.build()
                with torch.device("meta"):
                    mod_to_save = mod(cfg(**hf.cfg_extra_kwargs))
                mod_to_save.load_state_dict(state_dict, strict=True, assign=True)
                mod_to_save.save_pretrained(save_directory=os.path.join(local_folder, "hf"), safe_serialization=True)
        else:
            dcp.save(get_model_state_dict(model=model), checkpoint_id=local_folder)

        torch.distributed.barrier()
    else:
        raise RuntimeError("DCP is not supported in this version of PyTorch.")


def save_model_checkpoint(folder, model):
    """
    Save the model according to the relationship between tp and dp. The principle is that the data of each tp
    will not be gathered and saved separately, which is equivalent to actual sharding. The saved weight is named
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    If tensor parallel mode is isp, the saved weight is named:
    - folder
        - model_wp{wp_rank}_pp{pp_rank}.pt

    If the tp is inconsistent with the saved one in the future use, the weight needs to be converted before loading.

    Args:
        folder: The folder to save the model
        model: The model to be saved
    """

    if is_using_fsdp():
        return save_fsdp_model_checkpoint(folder, model)

    states = model.state_dict()

    # get non-expert parameters
    states = get_non_moe_state_dict(states)
    topo = get_model_topology(model)

    if folder is not None:
        dp_size = gpc.get_world_size(ParallelMode.DATA)
        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        dp_rank = gpc.get_local_rank(ParallelMode.DATA)
        tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
        wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
        pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
        wdp_rank = gpc.get_local_rank(ParallelMode.WEIGHT_DATA)

        should_save_rank_pair = set()  # (tp_rank, dp_rank)

        # TODO In theory, we should also consider pp level, but since pp is generally a state across machines,
        # even if pp is not considered, it will definitely not be written on the same machine.

        # for tensor parallel mode with isp
        if is_using_isp():
            if wdp_rank == 0:
                fn = f"model_wp{wp_rank}_pp{pp_rank}.pt"
                fp = os.path.join(folder, fn)
                llm_save(fp, saved_obj=states)
                topo_fn = f"topo_wp{wp_rank}_pp{pp_rank}.json"
                topo_fp = os.path.join(folder, topo_fn)
                llm_save(topo_fp, saved_obj=topo)
            expert_wp_rank = gpc.get_local_rank(ParallelMode.EXPERT_WEIGHT)
            expert_wdp_rank = gpc.get_local_rank(ParallelMode.EXPERT_DATA)
            if expert_wdp_rank == 0:
                try_save_moe_checkpoint(folder, model, expert_wp_rank, pp_rank)
        else:
            # for tensor parallel mode with mtp/msp/fsp
            for i in range(tp_size):
                should_save_rank_pair.add((i, i % dp_size))

                if (tp_rank, dp_rank) in should_save_rank_pair:
                    fn = f"model_tp{tp_rank}_pp{pp_rank}.pt"
                    fp = os.path.join(folder, fn)
                    llm_save(fp, saved_obj=states)
                    topo_fn = f"topo_tp{tp_rank}_pp{pp_rank}.json"
                    topo_fp = os.path.join(folder, topo_fn)
                    llm_save(topo_fp, saved_obj=topo)

            # try to save expert parameter to separate files if model have moe layer
            expert_dp_size = gpc.get_world_size(ParallelMode.EXPERT_DATA)
            expert_tp_size = 1 if gpc.config.parallel.expert.no_tp else tp_size
            expert_dp_rank = gpc.get_local_rank(ParallelMode.EXPERT_DATA)
            expert_tp_rank = 0 if gpc.config.parallel.expert.no_tp else tp_rank
            should_save_rank_pair.clear()
            for i in range(expert_tp_size):
                should_save_rank_pair.add((i, i % expert_dp_size))

            if (expert_tp_rank, expert_dp_rank) in should_save_rank_pair:
                try_save_moe_checkpoint(folder, model, expert_tp_rank, pp_rank)

    torch.distributed.barrier()


def load_optimizer_checkpoint(folder, optim):
    """Load the optimizer state from the local file system or remote
    object storage Service (OSS).

    Args:
        optim (Optimizer): optimizer
        folder (str): The FS/OSS path where the optimizer will be stored.
    """

    fns = get_fns(folder)
    max_tp, max_wp, max_pp, max_zero = 0, 0, 0, 0
    max_fsdp = 0
    for fn in fns:
        if fn.startswith("optimizer_") and not fn.endswith(".md5"):
            if isinstance(optim, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
                if is_using_isp():
                    _, wp, pp, zero = os.path.splitext(fn)[0].split("_")
                    max_zero = max(max_zero, int(zero[2:]))
                    max_wp = max(max_wp, int(wp[2:]))
                    max_pp = max(max_pp, int(pp[2:]))
                else:
                    _, tp, pp, zero = os.path.splitext(fn)[0].split("_")
                    max_zero = max(max_zero, int(zero[2:]))
                    max_tp = max(max_tp, int(tp[2:]))
                    max_pp = max(max_pp, int(pp[2:]))
            else:
                _, fsdp = os.path.splitext(fn)[0].split("_")
                max_fsdp = max(max_fsdp, int(fsdp[4:]))

    fsdp_size = gpc.get_world_size(ParallelMode.GLOBAL)
    zero_size = gpc.get_world_size(ParallelMode.ZERO1)
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    pp_size = gpc.get_world_size(ParallelMode.PIPELINE)

    assert zero_size == max_zero + 1, (
        f"The optimizer states are save for {max_zero+1} zero parallel, "
        f"while current has {zero_size} zero broadcast range."
    )
    assert (
        pp_size == max_pp + 1
    ), f"The optimizer states are save for {max_pp+1} pipelines, while current has {pp_size} pipelines"
    if not is_using_isp():
        assert (
            tp_size == max_tp + 1
        ), f"The optimizer states are save for {max_tp+1} parallelism, while current has {tp_size} tensor parallelism"
    assert (
        wp_size == max_wp + 1
    ), f"The optimizer states are save for {max_wp+1} parallelism, while current has {wp_size} weight parallelism"

    if not isinstance(optim, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
        assert (
            fsdp_size == max_fsdp + 1
        ), f"The optimizer states are save for {max_fsdp+1} parallelism, while current has {fsdp_size} fsdp parallelism"

    fsdp_rank = gpc.get_local_rank(ParallelMode.GLOBAL)
    zero_rank = gpc.get_local_rank(ParallelMode.ZERO1)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)

    if isinstance(optim, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
        if is_using_isp():
            fp = f"optimizer_wp{wp_rank}_pp{pp_rank}_zo{zero_rank}.pt"
        else:
            fp = f"optimizer_tp{tp_rank}_pp{pp_rank}_zo{zero_rank}.pt"
    else:
        fp = f"optimizer_fsdp{fsdp_rank}.pt"

    states = llm_load(os.path.join(folder, fp), map_location=get_current_device())

    if isinstance(optim, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
        fp_meta = os.path.join(folder, optim.rank_unique_id)
        try:
            zero_devide_optim_plan = llm_load(fp_meta)
            states.update({"zero_devide_optim_plan": zero_devide_optim_plan})
        except Exception as e:
            if gpc.is_rank_for_log():
                logger.warning(
                    f"Read zero optimzer split file '{fp_meta}', for '{e}'"
                    f"Please check whether loading ckpts are saved with the HybridZeroOptimizer."
                )

    # compatible with old code that only have one param group, need to align with both parameter groups
    if len(states["base_optim_states"]["param_groups"]) == 1:
        for group in optim.param_groups:
            # for new added empty group, since it has no params, just create it fakely
            if len(group["params"]) == 0:
                states["base_optim_states"]["param_groups"].append(group)
            # for origin group, create new added attributes in recent updates
            else:
                saved_group = states["base_optim_states"]["param_groups"][0]
                saved_group["dp_mode"] = group["dp_mode"]
                saved_group["dtype"] = group["dtype"]

    optim.load_state_dict(states)
    del states
    internlm_accelerator.empty_cache()


def save_optimizer_checkpoint(optim, state_path):
    """Store the state of the optimizer to the local file system or remote OSS.

    Args:
        optim (Optimizer)
        state_path (str): The state loading path of optimizer.
    """

    # TODO sanity check for optimizer type
    fsdp_rank = gpc.get_local_rank(ParallelMode.GLOBAL)
    zero_rank = gpc.get_local_rank(ParallelMode.ZERO1)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    wp_rank = gpc.get_local_rank(ParallelMode.WEIGHT)
    pp_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    zero_size = gpc.get_world_size(ParallelMode.ZERO1)
    tp_size = gpc.get_world_size(ParallelMode.TENSOR)
    wp_size = gpc.get_world_size(ParallelMode.WEIGHT)
    dp_size = gpc.get_world_size(ParallelMode.DATA)

    states = optim.state_dict()
    if isinstance(optim, (HybridZeroOptimizer, HybridZeroOptimizer_v2)):
        if is_using_isp():
            fp = f"optimizer_wp{wp_rank}_pp{pp_rank}_zo{zero_rank}.pt"
            if (gpc.get_global_rank() % (tp_size * dp_size)) < zero_size * wp_size:
                llm_save(os.path.join(state_path, fp), states)
        else:
            fp = f"optimizer_tp{tp_rank}_pp{pp_rank}_zo{zero_rank}.pt"
            if (gpc.get_global_rank() % (tp_size * dp_size)) < zero_size * tp_size:
                llm_save(os.path.join(state_path, fp), states)
        if "zero_devide_optim_plan" in states:
            params_per_rank_id_dict = states.pop("zero_devide_optim_plan")
            fp_meta = os.path.join(state_path, optim.rank_unique_id)
            llm_save(fp_meta, params_per_rank_id_dict)
    else:
        fp = f"optimizer_fsdp{fsdp_rank}.pt"
        llm_save(os.path.join(state_path, fp), states)


def load_sampler(ckpt_path: str, sampler):
    sampler_states = llm_load(os.path.join(ckpt_path, "sampler.pt"))
    sampler.load_state_dict(sampler_states)
    if gpc.is_rank_for_log():
        pstate = copy.deepcopy(sampler_states)
        pstate.pop("indices", None)
        pstate.pop("rng_state", None)
        logger.info(f"reload sampler_states:{pstate}")
    internlm_accelerator.empty_cache()


def load_context(ckpt_path: str, train_state: TrainState):
    context_stuffs = llm_load(os.path.join(ckpt_path, "context.pt"))
    train_state.load_state_dict(context_stuffs)
    if gpc.is_rank_for_log():
        logger.info(f"reload train_state:{train_state}")
    internlm_accelerator.empty_cache()


def load_scheduler(ckpt_path: str, lr_scheduler, optimizer, train_state: TrainState):
    learning_rate = train_state.lr
    scheduler_states = llm_load(os.path.join(ckpt_path, "schedulder.pt"))
    if learning_rate != scheduler_states["base_lrs"][0] and gpc.is_rank_for_log():
        logger.warning(
            f"Using new learning rate {learning_rate} to replace old learn rate {scheduler_states['base_lrs'][0]}."
        )

    base_lrs = copy.deepcopy(scheduler_states["base_lrs"])
    scheduler_states["base_lrs"] = [learning_rate] * len(scheduler_states["base_lrs"])
    if "after_scheduler_dict" in scheduler_states:
        scheduler_states["after_scheduler_dict"]["base_lrs"] = [learning_rate] * len(
            scheduler_states["after_scheduler_dict"]["base_lrs"]
        )

    lr_scheduler.load_state_dict(scheduler_states)

    # step_count have been updated before saving checkpoint.
    lr_scheduler.last_epoch = train_state.step_count

    # compatible with old code that only have one param group
    if len(base_lrs) == 1:
        base_lrs = base_lrs * len(optimizer.param_groups)

    ratios = [learning_rate / lr for lr in base_lrs]
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = param_group["lr"] * ratios[idx]
    internlm_accelerator.empty_cache()

    if gpc.is_rank_for_log():
        logger.info(f"reload load_scheduler:{lr_scheduler}")
