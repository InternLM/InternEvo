import collections
import itertools
from typing import List, Optional, Set, Union

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from internlm.accelerator.abstract_accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.parallel.shard import split_data_for_sequence_parallel
from internlm.data.utils import packed_data_normalizer, unpack_data
from internlm.utils.common import get_current_device
from internlm.utils.lazy import LazyObject
from internlm.utils.logger import get_logger
from internlm.utils.parallel import is_using_fsdp, is_using_hf, is_using_isp

internlm_accelerator = get_accelerator()
logger = get_logger(__file__)

try:
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.tensor import DeviceMesh
    FSDP2_SUPPORTED = True
except (ImportError, ModuleNotFoundError):
    FSDP2_SUPPORTED = False

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


def _get_modules_to_materialize(
    root_module: nn.Module,
    ignored_modules: Set[nn.Module],
) -> List[nn.Module]:
    # Run BFS to collect the modules to materialize via `reset_parameters()`,
    # stopping at any module with FSDP already applied or at ignored modules.
    modules_to_materialize: List[nn.Module] = []
    queue = collections.deque([root_module])
    visited_modules: Set[nn.Module] = {root_module}
    while queue:
        module = queue.popleft()
        modules_to_materialize.append(module)
        for child_module in module.children():
            if child_module not in visited_modules and child_module not in ignored_modules:
                visited_modules.add(child_module)
                queue.append(child_module)
    return modules_to_materialize


def _materialize_meta_module(
    root_module: nn.Module,
    ignored_modules: Set[nn.Module],
    device_id: Optional[torch.device],
) -> None:
    # Run default meta device initialization
    modules_to_materialize = _get_modules_to_materialize(root_module, ignored_modules)
    module = None
    try:
        # Assume that each module's `reset_parameters()` only initializes its
        # own parameters and not those of its children
        with torch.no_grad():
            for module in modules_to_materialize:
                # As a contract to the user, only call `reset_parameters()` if
                # the module has directly managed parameters/buffers
                module_state_iter = itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))
                has_module_states = len(list(module_state_iter)) > 0
                if has_module_states:
                    module.to_empty(device=device_id, recurse=False)
                    module.reset_parameters()  # type: ignore[operator]
    except BaseException as e:
        logger.warning(
            "Unable to call `reset_parameters()` for module on meta "
            f"device with error {str(e)}. Please ensure that your module of"
            f"type {type(module)} implements a `reset_parameters()` method."  # type: ignore[possibly-undefined]
        )
        raise e


def _init_fsdp_v1(model: FSDP, device: torch.device) -> FSDP:
    """
    Initialize Fully Sharded Data Parallel (FSDP) for the model.
    This function is needed to properly initialize FSDP when resuming from a checkpoint.
    It runs a forward pass with dummy inputs to ensure FSDP is fully initialized.

    References:
    https://github.com/pytorch/pytorch/issues/113496
    https://github.com/huggingface/transformers/pull/34032
    https://github.com/huggingface/transformers/issues/31892

    Args:
        model: The model to initialize with FSDP.
        device: The device to run the model on.

    Returns:
        The initialized FSDP model.
    """
    model.train()
    with torch.no_grad():
        # generate dummy packed sequence
        seq_len = gpc.config.data.seq_len * gpc.config.data.micro_bsz
        input_ids = [1] * seq_len
        label = input_ids[1:] + [-100]
        cu_seqlens = list(range(0, seq_len + gpc.config.data.seq_len, gpc.config.data.seq_len))

        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        label = torch.tensor(label, device=device).unsqueeze(0)
        indexes = torch.tensor(
            list(itertools.chain(*[np.arange(l2 - l1) for l1, l2 in zip(cu_seqlens[:-1], cu_seqlens[1:])])),
            device=device,
        ).unsqueeze(0)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32).unsqueeze(0)

        data = {
            "input_ids": input_ids,
            "cu_seqlens": cu_seqlens,
            "indexes": indexes,
            "max_seqlen": seq_len,
        }

        data_fns = []

        # default data process function
        if gpc.config.data.use_packed_dataset:
            data_fns.append(packed_data_normalizer)
        else:
            data_fns.append(unpack_data)

        # support sequence parallel for isp
        if is_using_isp():
            data_fns.append(split_data_for_sequence_parallel)

        # generate dummy_input
        _data, _label = data, label
        for fn in data_fns:
            _data, _label = fn(_data, _label)
        dummy_input = _data

        # run a forward pass with dummy_input to initialize FSDP
        _ = model(**dummy_input)
    return model


def wrap_FSDP_model(model: Union[nn.Module, nn.ModuleList]):
    if is_using_fsdp():
        assert isinstance(model, nn.Module), "Currently FSDP does not support pipeline parallel."
        wrap_cls = tuple(
            LazyObject(warp_cls["mod"], warp_cls["mod_cls"]).build() for warp_cls in gpc.config.get("fsdp_wrap_cls", [])
        )
        fsdp_mode = gpc.config.parallel.fsdp.get("mode", "v1")
        fsdp_init_method = gpc.config.parallel.fsdp.get("init_method", "cuda")
        if gpc.is_using_parallel_mode(ParallelMode.EXPERT):
            assert gpc.get_world_size(ParallelMode.EXPERT_DATA) * gpc.get_world_size(ParallelMode.EXPERT) == gpc.get_world_size(ParallelMode.GLOBAL)

        if fsdp_mode == "v1":
            ignored_mod = []
            if gpc.is_using_parallel_mode(ParallelMode.EXPERT):
                for layer_id, layer in enumerate(model.model.layers):
                    if layer_id >= gpc.config.model.first_k_dense_replace:
                        # Should follow this modeling pattern if EP is enabled.
                        # Change the expert module name if needed.
                        # TODO: Make this part hard-coded or config-driven?
                        layer.feed_forward.moe_layer.experts = FSDP(
                            layer.feed_forward.moe_layer.experts, 
                            process_group=gpc.get_group(ParallelMode.EXPERT_DATA),
                            sharding_strategy=ShardingStrategy.FULL_SHARD, 
                            sync_module_states=fsdp_init_method != "cuda",  # sync model paramters
                            forward_prefetch=True,
                            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                            limit_all_gathers=True,
                            use_orig_params=True,
                            device_id=None if fsdp_init_method == "cuda" else get_current_device(),  # needed for sync_module_states
                        )
                        ignored_mod.append(layer.feed_forward.moe_layer.experts)
            model = FSDP(
                module=model,
                process_group=gpc.get_group(ParallelMode.GLOBAL),
                sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO2: SHARD_GRAD_OP, ZeRO3: FULL_SHARD
                auto_wrap_policy=ModuleWrapPolicy(wrap_cls),
                sync_module_states=fsdp_init_method != "cuda",  # sync model paramters
                forward_prefetch=True,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                limit_all_gathers=True,
                use_orig_params=True,
                device_id=None if fsdp_init_method == "cuda" else get_current_device(),  # needed for sync_module_states
                ignored_modules=ignored_mod,
            )
            # For FSDP v1, to get ckpt resuming work normally, we do dummy forward.
            # This hack is needed due to FSDP v1 lazy initialization in model construction.
            # FYI: https://github.com/pytorch/pytorch/issues/113496
            model = _init_fsdp_v1(model, get_current_device())
        elif FSDP2_SUPPORTED and fsdp_mode == "v2":
            fsdp_kwargs = {
                "reshard_after_forward": True,  # ZeRO2: False, ZeRO3: True
            }
            if gpc.is_using_parallel_mode(ParallelMode.EXPERT):
                device_mesh = DeviceMesh.from_group(
                    group=[gpc.get_group(ParallelMode.EXPERT), gpc.get_group(ParallelMode.EXPERT_DATA)], 
                    device_type="cuda", 
                    mesh=torch.arange(
                        gpc.get_world_size(ParallelMode.GLOBAL), 
                    ).view((gpc.get_world_size(ParallelMode.EXPERT), gpc.get_world_size(ParallelMode.EXPERT_DATA))), 
                    mesh_dim_names=("ep", "edp"),
                )
                for layer_id, layer in enumerate(model.model.layers):
                    if layer_id >= gpc.config.model.first_k_dense_replace:
                        # Should follow this modeling pattern if EP is enabled.
                        # Change the expert module name if needed.
                        # TODO: Make this part hard-coded or config-driven?
                        fully_shard(layer.feed_forward.moe_layer.experts, mesh=device_mesh["edp"], **fsdp_kwargs)
            for module in model.modules():
                if isinstance(module, wrap_cls):
                    fully_shard(module, **fsdp_kwargs)
            fully_shard(model, **fsdp_kwargs)
            if fsdp_init_method == "meta":
                _materialize_meta_module(model, set(), get_current_device())
            elif fsdp_init_method == "cpu":
                model.to(get_current_device())
        else:
            raise ValueError(f"Unsupported FSDP mode: {fsdp_mode}")

        if not gpc.config.ckpt.get("auto_resume", False):
            load_ckpt_info = gpc.config.ckpt.load_ckpt_info
            load_ckpt_path = load_ckpt_info.get("path", None)
            load_ckpt_content = load_ckpt_info.get("content", [])
            if load_ckpt_path:
                assert load_ckpt_content == (
                    "model",
                ), "If auto_resume=False and checkpoint path is given, only model can be loaded"
                if DCP_SUPPORTED:
                    if is_using_hf():
                        hf = gpc.config.hf
                        mod = LazyObject(hf.mod, hf.mod_cls)
                        mod = mod.build()
                        state_dict = mod.from_pretrained(
                            pretrained_model_name_or_path=load_ckpt_path, use_safetensors=True
                        ).state_dict()
                        state_dict = {f"model.{key}": state_dict[key].clone().detach() for key in state_dict}
                        set_model_state_dict(
                            model=model, model_state_dict=state_dict, options=StateDictOptions(full_state_dict=True)
                        )
                    else:
                        state_dict = get_model_state_dict(model=model)
                        state_dict = {key: state_dict[key].clone().detach() for key in state_dict}
                        dcp.load(state_dict=state_dict, checkpoint_id=load_ckpt_path)
                        set_model_state_dict(model=model, model_state_dict=state_dict)
                    del state_dict
                    internlm_accelerator.empty_cache()
                else:
                    raise RuntimeError("DCP is not supported in this version of PyTorch.")

    return model
