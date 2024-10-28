#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Dict, Union

import torch

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import Config
from internlm.core.context import global_context as gpc
from internlm.core.context.process_group_initializer import ParallelMode
from internlm.utils.common import get_master_node
from internlm.utils.gputest import warmup_process_group
from internlm.utils.logger import get_logger
from internlm.utils.timeout import llm_timeout
from internlm.utils.utils import DataType, ModelType, TensorParallelMode

# check package
try:
    import numa
    from numa import memory, schedule
    from pynvml.smi import nvidia_smi
except (AttributeError, ImportError):
    get_numa = False
else:
    get_numa = True

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def get_default_parser():
    """Reads user command line and uses an argument parser to parse the input arguments.
    Input arguments include configuration, host, port, world size, local rank, backend for torch.distributed.

    Returns:
       Parser: Returns the parser with the default arguments, the user may add customized arguments into this parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config file")
    parser.add_argument(
        "--launcher",
        type=str,
        default="slurm",
        choices=["slurm", "torch"],
        help="launcher for launching distributed environment",
    )
    parser.add_argument("--host", type=str, help="the master address for distributed training")
    parser.add_argument("--port", type=int, default=8888, help="the master port for distributed training")
    parser.add_argument("--world_size", type=int, help="world size for distributed training")
    parser.add_argument("--rank", type=int, help="rank for the default process group")
    parser.add_argument("--local_rank", type=int, help="local rank on the node")
    parser.add_argument("--backend", type=str, default="nccl", help="backend for distributed communication")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--profiling", default=False, action="store_true", help="enable/disable profiling.")
    parser.add_argument("--enable_ali_topology", default=False, action="store_true", help="enable ali switch topology.")
    parser.add_argument(
        "--disable_volc_topology", default=False, action="store_true", help="disable volc switch topology."
    )
    return parser


def args_sanity_check():
    assert gpc.config is not None, "config is not load!"

    if "JOB_NAME" not in gpc.config:
        gpc.config._add_item("JOB_NAME", "AnonymousJob")

    # the default model type is INTERNLM
    if "model_type" not in gpc.config:
        gpc.config._add_item("model_type", ModelType.INTERNLM.name)

    if "use_apex_adam" not in gpc.config:
        gpc.config._add_item("use_apex_adam", False)

    # procssing the parallel config in gpc
    if "zero1" not in gpc.config.parallel:
        gpc.config.parallel._add_item("zero1", dict(size=-1, fsdp=False))

    if isinstance(gpc.config.parallel.zero1, int):
        zero1_size = gpc.config.parallel.zero1
        gpc.config.parallel._add_item("zero1", dict(size=zero1_size, fsdp=False))

    if "pipeline" not in gpc.config.parallel:
        gpc.config.parallel._add_item("pipeline", dict(size=1, interleaved_overlap=False, mode="1F1B"))

    if isinstance(gpc.config.parallel.pipeline, dict) and "mode" not in gpc.config.parallel.pipeline:
        gpc.config.parallel.pipeline._add_item("mode", "1F1B")

    if "tensor" not in gpc.config.parallel:
        gpc.config.parallel._add_item("tensor", dict(size=1, mode=TensorParallelMode.mtp.name))

    if "weight" not in gpc.config.parallel:
        gpc.config.parallel._add_item("weight", dict(size=1, overlap=False))

    if "expert" not in gpc.config.parallel:
        gpc.config.parallel._add_item("expert", dict(size=-1, no_tp=False))

    if "expert_weight" not in gpc.config.parallel:
        gpc.config.parallel._add_item("expert_weight", dict(size=1, overlap=False))

    if isinstance(gpc.config.parallel.pipeline, int):
        pp = gpc.config.parallel.pipeline
    else:
        pp = gpc.config.parallel.pipeline.size

    if isinstance(gpc.config.parallel.pipeline, dict):
        gpc.config.parallel.pipeline["mode"] = gpc.config.parallel.pipeline["mode"].upper()
        assert gpc.config.parallel.pipeline["mode"] in [
            "1F1B",
            "ZBH1",
            "ZBV",
        ], f"unsupported pp mode {gpc.config.parallel.pipeline['mode']}"
        if gpc.config.parallel.pipeline["mode"] == "ZBV":
            gpc.v_shape = True

    # check fsdp config
    if "fsdp" not in gpc.config.parallel.zero1:
        gpc.config.parallel.zero1._add_item("fsdp", False)

    assert not (
        gpc.config.parallel.zero1.fsdp and pp > 1
    ), "FSDP is not supportted when pipeline size > 1, please set pipeline size to 1 or disabled FSDP"

    if gpc.config.parallel.zero1.fsdp:
        assert (
            torch.__version__ >= "2.0.1"
        ), f"requires torch>=2.0.1 when using fsdp but current version is {torch.__version__}"

    # processing the data config in gpc
    data = gpc.config.data

    assert data.seq_len is not None, "'seq_len' must be given a value"
    assert data.micro_bsz is not None, "'micro_bsz' must be given a value"

    if "packed_length" in data and gpc.is_rank_for_log():
        logger.warning("packed_length would be ignored and will be setted as seq_len * micro_bsz.")

    data._add_item("packed_length", data.seq_len * data.micro_bsz)

    if "type" not in data:
        data._add_item("type", DataType.tokenized.name)

    if "micro_num" not in data:
        data._add_item("micro_num", 1)

    if "gradient_accumulation" not in data:
        data._add_item("gradient_accumulation", data.micro_num)
        if gpc.is_rank_for_log():
            logger.info(f"gradient_accumulation size will be setted to {data.micro_num}.")
    else:
        if pp == 1:
            assert (
                data.gradient_accumulation == data.micro_num
            ), "for nopp 'gradient_accumulation' should equal with 'micro_num'"

    # batch_size should be equal with micro_num, should not use it directly
    data._add_item("batch_size", data.micro_num)

    if "min_length" not in data:
        data._add_item("min_length", 0)

    if "train_folder" not in data:
        data._add_item("train_folder", None)

    if "valid_folder" not in data:
        data._add_item("valid_folder", None)

    if "valid_micro_num" not in data:
        data._add_item("valid_micro_num", data.micro_num)

    if "valid_every" not in data:
        data._add_item("valid_every", 0)

    if "empty_cache_and_diag_interval" not in data:
        data._add_item("empty_cache_and_diag_interval", 50)

    if "diag_outlier_ratio" not in data:
        data._add_item("diag_outlier_ratio", 1.1)

    data.diag_outlier_ratio = max(1, data.diag_outlier_ratio)

    if "use_shm" not in data:
        data._add_item("use_shm", False)
    elif data.use_shm and "shm_path" not in data:
        data._add_item("shm_path", "/dev/shm/metacache")

    if data.train_folder is None:
        data.use_shm = False

    if "use_packed_dataset" not in data:
        data._add_item("use_packed_dataset", True)

    if "fixed_random_dataset_seqlen" not in data:
        data._add_item("fixed_random_dataset_seqlen", True)

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Data Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"seq_len: {data.seq_len}")
        logger.info(f"micro_num: {data.micro_num}")
        logger.info(f"micro_bsz: {data.micro_bsz}")
        logger.info(f"packed_length: {data.packed_length}")
        logger.info(f"pack_sample_into_one: {data.pack_sample_into_one}")
        logger.info(f"min_length: {data.min_length}")
        logger.info(f"valid_micro_num: {data.valid_micro_num}")
        logger.info(f"valid_every: {data.valid_every}")
        logger.info(f"rampup_batch_size: {data.rampup_batch_size}")

    # processing the checkpoint config
    ckpt = gpc.config.ckpt
    if "enable_save_ckpt" not in ckpt:
        ckpt._add_item("enable_save_ckpt", True)

    # Saving checkpoint args.
    if ckpt.enable_save_ckpt:
        assert "checkpoint_every" in ckpt, "If enable save checkpoint, must give checkpoint_every in config.data!"
        assert ckpt.checkpoint_every > 0
        assert "save_ckpt_folder" in ckpt, "If enable save checkpoint, must give save_ckpt_folder in config.data!"

        if "async_upload" not in ckpt:
            ckpt._add_item("async_upload", False)  # async defalut is False.
        else:
            if ckpt.async_upload:
                assert "save_ckpt_folder" in ckpt
                prefix_list = ["boto3:", "volc:", "oss2:"]
                if not any(ckpt.save_ckpt_folder.startswith(prefix) for prefix in prefix_list):
                    if gpc.is_rank_for_log():
                        logger.warning(
                            "Storing ckpt on file system does not support asynchronous storage, will use sync save!"
                        )
                    ckpt.async_upload = False
                else:
                    if "async_upload_tmp_folder" not in ckpt:
                        ckpt._add_item("async_upload_tmp_folder", "/dev/shm/internlm_tmp_ckpt/")

        if not ckpt.async_upload:
            ckpt._add_item("async_upload_tmp_folder", None)

        if "oss_snapshot_freq" not in ckpt:
            ckpt._add_item("oss_snapshot_freq", float("inf"))  # if oss_snapshot_freq not given, we disable.
    else:
        ckpt._add_item("checkpoint_every", float("inf"))
        ckpt._add_item("oss_snapshot_freq", float("inf"))
        ckpt._add_item("save_ckpt_folder", None)
        ckpt._add_item("async_upload", False)
        ckpt._add_item("async_upload_tmp_folder", None)
        ckpt._add_item("snapshot_ckpt_folder", None)

    if "load_ckpt_folder" not in ckpt:
        ckpt._add_item("load_ckpt_folder", None)

    if "stop_file_path" not in ckpt:
        ckpt._add_item("stop_file_path", None)

    if "auto_resume" not in ckpt:
        # If 'auto_resume' is not given, we set it to True, so internlm can have opportunity
        # to auto-load latest checkpoint.
        ckpt._add_item("auto_resume", True)

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Ckpt Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"is enable save ckpt: {ckpt.enable_save_ckpt}")
        logger.info(f"save_ckpt_folder: {ckpt.save_ckpt_folder}")
        logger.info(f"checkpoint_every: {ckpt.checkpoint_every}")

    # tensorboard writer config
    if "enable_tb" not in gpc.config:
        gpc.config._add_item("enable_tb", True)
    if "tensorboard_folder" not in gpc.config:
        gpc.config._add_item(
            "tensorboard_folder", os.environ["tensorboard_folder"] if "tensorboard_folder" in os.environ else None
        )
    if "resume_tb_folder" not in gpc.config:
        gpc.config._add_item(
            "resume_tb_folder", os.environ["resume_tb_folder"] if "resume_tb_folder" in os.environ else None
        )

    if gpc.is_rank_for_log():
        logger.info(f"tensorboard_folder: {gpc.config.tensorboard_folder}")
        logger.info(f"resume_tb_folder: {gpc.config.resume_tb_folder}")

    # cudnn
    torch.backends.cudnn.benchmark = gpc.config.get("cudnn_benchmark", False)
    torch.backends.cudnn.deterministic = gpc.config.get("cudnn_deterministic", False)
    clip_grad_norm = gpc.config.hybrid_zero_optimizer.get("clip_grad_norm", 0.0)

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Other Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"cudnn.benchmark: {torch.backends.cudnn.benchmark }")
        logger.info(f"cudnn.deterministic: {torch.backends.cudnn.deterministic }")
        logger.info(f"clip_grad_norm: {clip_grad_norm}")

    model = gpc.config.model
    if "dtype" not in model:
        logger.warning("dtype is not set, use torch.float16 by defalut!")
        model._add_item("dtype", torch.float16)
    else:
        if gpc.config.model.dtype == "torch.bfloat16":
            gpc.config.model.dtype = torch.bfloat16
        elif gpc.config.model.dtype in ("torch.float16", "torch.half"):
            gpc.config.model.dtype = torch.float16
        elif gpc.config.model.dtype == "torch.float32":
            gpc.config.model.dtype = torch.float32
        elif gpc.config.model.dtype == "torch.tf32":
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            gpc.config.model.dtype = torch.float32
        else:
            assert gpc.config.model.dtype in [
                "torch.float16",
                "torch.half",
                "torch.bfloat16",
                "torch.float32",
                "torch.tf32",
            ]

    if "checkpoint" in model:
        if model.checkpoint is True:
            model.checkpoint = 1
        elif model.checkpoint is False:
            model.checkpoint = 0
        else:
            assert (
                model.checkpoint >= 0 and model.checkpoint <= 1
            ), f'model.checkpoint: "{model.checkpoint}" should >=0 and <=1'

    if gpc.is_rank_for_log():
        logger.info("+" * 15 + " Model Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"Model: {gpc.config.model}")

        logger.info("+" * 15 + " grad_scaler Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"grad_scaler: {gpc.config.grad_scaler}")

        logger.info("+" * 15 + " hybrid_zero_optimizer Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"hybrid_zero_optimizer: {gpc.config.hybrid_zero_optimizer}")

        logger.info("+" * 15 + " adam Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"adam: {gpc.config.adam}")

        logger.info("+" * 15 + " beta2_scheduler Info " + "+" * 15)  # pylint: disable=W1201
        logger.info(f"beta2_scheduler: {gpc.config.beta2_scheduler}")

    # process the model config
    if "use_flash_attn" not in gpc.config.model:
        gpc.config.model._add_item("use_flash_attn", True)

    old_parallel_output = gpc.config.model.get("parallel_output", None)
    # Try to change user setting
    if internlm_accelerator.get_accelerator_backend() is not AcceleratorType.GPU:
        gpc.config.model.update({"parallel_output": False})
        if old_parallel_output is True and gpc.is_rank_for_log():
            logger.warning(
                "'parallel_output' is converted from 'True' to 'False'."
                "Because 'parallel_output' only support by FlashCrossEntropyLoss."
                "Please make sure you are using flash attention in cuda device."
            )

    if "MoE" in gpc.config.get("model_type", ModelType.INTERNLM.name):
        if "num_experts" not in model:
            model._add_item("num_experts", 1)
        if "top_k" not in model:
            model.top_k = 1
        if "num_shared_experts" not in model:
            model.num_shared_experts = 1 if getattr(model, "moe_use_residual", False) else 0
            if hasattr(model, "moe_use_residual"):
                delattr(model, "moe_use_residual")
        if "moe_type" not in model:
            model._add_item("moe_type", "GShard")
        if "moe_layer_kwargs" not in model:
            model.moe_layer_kwargs = {}

    if "mlp_layer_fusion" not in model:
        model._add_item("mlp_layer_fusion", False)

    # qk_interleaved config
    if "qk_interleaved" not in gpc.config.model:
        if "adapt_hf" in gpc.config.model:
            model._add_item("qk_interleaved", not gpc.config.model.adapt_hf)
        else:
            model._add_item("qk_interleaved", False)
    elif "adapt_hf" in gpc.config.model:
        assert gpc.config.model.adapt_hf == (
            not gpc.config.model.qk_interleaved
        ), "adapt_hf and qk_interleaved must be opposite"

    # process the parallel config
    if "sequence_parallel" not in gpc.config.parallel:
        gpc.config.parallel._add_item("sequence_parallel", False)

    # set default value for tensor parallel
    if isinstance(gpc.config.parallel["tensor"], int):
        gpc.config.parallel["tensor"] = dict(size=gpc.config.parallel["tensor"], mode=TensorParallelMode.mtp.name)
    if gpc.config.parallel["tensor"].get("mode", None) is None:
        gpc.config.parallel["tensor"]["mode"] = TensorParallelMode.mtp.name
    if gpc.config.parallel["tensor"]["mode"] == TensorParallelMode.isp.name:
        assert not gpc.config.parallel.zero1.fsdp, "FSDP does not support isp"
        assert (
            torch.__version__ >= "2.1.0"
        ), f"requires torch>=2.1.0 when using isp but current version is {torch.__version__}"

    assert gpc.config.parallel["tensor"].get("mode", None) in [
        TensorParallelMode.mtp.name,
        TensorParallelMode.msp.name,
        TensorParallelMode.fsp.name,
        TensorParallelMode.isp.name,
    ], "invalid tensor parallel mode, only ['mtp', 'msp', 'fsp', 'isp'] is supported"

    # for NPU accelerator supports: 1）FA-True + Packed-True 2) FA-False + Packed-False
    # for DIPU accelerator supports: 1）FA-True + Packed-False 2) FA-False + Packed-False
    # for GPU accelerator supports: 1）FA-True + Packed-True 2) FA-False + Packed-False
    if gpc.config.parallel["tensor"][
        "mode"
    ] == TensorParallelMode.isp.name and internlm_accelerator.get_accelerator_backend() in [
        AcceleratorType.NPU,
        AcceleratorType.DIPU,
        AcceleratorType.DITORCH,
    ]:
        assert (
            gpc.config.data.use_packed_dataset is False
        ), "only unpacked data is supported when tensor parallel mode is isp and accelerator type is NPU or DIPU"

    if internlm_accelerator.get_accelerator_backend() in [
        AcceleratorType.NPU,
        AcceleratorType.DIPU,
        AcceleratorType.DITORCH,
    ]:
        assert (
            gpc.config.model.use_flash_attn == gpc.config.data.use_packed_dataset
        ), "use_packed_dataset should be set same value as use_flash_attn"

    # adapt to old version's sequence parallel config
    if gpc.config.parallel["tensor"].get("mode", None) in [
        TensorParallelMode.msp.name,
        TensorParallelMode.fsp.name,
        TensorParallelMode.isp.name,
    ]:
        gpc.config.parallel.sequence_parallel = True

    # set default value for weight parallel
    if gpc.config.parallel["weight"].get("overlap", None) is None:
        gpc.config.parallel["weight"]["overlap"] = False
    if gpc.config.parallel["tensor"]["mode"] != TensorParallelMode.isp.name:
        assert gpc.config.parallel["weight"]["size"] <= 1, "weight parallel is only supported with isp"
    # set default value for expert_weight parallel
    if gpc.config.parallel["expert_weight"].get("overlap", None) is None:
        gpc.config.parallel["expert_weight"]["overlap"] = False
    if gpc.config.parallel["expert"].get("no_tp", None) is None:
        gpc.config.parallel["expert"]["no_tp"] = False
    # currently only interleaved pipeline scheduler with overlap can guarantee loss accuracy
    if hasattr(gpc.config.model, "num_chunks") and gpc.config.model.num_chunks > 1:
        assert (
            gpc.config.parallel["pipeline"].get("interleaved_overlap", False) is True
        ), "only support interleaved pipeline scheduler with overlap"

    if gpc.config.parallel["pipeline"]["mode"] == "ZBV":
        gpc.config.model.num_chunks = 2
        if gpc.is_rank_for_log():
            logger.info("Using zero_bubble_v, num_chunks is set to 2.")

    # monitoring default config
    monitor_default_config = {
        "alert_address": None,  # compatible with old alert config
        "monitor": {  # new monitoring config
            "alert": {
                "enable_feishu_alert": False,
                "feishu_alert_address": None,
                "light_monitor_address": None,
                "alert_file_path": None,
            }
        },
        "tensorboard": {
            "queue_max_length": 1,
        },
    }

    for key, value in monitor_default_config.items():
        if key not in gpc.config:
            gpc.config._add_item(key, value)

    alert = gpc.config.monitor.alert

    if alert.enable_feishu_alert and not alert.feishu_alert_address and gpc.is_rank_for_log():
        logger.warning("alert is enable but alert_address is not set")

    optim_ckpt = gpc.config.hybrid_zero_optimizer
    if "zero_overlap_communication" in optim_ckpt:
        # Compatible with the old interfaces.
        optim_ckpt._add_item("overlap_sync_grad", optim_ckpt.zero_overlap_communication)
    if "overlap_sync_grad" not in optim_ckpt:
        optim_ckpt._add_item("overlap_sync_grad", False)
    if "overlap_sync_param" not in optim_ckpt:
        optim_ckpt._add_item("overlap_sync_param", False)
    if "use_split_tensor_optim" not in optim_ckpt:
        optim_ckpt._add_item("use_split_tensor_optim", False)
    elif optim_ckpt.use_split_tensor_optim and "all_gather_size" not in optim_ckpt:
        optim_ckpt._add_item("all_gather_size", 512 * 1024 * 1024)

    if gpc.config.parallel["pipeline"]["mode"] == "ZBH1":
        assert (
            not optim_ckpt.overlap_sync_grad
        ), "When using zero_bubble pipeline parallelism, overlap_sync_grad must be false"
        assert (
            getattr(gpc.config.model, "num_chunks", 1) == 1
        ), "zero_bubble pp and interleaved pp cannot be used at the same time"
        if gpc.config.parallel["tensor"]["mode"] == "isp":
            assert not gpc.config.parallel["weight"].get(
                "overlap", False
            ), "When using zero_bubble pipeline parallelism, isp_overlap must be false"

    if gpc.is_rank_for_log():
        logger.info(
            f"overlap_sync_grad:{optim_ckpt.overlap_sync_grad}, overlap_sync_param:{optim_ckpt.overlap_sync_param}"
        )

    if "batch_count" not in gpc.config:
        gpc.config._add_item("batch_count", 0)

    if "moe_loss_coeff" not in gpc.config.loss:
        gpc.config.loss._add_item("moe_loss_coeff", 1.0)

    if "selective_checkpoint" not in gpc.config:
        gpc.config._add_item("selective_checkpoint", False)

    # moe not support overlap and zero1.5 for now
    if gpc.config.model.get("num_experts", 1) > 1:
        assert not gpc.config.parallel.zero1.fsdp, "FSDP does not support num_experts > 1"
        assert (
            not optim_ckpt.overlap_sync_grad & optim_ckpt.overlap_sync_param
        ), "not support overlap and moe at the same time"
        assert gpc.config.parallel.zero1.size in (
            -1,
            gpc.get_world_size(ParallelMode.DATA),
        ), "moe only support zero1, set zero1=dict(size=-1,...) can fix this"

        if gpc.config.parallel.tensor.mode != "isp":
            assert gpc.config.parallel.expert_weight.size <= 1, "expert weight parallel is only supported with isp"
    else:
        assert (
            gpc.config.parallel.expert.size <= 1 and gpc.config.parallel.expert_weight.size <= 1
        ), "expert parallel is only supported in MoE setting"

    # sequence_2D
    if "sequence_2D" not in gpc.config.parallel:
        gpc.config.parallel._add_item(
            "sequence_2D",
            {
                "enable": False,
                "head_size": 1,
                "context_size": 1,
                "window_size": 1,
                "device_placement_strategy": {"head_first": True, "interleaved": False},
            },
        )
    else:
        if gpc.config.parallel.sequence_2D.enable is True:
            parallel_cfg = gpc.config.parallel
            assert (
                parallel_cfg.sequence_2D.head_size * parallel_cfg.sequence_2D.context_size == parallel_cfg.tensor.size
            ), "the head_size * context_size should be equal to the tensor size."

            if (
                parallel_cfg.sequence_2D.device_placement_strategy.head_first is True
                and parallel_cfg.sequence_2D.head_size > 1
            ):
                assert (
                    parallel_cfg.sequence_2D.device_placement_strategy.interleaved is False
                ), "if head_first is True, the interleaved should be False."

            assert (
                gpc.config.data.use_packed_dataset is False
            ), "only unpacked data is supported when using 2D sequence parallel."


def launch(
    config: Union[str, Path, Config, Dict],
    rank: int,
    world_size: int,
    host: str,
    port: int,
    backend: str = "nccl",
    local_rank: int = None,
    seed: int = 1024,
):
    """This function first parses the configuration arguments, using :func:`parse_args()` in case one of the input
    arguments are not given. Then initialize and set distributed environment by calling global_context's functions.

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        rank (int): Rank for the default process group
        world_size (int): World size of the default process group
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        local_rank (int, optional):
            Rank for the process on the node and is used to set the default CUDA device,
            defaults to None. If local_rank = None, the default device ordinal will be calculated automatically.
        seed (int, optional): Specified random seed for every process. Defaults to 1024.

    Raises:
        Exception: Raise exception when config type is wrong
    """

    # set config
    assert isinstance(
        config, (Config, str, Path, dict)
    ), f"expected argument config to be Config, str or Path, but got {type(config)}"
    if not isinstance(config, Config) and isinstance(config, dict):
        config = Config(config)
    if isinstance(config, (str, Path)):
        config = Config.from_file(config)
    gpc.load_config(config)

    # init default process group
    gpc.init_global_dist(rank, world_size, backend, host, port)

    # init process groups for different parallel modes from config
    gpc.init_parallel_groups()

    # set cuda device
    if internlm_accelerator.is_available():
        # if local rank is not given, calculate automatically
        gpc.set_device(local_rank)

    gpc.set_seed(seed)

    warmup_process_group()

    if gpc.is_rank_for_log():
        logger.info(
            f"Distributed environment is initialized, "
            f"data parallel size: {gpc.data_parallel_size}, pipeline parallel size: {gpc.pipeline_parallel_size}, "
            f"tensor parallel size: {gpc.tensor_parallel_size}, weight parallel size: {gpc.weight_parallel_size}",
        )
        if gpc.config.model.get("num_experts", 1) > 1:
            logger.info(
                f"Creating MoE with num_experts: {gpc.config.model.num_experts} | "
                f"expert parallel size: {gpc.expert_parallel_size} | "
                f"number of local experts: {gpc.config.model.num_experts//gpc.expert_parallel_size}"
            )


def launch_from_slurm(
    config: Union[str, Path, Config, Dict],
    host: str,
    port: int,
    backend: str = "nccl",
    seed: int = 1024,
):
    """A wrapper for internlm.launch for SLURM launcher by reading rank and world size from the environment variables
    set by SLURM

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        host (str): The master address for distributed training
        port (str): The master port for distributed training
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
    """
    try:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NPROCS"])
    except KeyError as e:
        raise RuntimeError(f"Could not find {e} in the SLURM environment")

    try_bind_numa(global_rank=rank, world_size=world_size)

    launch(
        config=config,
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
    )


def launch_from_torch(
    config: Union[str, Path, Config, Dict],
    backend: str = "nccl",
    seed: int = 1024,
):
    """A wrapper for internlm.launch for torchrun or torch.distributed.launch by reading rank and world size
    from the environment variables set by PyTorch

    Args:
        config (Union[str, dict, Config]): Config file or config file path are both acceptable
        backend (str, optional): Backend for ``torch.distributed``, defaults to ``nccl``
        seed (int, optional): Specified random seed for every process. Defaults to 1024.
    """
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        host = os.environ["MASTER_ADDR"]
        port = int(os.environ["MASTER_PORT"])
    except KeyError as e:
        raise RuntimeError(f"Could not find {e} in the torch environment")

    try_bind_numa(global_rank=rank, world_size=world_size, local_rank=local_rank)

    launch(
        config=config,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        host=host,
        port=port,
        backend=backend,
        seed=seed,
    )


@llm_timeout(func_name="initialize_distributed_env")
def initialize_distributed_env(
    config: str,
    launcher: str = "slurm",
    master_port: int = 8888,
    seed: int = 1024,
    args_check=True,
    backend: str = "nccl",
):
    """
    Initialize distributed environment for distributed training.

    Args:
        config (str): Config file path.
        launcher (str): Launcher for launching distributed environment, can be slurm or torch. "slurm" by default.
        master_port (str): The master port for distributed training. 8888 by default.
        seed (int, optional): Specified random seed for every process. 1024 by default.
    """
    backend = internlm_accelerator._communication_backend_name

    if launcher == "torch":
        launch_from_torch(config=config, seed=seed, backend=backend)
    elif launcher == "slurm":
        launch_from_slurm(
            config=config,
            host=get_master_node(),
            port=master_port,
            seed=seed,
        )
    else:
        assert launcher in ["slurm", "torch"], "launcher only support slurm or torch"

    if args_check:
        args_sanity_check()


def get_config_value(config, key, defalut):
    try:
        value = config[key]
    except KeyError:
        value = defalut
    return value


def try_bind_numa(global_rank, world_size, local_rank=None):
    # Early return if numa module not available
    if not get_numa:
        if global_rank == 0:
            logger.info(
                "Try bind numa failed! Package import error, if numa is not installed, "
                "please implement: pip install --upgrade py-libnuma, Ref: https://pypi.org/project/py-libnuma/"
            )

    # get numa node number
    try:
        numa_node_num = numa.info.get_max_node() + 1
        # get total gpu number of current node
        nvsmi = nvidia_smi.getInstance()
        total_GPU_per_node = len(nvsmi.DeviceQuery("memory.total")["gpu"])

        # return while total_GPU_per_node is larger than numa_node_num or is not divisible by numa_node_num
        if total_GPU_per_node <= numa_node_num:
            return
        if total_GPU_per_node % numa_node_num != 0:
            return
        # return while the number of processes is smaller than one node GPUs num
        if world_size < total_GPU_per_node:
            return

        if local_rank is None:
            devices_per_node = internlm_accelerator.device_count()
            local_rank = global_rank % devices_per_node

        # compute numa id for each locak rank
        per_numa = total_GPU_per_node // numa_node_num
        numa_id = local_rank // per_numa

        # bind numa node
        schedule.run_on_nodes(numa_id)
        memory.set_membind_nodes(numa_id)
    except Exception:
        return  # try_bind_numa should not raise exception
    else:
        logger.info(f"Rank: {global_rank} success bind process to numa node: {numa_id}")
