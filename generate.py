#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import gc
import json
import logging
import os
import shutil
import socket
import traceback
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from internlm.accelerator import get_accelerator
from internlm.apis.inference import SequenceGenerator
from internlm.core.context import global_context as gpc
from internlm.data import build_generation_loader_with_data_type
from internlm.initialize import initialize_distributed_env
from internlm.monitor import initialize_monitor_manager
from internlm.monitor.monitor import monitor_manager as mm
from internlm.train import initialize_model, initialize_parallel_communicator
from internlm.utils.common import (
    enable_pytorch_expandable_segments,
    launch_time,
    parse_args,
)
from internlm.utils.gputest import empty_cache_and_diag
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.parallel import get_parallel_log_file_name
from internlm.utils.storage_manager import init_storage_manager
from tools.load_internlm2_model import get_model_device, merge_pp_within_tp

# global llm logger
logger = logging.getLogger(__file__)
internlm_accelerator = get_accelerator()


def get_latest_subdirectory(folder_path):
    if ":" in folder_path:
        prefix, folder_path = folder_path.split(":", 1)
        prefix += ":"
    else:
        prefix = ""
    subdirectories = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    subdirectories_sorted = sorted(
        subdirectories, key=lambda x: os.path.getctime(os.path.join(folder_path, x)), reverse=True
    )
    if subdirectories_sorted:
        return prefix + os.path.join(folder_path, subdirectories_sorted[0])
    else:
        return None


def main():
    enable_pytorch_expandable_segments()

    generation_config = gpc.config["generation"]

    generation_config = type(
        "",
        (object,),
        {
            "output_folder": Path(generation_config["output_folder"]),
            "ckpt_folder": generation_config["ckpt_folder"]
            if "ckpt_folder" in generation_config
            else get_latest_subdirectory(gpc.config.ckpt.save_ckpt_folder),
            "data_folder": generation_config["data_folder"] if "data_folder" in generation_config else None,
            "batch_size": generation_config.get("batch_size", None),
            "eos_id": generation_config.get("eos_id", 2),
            "bos_id": generation_config.get("bos_id", 1),
            "pad_id": generation_config.get("bos_id", 1),
            "additional_eos_token_list": generation_config.get("additional_eos_token_list", None),
            "max_length": generation_config.get("max_length", 100),
            "do_sample": generation_config.get("do_sample", True),
            "temperature": generation_config.get("temperature", 1.0),
            "num_beams": generation_config.get("num_beams", 1),
            "top_k": generation_config.get("top_k", 50),
            "top_p": generation_config.get("top_p", 1.0),
            "repetition_penalty": generation_config.get("repetition_penalty", 1),
            "length_penalty": generation_config.get("length_penalty", 1.0),
        },
    )

    if not os.path.exists(generation_config.output_folder.absolute()):
        generation_config.output_folder.mkdir(exist_ok=True, parents=True)

    # get and broadcast current time
    current_time = launch_time()
    objs = [current_time]
    torch.distributed.broadcast_object_list(objs, src=0)
    current_time = objs[0].replace(":", ".")
    global logger
    logger = get_logger(
        __file__, launch_time=current_time, job_name=gpc.config.JOB_NAME, file_name=get_parallel_log_file_name()
    )

    try:
        init_storage_manager(False, None, None)
    except AssertionError:
        pass
    except Exception as e:
        raise e

    # initialize model
    model = initialize_model()
    _ = initialize_parallel_communicator(model)
    model = model.model

    state_dict = merge_pp_within_tp(generation_config.ckpt_folder, del_model_prefix=True)
    missing_k, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if len(missing_k) != 0:
        logger.warning(f"Warning: missing keys {missing_k}")
    if len(unexpected_keys) != 0:
        logger.warning(f"Warning: unexpected keys {unexpected_keys}")

    param_dtype = gpc.config.model.dtype
    if isinstance(param_dtype, str):
        try:
            param_dtype = eval(param_dtype)  # pylint: disable=W0123
        finally:
            pass
    if param_dtype == "torch.tf32":
        param_dtype = torch.float32
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    model.to(param_dtype)
    model.eval()
    torch.distributed.barrier()

    data_cfg = gpc.config.data
    if generation_config.data_folder:
        data_cfg.valid_folder = generation_config.data_folder
    gene_dls = build_generation_loader_with_data_type(data_cfg, generation_config)

    sequenece_generator = SequenceGenerator(
        decoder=model,
        eos_token_id=generation_config.eos_id,
        pad_token_id=generation_config.bos_id,
        bos_token_id=generation_config.pad_id,
        additional_eos_token_list=generation_config.additional_eos_token_list,
    )

    ds_count = 0
    gc.disable()
    with torch.inference_mode():
        for ds_name, gene_dl in gene_dls.items():
            if len(gene_dl) == 0:
                logger.info(f"Validation dataset: {ds_name} is empty")
                continue
            timer(f"dataset {ds_count}").start()

            # pylint: disable=forgotten-debug-statement
            all_output_str = []
            # pylint: disable=unused-variable
            for val_idx, (labels, input_ids) in tqdm(
                enumerate(gene_dl),
                desc="generate.",
                total=len(gene_dl),
                position=1,
                leave=False,
            ):
                empty_cache_and_diag(val_idx, interval=gpc.config.data.empty_cache_and_diag_interval)
                input_ids = torch.LongTensor(input_ids)
                if input_ids.size(1) >= generation_config.max_length:
                    logger.warning(
                        f"Not generating for the {val_idx}'th batch, because the sequence "
                        f"length of the batch is {input_ids.size(1)} over the max generation"
                        f"length {generation_config.max_length}"
                    )
                    output_ids = input_ids[:, : generation_config.max_length, ...]
                else:
                    input_ids = input_ids.clamp(min=0, max=gpc.config.model.vocab_size).to(get_model_device(model))
                    output_ids = sequenece_generator.generate(
                        tokens=input_ids,
                        max_length=generation_config.max_length,
                        do_sample=generation_config.do_sample,
                        temperature=generation_config.temperature,
                        num_beams=generation_config.num_beams,
                        top_k=generation_config.top_k,
                        top_p=generation_config.top_p,
                        repetition_penalty=generation_config.repetition_penalty,
                        length_penalty=generation_config.length_penalty,
                    )
                for output in output_ids:
                    not_pad_indices = torch.nonzero(output != generation_config.pad_id)
                    if not_pad_indices.nelement() != 0:
                        sequence = output[not_pad_indices[0] :]
                    else:
                        sequence = output
                    sequence = sequence.tolist()
                    line = str.encode(json.dumps({"tokens": sequence}))
                    all_output_str.append(
                        (
                            line,
                            len(line),
                        )
                    )

            bin_meta, last_position = [], 0
            with open(generation_config.output_folder.joinpath(f"{ds_name}.bin"), "wb") as file:
                for line, token_num in all_output_str:
                    file.write(line)
                    bin_meta.append((last_position, token_num))
                    last_position += len(line)

            with open(generation_config.output_folder.joinpath(f"{ds_name}.bin.meta"), "wb") as file:
                np.save(file, bin_meta)

            timer(f"dataset {ds_count}").stop()
            ds_count += 1


if __name__ == "__main__":
    args = parse_args()
    hostname = socket.gethostname()

    # initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None
    assert "generation" in gpc.config, f"Please set `generation` config in `{args.config}` file"
    assert (
        "output_folder" in gpc.config["generation"]
    ), "Must set `output_folder` for the save folder of generation data"

    # initialize monitor manager context
    with initialize_monitor_manager(
        job_name=gpc.config.JOB_NAME, alert_address=gpc.config.monitor.alert.feishu_alert_address
    ):
        try:
            main()
        except Exception:
            logger.error(
                f"Raise exception from {hostname} with rank id: {gpc.get_global_rank()}\n{traceback.format_exc()}",
            )
            mm.monitor_exception(
                alert_address=gpc.config.monitor.alert.feishu_alert_address, excp_info=traceback.format_exc()
            )

            # internlm_accelerator.memory._dump_snapshot(f"my_snapshot_{gpc.get_global_rank()}.pickle")
        finally:
            # local rank0 delete all files in shm_path, when use shm
            devices_per_node = internlm_accelerator.device_count()
            local_rank = gpc.get_global_rank() % devices_per_node
            if gpc.config.data.use_shm and local_rank == 0:
                if os.path.exists(gpc.config.data.shm_path):
                    shutil.rmtree(gpc.config.data.shm_path)
