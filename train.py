#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import os
import shutil
import socket
import traceback

from internlm.accelerator import get_accelerator
from internlm.core.context import global_context as gpc
from internlm.core.trainer import Trainer
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.monitor.monitor import initialize_monitor_manager
from internlm.monitor.monitor import monitor_manager as mm
from internlm.train import initialize_model
from internlm.utils.common import parse_args

# global llm logger
logger = logging.getLogger(__file__)
internlm_accelerator = get_accelerator()


def main(args):

    # initialize model
    model = initialize_model()

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # setup trainer
    trainer = Trainer(model, train_dl, dataset_types, val_dls, args)

    # train
    trainer.fit()


if __name__ == "__main__":
    args = parse_args()
    hostname = socket.gethostname()

    # initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # initialize monitor manager context
    with initialize_monitor_manager(
        job_name=gpc.config.JOB_NAME, alert_address=gpc.config.monitor.alert.feishu_alert_address
    ):
        try:
            main(args)
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
