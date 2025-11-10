#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import time
from internlm.core.context import global_context as gpc
from internlm.core.trainer_builder import TrainerBuilder
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.model.builder import create_model
from internlm.monitor import internevo_monitor
from internlm.utils.common import parse_args
from internlm.utils.logger import get_logger


@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    # initialize model
    model = create_model()

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # build trainer
    merged_args = {**vars(args), "dataset_types": dataset_types}
    trainer = TrainerBuilder(model, train_dl, val_dls, **merged_args)

    # logging training start time
    training_start_time = time.time()

    # training
    trainer.fit()
    training_end = time.time()
    total_time = training_end - training_start_time
    print(f"----------Training completed in {total_time:.2f}s ({total_time/3600:.2f}h)----------")


if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)
