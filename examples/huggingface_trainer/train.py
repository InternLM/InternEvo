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
from internlm.monitor import internevo_monitor
from internlm.train import initialize_model
from internlm.utils.common import enable_pytorch_expandable_segments, parse_args

@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    very_begining_time = time.time()
    enable_pytorch_expandable_segments()

    # initialize model
    model = initialize_model()

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # initialize kwargs
    kwargs = vars(args) | {"dataset_types": dataset_types, "very_begining_time": very_begining_time}

    # build trainer
    trainer = TrainerBuilder(model, train_dl, val_dls, **kwargs)

    # training
    trainer.fit()


if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)