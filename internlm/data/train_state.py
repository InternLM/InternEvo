# Copyright (c) InternLM. All rights reserved.
from internlm.core.context import global_context as gpc
from internlm.core.trainer import TrainState


def get_train_state(dataloader):
    # initialize and resume train state
    if gpc.config.data.type in ["tokenized", "hf"]:
        train_state = TrainState(gpc.config, dataloader.batch_sampler)
    else:
        raise ValueError(f"dataset type {gpc.config.data.type} is not supported")

    return train_state
