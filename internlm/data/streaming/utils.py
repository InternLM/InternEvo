#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from internlm.core.context import global_context as gpc


# simple auto_resume for huggingface streaming dataloader
def hf_simple_resume(train_state):
    skip_batches = gpc.config.data.get("skip_batches", "")
    if train_state.batch_count > 0:
        assert skip_batches == "", "skip_batches should be empty when huggingface dataloader resume from ckpts"
        skip_batches = f"0-{train_state.batch_count - 1}"
        train_state.batch_count = 0
        train_state.num_consumed_samples_in_epoch = 0
        if hasattr(train_state, "batch_sampler"):
            train_state.batch_sampler.batch_count = 0
            train_state.batch_sampler.num_consumed_samples_in_epoch = 0
            train_state.batch_sampler_iter = iter(train_state.batch_sampler)
    return skip_batches
