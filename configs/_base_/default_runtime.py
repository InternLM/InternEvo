# Copyright (c) InternLM. All rights reserved.

cudnn_deterministic = False
cudnn_benchmark = False

enable_tb = True

grad_profiling = dict(
    # calculate layer norms and parameter norms, and show them on tensorboard
    grad_norm_profiling=False,
    # count zero gradients, and show them on tensorboard
    zero_grad_profiling=False,
    # [optional] layers displayed on tensorboard, default: layers=["ScaleColumnParallelLinear"]
    # if not set, display all layers
    layers=["ScaleColumnParallelLinear"],
    vocab_grad_norm_profiling=False,
    interval_steps=5,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)
