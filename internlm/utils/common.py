#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import bisect
import inspect
import os
import random
import threading
from abc import ABC, abstractmethod
from collections import ChainMap
from contextlib import contextmanager
from datetime import datetime
from typing import Union

import numpy as np
import torch

import internlm
from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.utils.logger import get_logger

CURRENT_TIME = None
logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


def parse_args():
    parser = internlm.get_default_parser()
    args = parser.parse_args()

    return args


def get_master_node():
    import subprocess

    if os.getenv("SLURM_JOB_ID") is None:
        raise RuntimeError("get_master_node can only used in Slurm launch!")
    result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
    result = result.decode("utf8").strip()
    return result


def move_norm_to_cuda(norm: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
    if torch.is_tensor(norm) and norm.device.type != internlm_accelerator.get_backend_name():
        norm = norm.to(get_current_device())
    return norm


def move_to_device(data):
    if isinstance(data, torch.Tensor):
        if data.device.type == "cpu":
            data = data.to(get_current_device()).detach()
    elif isinstance(data, (list, tuple)):
        data = [move_to_device(x) for x in data]
    elif isinstance(data, dict):
        data = {k: move_to_device(v) for k, v in data.items()}
    else:
        # other types like scalar, other params, return the value itself.
        return data
    return data


def get_tensor_norm(norm: Union[float, torch.Tensor], move_to_cuda) -> torch.Tensor:
    if isinstance(norm, float):
        norm = torch.Tensor([norm])
    if move_to_cuda:
        norm = norm.to(get_current_device())
    return norm


def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if internlm_accelerator.is_available():
        return torch.device(f"{internlm_accelerator.current_device_name()}")
    else:
        return torch.device("cpu")


def get_batch_size(data):
    if isinstance(data, torch.Tensor):
        return data.size(0)
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], dict):
            return data[0][list(data[0].keys())[0]].size(0)
        return data[0].size(0)
    elif isinstance(data, dict):
        return data[list(data.keys())[0]].size(0)


def check_data_is_packed(data):
    if isinstance(data, torch.Tensor):
        return False
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], dict):
            return "cu_seqlens" in data[0]
        return False
    elif isinstance(data, dict):
        return "cu_seqlens" in data[0]


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def launch_time():
    global CURRENT_TIME
    if not CURRENT_TIME:
        CURRENT_TIME = datetime.now().strftime("%m-%d-%H:%M:%S")
    return CURRENT_TIME


def set_random_seed(seed, cuda_deterministic=False):
    """Set all random seed for reproducability."""
    # It is recommended to use this only when inference.
    assert seed > 0, f"Seed should be a positive integer, but got {seed}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if internlm_accelerator.is_available():
        internlm_accelerator.manual_seed(seed)
        # if you are using multi-GPU.
        internlm_accelerator.manual_seed_all(seed)

    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


@contextmanager
def conditional_context(context_manager, enable=True):
    if enable:
        with context_manager:
            yield
    else:
        yield


class BatchSkipper:
    """
    BatchSkipper is used to determine whether to skip the current batch_idx.
    """

    def __init__(self, skip_batches):
        if skip_batches == "":
            pass
        intervals = skip_batches.split(",")
        spans = []
        if skip_batches != "":
            for interval in intervals:
                if "-" in interval:
                    start, end = map(int, interval.split("-"))
                else:
                    start, end = int(interval), int(interval)
                if spans:
                    assert spans[-1] <= start
                spans.extend((start, end + 1))
        self.spans = spans

    def __call__(self, batch_count):
        index = bisect.bisect_right(self.spans, batch_count)
        return index % 2 == 1


class SingletonMeta(type):
    """
    Thread-safe Singleton Meta with double-checked locking.
    Reference: https://en.wikipedia.org/wiki/Double-checked_locking
    """

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # First check (without locking) for performance reasons
        if cls not in cls._instances:
            # Acquire a lock before proceeding to the second check
            with cls._lock:
                # Second check with lock held to ensure thread safety
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        else:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f"{cls.__name__} is a singleton class and an instance has been created."

        return cls._instances[cls]


def get_megatron_flops(
    elapsed_time_per_iter,
    checkpoint=False,
    selective_checkpoint=False,
    seq_len=2048,
    hidden_size=12,
    num_layers=32,
    vocab_size=12,
    global_batch_size=4,
    global_world_size=1,
    mlp_ratio=4,
    use_swiglu=True,
):
    """
    Calc flops based on the paper of Megatron https://deepakn94.github.io/assets/papers/megatron-sc21.pdf
    """

    checkpoint_activations_factor = 3
    attn_checkpoint_activation_factor = 3

    flops_per_iteration = (
        # wqkv wo mlp
        (checkpoint_activations_factor * ((8 + mlp_ratio * 6) * global_batch_size * seq_len * hidden_size**2))
        * num_layers
        # attn
        + attn_checkpoint_activation_factor * (4 * global_batch_size * seq_len**2 * hidden_size) * num_layers / 2
        # head
        + 6 * global_batch_size * seq_len * hidden_size * vocab_size
    )
    tflops = flops_per_iteration / (elapsed_time_per_iter * global_world_size * (10**12))
    return tflops


def get_megatron_flops_mla(
    elapsed_time_per_iter,
    checkpoint=False,
    seq_len=2048,
    hidden_size=12,
    num_layers=32,
    vocab_size=12,
    global_batch_size=4,
    global_world_size=1,
    mlp_ratio=4,
    use_swiglu=True,
    # moe_kwargs
    topk=1,
    num_experts=0,
    moe_mlp_ratio=0.5,
    first_k_dense=-1,
    num_shared_experts=0,
    # mla_kwargs
    num_heads=32,
    v_head_dim=128,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
):
    """
    Calc flops based on the paper of Megatron https://deepakn94.github.io/assets/papers/megatron-sc21.pdf
    """

    # checkpoint_activations_factor = 4 if checkpoint else 3
    checkpoint_activations_factor = 3

    if use_swiglu:
        mlp_ratio = mlp_ratio * 3 / 2
        moe_mlp_ratio = moe_mlp_ratio * 3 / 2

    # first k dense / dense model
    # sum=2*3*(b*s*mlp_ratio*d^2) = 4*mlp_ratio *bs* d^2
    dense_mlp_flops = mlp_ratio * 4 * global_batch_size * seq_len * hidden_size**2

    if num_experts > 0:
        # moe
        # total tokens: b*s; processed by E experts, with topk
        per_expert_token = global_batch_size * seq_len * topk / num_experts
        per_token_flops = moe_mlp_ratio * 4 * hidden_size**2
        moe_flops = per_expert_token * per_token_flops * num_experts

        if num_shared_experts > 0:
            shared_mlp_flops = moe_mlp_ratio * num_shared_experts * 4 * global_batch_size * seq_len * hidden_size**2
            moe_flops += shared_mlp_flops
    else:
        moe_flops = 0
        first_k_dense = num_layers

    global_tokens = global_batch_size * seq_len
    if q_lora_rank is not None or kv_lora_rank is not None:
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        q_out_dim = q_head_dim * num_heads
    else:
        q_out_dim = hidden_size
    if q_lora_rank is not None:
        # mla
        ## q: (bs, d) @(d, q_lora) -> (bs, q_lora);@(q_lora, q_out) ->(bs, q_out)
        q_flops = 2 * (global_tokens * hidden_size * q_lora_rank + global_tokens * q_out_dim * q_lora_rank)
    else:
        # q: (bs, d) @(d, q_out_dim) -> (bs, q_out_dim)
        q_flops = 2 * global_tokens * hidden_size * q_out_dim

    if kv_lora_rank is not None:
        # kv:
        ## (bs, d) @(d, kv_a_out) -> (bs, kv_a_out)
        ## (bs, kv_lora_rank) @(kv_lora, kv_b_out) -> (bs, kv_b_out)
        kv_a_out = kv_lora_rank + qk_rope_head_dim
        kv_b_out = num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim)

        kv_flops = 2 * (global_tokens * kv_a_out * hidden_size + global_tokens * kv_b_out * kv_lora_rank)

        ## (bs, d) @(d, v_dim) -> (bs, v_dim)
        v_dim = num_heads * v_head_dim
        attn_out_flops = 2 * global_tokens * v_dim * hidden_size
    else:
        kv_flops = 4 * global_tokens * hidden_size**2
        attn_out_flops = 2 * global_tokens * hidden_size**2

    qkv_flops = kv_flops + q_flops

    # attn: 2*2*bds**2
    ## (b, nh, s, hd) @(b, nh, hd, s) -> (b, nh, s, s) : b*nh*s**2*hd -> b*d*s**2
    ## (b, nh, s, s) @ (b, nh, s, hd) -> (b, nh, s, hd): b*s*d*s -> b*d*s**2
    if q_lora_rank is not None:
        attn_hidden_size = num_heads * q_head_dim
        attn_flops = 4 * global_batch_size * seq_len**2 * attn_hidden_size
    else:
        attn_flops = 4 * global_batch_size * seq_len**2 * hidden_size
    attn_flops = attn_flops / 2

    # vocab
    vocab_flops = 6 * global_batch_size * seq_len * hidden_size * vocab_size

    flops_per_iteration = (
        checkpoint_activations_factor
        * (
            dense_mlp_flops * first_k_dense
            + (num_layers - first_k_dense) * moe_flops
            + num_layers * (qkv_flops + attn_flops + attn_out_flops)
        )
        + vocab_flops
    )

    tflops = flops_per_iteration / (elapsed_time_per_iter * global_world_size * (10**12))

    return tflops


def enable_pytorch_expandable_segments():
    if torch.__version__ >= "2.1.0" and AcceleratorType.GPU == internlm_accelerator.get_accelerator_backend():
        _expandable_segments_conf = "expandable_segments:True"
        _alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF", None)
        if _alloc_conf is None:
            _alloc_conf = _expandable_segments_conf
        elif "max_split_size_mb" not in _alloc_conf:
            _alloc_conf = _alloc_conf + "," + _expandable_segments_conf

        internlm_accelerator.memory._set_allocator_settings(_alloc_conf)
    else:
        logger.warning("To support the 'expandable_segments' configuration, please upgrade torch to version 2.1.0.")


def check_cuda_env():
    if os.getenv("CUDA_DEVICE_MAX_CONNECTIONS") is None:
        logger.warning("Env var CUDA_DEVICE_MAX_CONNECTIONS has not be set, please note this!")


class DummyProfile:
    """
    Dummy Profile.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        pass

    def step(self):
        pass


class SchedulerHook(ABC):
    """
    Scheduler Hook.
    """

    @abstractmethod
    def before_forward(self, scheduler, inputs) -> None:
        """Actions before forward"""

    @abstractmethod
    def after_forward(self, scheduler, outputs) -> None:
        """Actions after forward"""

    @abstractmethod
    def before_criterion(self, scheduler, outputs, label) -> None:
        """Actions before criterion"""

    @abstractmethod
    def after_criterion(self, scheduler, loss) -> None:
        """Actions after criterion"""

    @abstractmethod
    def before_backward(self, scheduler, outputs, outputs_grad) -> None:
        """Actions before backward"""

    @abstractmethod
    def after_backward(self, scheduler, inputs_grad) -> None:
        """Actions after backward"""

    @abstractmethod
    def post_helper_func(self, scheduler, outputs, label) -> None:
        """A post helper function"""


class UniqueChainMap(ChainMap):
    """
    UniqueChainMap updates the first mapping containing a given key when assigning a value.
    If the key is not found, it adds the key-value pair to the first mapping.
    """

    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value
