#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import bisect
import inspect
import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Union

import numpy as np
import torch

import internlm
from internlm.accelerator import get_accelerator
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


def _move_tensor(element):
    if not torch.is_tensor(element):
        # we expecte the data type if a list of dictionaries
        for idx, item in enumerate(element):
            if isinstance(item, dict):
                for key, value in item.items():
                    assert value.device.type == "cpu"
                    item[key] = value.to(get_current_device()).detach()
            elif isinstance(item, list):
                for index, value in enumerate(item):
                    assert value.device.type == "cpu"
                    item[index] = value.to(get_current_device()).detach()
            elif torch.is_tensor(item):
                if item.device.type == "cpu":
                    element[idx] = item.to(get_current_device()).detach()
            else:
                assert False, f"{type(item)}, {item}"
    else:
        assert torch.is_tensor(element), f"element should be of type tensor, but got {type(element)}"
        if element.device.type == "cpu":
            element = element.to(get_current_device()).detach()
    return element


def move_to_device(data):
    if isinstance(data, torch.Tensor):
        data = data.to(get_current_device())
    elif isinstance(data, (list, tuple)):
        data_to_return = []
        for element in data:
            if isinstance(element, dict):
                data_to_return.append({k: _move_tensor(v) for k, v in element.items()})
            else:
                data_to_return.append(_move_tensor(element))
        data = data_to_return
    elif isinstance(data, dict):
        data = {k: _move_tensor(v) for k, v in data.items()}
    else:
        raise TypeError(f"Expected batch data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")
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
            return "indexes" in data[0]
        return False
    elif isinstance(data, dict):
        return "indexes" in data[0]


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def launch_time():
    global CURRENT_TIME
    if not CURRENT_TIME:
        CURRENT_TIME = datetime.now().strftime("%m-%d-%H:%M:%S")
    return CURRENT_TIME


def set_random_seed(seed):
    """Set random seed for reproducability."""
    # It is recommended to use this only when inference.
    if seed is not None:
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        internlm_accelerator.manual_seed(seed)
        # if you are using multi-GPU.
        internlm_accelerator.manual_seed_all(seed)


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
    Singleton Meta.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f"{cls.__name__} is a singleton class and a instance has been created."
        return cls._instances[cls]


def get_megatron_flops(
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
):
    """
    Calc flops based on the paper of Megatron https://deepakn94.github.io/assets/papers/megatron-sc21.pdf
    """

    checkpoint_activations_factor = 4 if checkpoint else 3

    if use_swiglu:
        mlp_ratio = mlp_ratio * 3 / 2

    flops_per_iteration = (
        checkpoint_activations_factor
        * (
            (8 + mlp_ratio * 4) * global_batch_size * seq_len * hidden_size**2
            + 4 * global_batch_size * seq_len**2 * hidden_size
        )
    ) * num_layers + 6 * global_batch_size * seq_len * hidden_size * vocab_size

    tflops = flops_per_iteration / (elapsed_time_per_iter * global_world_size * (10**12))
    return tflops


def enable_pytorch_expandable_segments():
    if torch.__version__ >= "2.1.0" and "cuda" in internlm_accelerator.current_device_name():
        _alloc_setting = "expandable_segments:True"
        if os.getenv("PYTORCH_CUDA_ALLOC_CONF", None) is not None:
            _alloc_setting = os.getenv("PYTORCH_CUDA_ALLOC_CONF") + "," + _alloc_setting
        internlm_accelerator.memory._set_allocator_settings(_alloc_setting)
    else:
        logger.warning("To support the 'expandable_segments' configuration, please upgrade torch to version 2.1.0.")


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
