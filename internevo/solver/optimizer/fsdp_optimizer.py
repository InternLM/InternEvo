#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
from typing import Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from internevo.accelerator import get_accelerator
from internevo.core.context import Config, ParallelMode
from internevo.core.context import global_context as gpc
from internevo.solver.optimizer.base_optimizer import BaseOptimizer
from internevo.solver.optimizer.utils import (
    DynamicGradScaler,
    get_norm,
    release_param_grad,
)
from internevo.utils.common import get_tensor_norm, move_norm_to_cuda
from internevo.utils.logger import get_logger

try:
    from torch.distributed.tensor import DTensor

    DTENSOR_SUPPORTED = True
except (ModuleNotFoundError, ImportError):
    DTENSOR_SUPPORTED = False

logger = get_logger(__file__)

inf = math.inf

internlm_accelerator = get_accelerator()


def compute_norm(
    gradients: Iterable[torch.Tensor],
    parameters: Iterable[torch.Tensor],
) -> float:
    """Get L2 norm
    Arguments:
        gradients (Iterable[Tensor]): The gradient value.
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to.

    Returns:
        Total norm of the parameters, need total_norm**(1/norm) before using.
    """

    enable_cuda_kernels = gradients[0].device.type != "cpu"

    # Calculate norm.
    tensor_parallel_grads = [g.data.float() for g, _ in zip(gradients, parameters)]
    tensor_parallel_norm = get_norm(tensor_parallel_grads, float(2), enable_cuda_kernels)
    # If norm is type of float, then we convert them into torch.Tensor.
    total_norm = get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
    # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
    if not enable_cuda_kernels:
        total_norm = move_norm_to_cuda(total_norm)

    if DTENSOR_SUPPORTED and isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=gpc.get_group(ParallelMode.GLOBAL))

    if torch.is_tensor(total_norm):
        total_norm = total_norm.item()

    # Scale.
    if total_norm == float("inf") or total_norm == -float("inf"):
        total_norm = -1

    if math.isnan(total_norm):
        total_norm = -2

    return total_norm


class FSDPadaptOptimizer(BaseOptimizer):
    """
    optimizer for Pytorch FSDP if 'parallel.fsdp' is not None in config file
    reserve some necessary components of hybird-optim:
        grad_scaler;
        grad_clip and unscale;
        state_dict and load_state_dict
    """

    def __init__(
        self,
        optimizer: Optimizer,
        grad_scal_cfg: Config = None,
        zero_cfg: Config = None,
    ):
        super().__init__(optim=optimizer)

        # gradient scaler
        self.grad_scaler = DynamicGradScaler(
            initial_scale=grad_scal_cfg.fp16.initial_scale,
            min_scale=grad_scal_cfg.fp16.min_scale,
            growth_factor=grad_scal_cfg.growth_factor,
            backoff_factor=grad_scal_cfg.backoff_factor,
            growth_interval=grad_scal_cfg.fp16.growth_interval,
            hysteresis=grad_scal_cfg.hysteresis,
            max_scale=grad_scal_cfg.max_scale,
            dtype=gpc.config.model.dtype,
        )

        # clip gradient
        self._clip_grad_norm = zero_cfg.clip_grad_norm

        # fp16 and fp32 params
        # fp16 share mem space with model.FlatParam, fp32 share mem space with optim.param_group
        self._fp16_param_groups = dict()
        self._fp32_param_tensor_groups = dict()

        # init fp16 and fp32 params
        for group_idx, param_group in enumerate(self.optim.param_groups):
            group_params = param_group["params"]

            # fp16 FlatParam storage
            self._fp16_param_groups[group_idx] = group_params

            # create copy of fp32 weight
            fp32_tensor_param = [param.data.float() for param in group_params]
            self._fp32_param_tensor_groups[group_idx] = fp32_tensor_param

            # replace
            param_group["params"] = fp32_tensor_param

    @property
    def loss_scale(self):
        return self.grad_scaler.scale

    def backward(self, loss, retain_graph=False):
        loss = self.loss_scale * loss
        loss.backward(retain_graph=retain_graph)

    def _compute_norm_with_fsdp_flatten(self, group_id):
        params = [p for p in self._fp16_param_groups[group_id] if p.untyped_storage().size() != 0]
        gradients = [p.grad for p in params if p.untyped_storage().size() != 0]

        norm_group = 0
        if len(params) <= 0 or len(gradients) <= 0:
            return norm_group
        norm_group = compute_norm(gradients=gradients, parameters=params)

        return norm_group

    def zero_grad(self):
        for _, param_group in self._fp16_param_groups.items():
            for param in param_group:
                param.grad = None

    def step(self):
        # compute norm
        found_inf = False
        norm_groups = {}
        for group_idx in range(len(self.param_groups)):
            group_name = self.param_groups[group_idx]["name"] if "name" in self.param_groups[group_idx] else "default"
            group_name = f"{group_idx}_{group_name}"
            norm_group = self._compute_norm_with_fsdp_flatten(group_idx)
            if norm_group == -1:
                found_inf = True
            norm_groups[group_name] = norm_group

        loss_scale = float(self.loss_scale.item())  # backup
        self.grad_scaler.update(found_inf)
        if found_inf:
            if gpc.is_rank_for_log():
                logger.warning("Overflow occurs, please check it.")
            self.zero_grad()
            return False, norm_groups

        # get the global norm
        global_norm_groups = {}
        if self._clip_grad_norm > 0:
            for group_name, norm in norm_groups.items():
                global_norm_groups[group_name] = norm**0.5

        # create gradient for fp32 params
        for group_idx in range(len(self.param_groups)):
            if len(self._fp32_param_tensor_groups[group_idx]) <= 0:
                continue
            dtype = self._fp32_param_tensor_groups[group_idx][0].dtype
            fp16_params = [p for p in self._fp16_param_groups[group_idx] if p.untyped_storage().size() != 0]
            grad_fp32 = [p.grad.to(dtype) for p in fp16_params]

            device = self._fp32_param_tensor_groups[group_idx][0].device
            nonzero_fp32 = [p for p in self._fp32_param_tensor_groups[group_idx] if p.untyped_storage().size() != 0]
            for p, g in zip(nonzero_fp32, grad_fp32):
                p.grad = g.to(device)

        # unscale
        self._unscale_and_clip_grads(list(global_norm_groups.values()), loss_scale)

        self.optim.step()
        self.zero_grad()

        for group_idx in range(len(self._fp16_param_groups)):
            fp16_params = [p for p in self._fp16_param_groups[group_idx] if p.untyped_storage().size() != 0]
            fp32_tensor_params = [
                p for p in self._fp32_param_tensor_groups[group_idx] if p.untyped_storage().size() != 0
            ]
            # release fp32 grad
            release_param_grad(fp32_tensor_params)
            # update fp16 param
            for p, q in zip(fp16_params, fp32_tensor_params):
                p.data.copy_(q)

        for group_name, global_norm in global_norm_groups.items():
            global_norm_groups[group_name] = global_norm / loss_scale
        return True, global_norm_groups

    def clip_grad_norm(self, model, max_norm):
        # will conduct in the step()
        pass

    #########################
    # utils from hybirdzero #
    #########################

    def _unscale_and_clip_grads(self, total_norm_groups, loss_scale):
        # compute combined scale factor for this group
        combined_scale_groups = []

        if self._clip_grad_norm > 0.0:
            # norm is in fact norm*scale
            for group_id, total_norm in enumerate(total_norm_groups):
                combined_scale_groups.append(loss_scale)
                clip = ((total_norm / loss_scale) + 1e-6) / self._clip_grad_norm
                if clip > 1.0:
                    combined_scale_groups[group_id] = clip * loss_scale

        for group_id, param in self._fp32_param_tensor_groups.items():
            for p in param:
                if p.untyped_storage().size() != 0:
                    p.grad.data.mul_(1.0 / combined_scale_groups[group_id])

    def state_dict(self):
        states = {}
        grad_scaler = self.grad_scaler.state_dict()
        states["grad_scaler"] = grad_scaler
        optim_states = self.optim.state_dict()
        states["base_optim_states"] = optim_states

        flat_fp32_weights = {}
        for group_idx, param in self._fp32_param_tensor_groups.items():
            flat_fp32_weights[group_idx] = param
        states["flat_fp32_weights"] = flat_fp32_weights

        return states

    def load_state_dict(self, states):
        assert "grad_scaler" in states, "Not found grad_scaler state!"
        grad_scaler = states["grad_scaler"]
        self.grad_scaler.load_state_dict(grad_scaler)
        optim_states = states["base_optim_states"]

        self.optim.load_state_dict(optim_states)

        # load fp32 optimizer weight
        flat_fp32_weights = states["flat_fp32_weights"]
        assert set(flat_fp32_weights.keys()) == set(self._fp32_param_tensor_groups)
        for group_idx, param in flat_fp32_weights.items():
            self_param = self._fp32_param_tensor_groups[group_idx]
            assert len(self_param) == len(
                param
            ), f"The number of flat tensor is inconsistent, {len(self_param)} != {len(param)}"
            for p, q in zip(self_param, param):
                p.data.copy_(q.data)

        # load fp16 model weight
        for group_idx, param in flat_fp32_weights.items():
            fp16_param = self._fp16_param_groups[group_idx]
            fp32_param = self._fp32_param_tensor_groups[group_idx]
            for p, q in zip(fp16_param, fp32_param):
                p.data.copy_(q.data)
