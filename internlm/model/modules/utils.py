#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn.functional as F

from internlm.utils.logger import get_logger

logger = get_logger(__file__)


def try_import_RMSNorm():
    """
    Try import MixFusedRMSNorm from apex, if failed, return our RMSNorm

    """
    try:
        from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as RMSNorm

        return RMSNorm
    except ModuleNotFoundError:
        logger.warning("The torch implementation for MixFusedRMSNorm is slower than apex. Please note this!")
        from internlm.model.norm import RMSNormTorch as RMSNorm

        return RMSNorm


def is_moe_param(param: torch.Tensor) -> bool:
    if hasattr(param, "is_expert") and param.is_expert:
        return True
    return False


def Silu(w1_o, w2_o):
    return F.silu(w1_o) * w2_o


Silu = torch.jit.script(Silu)
