#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import nn

def calculate_z_loss(*args) -> torch.Tensor:
    if len(args) == 3:
            # residual is to match prenorm
        logits, _, labels = args
    elif len(args) == 2:
        # When using postnorm
        logits, labels = args
    else:
        raise RuntimeError(f"The number of criterion inputs are:{len(args)}")
    shift_logits = logits.contiguous().view(-1, logits.size(-1))
    shift_labels = labels.contiguous().view(-1)

    valid_mask = shift_labels >= 0
    z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
    return z_loss
