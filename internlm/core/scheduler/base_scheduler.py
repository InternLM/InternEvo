#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable

import torch

from internlm.core.engine import Engine


class BaseScheduler(ABC):
    """A basic helper class to control the process of training or evaluation.
    It mainly composes of forward_backward_step for gradient backward and
    optimizer_step for parameters update.
    For the convenience to enable FP16, we aggregate all codes that contain the
    control of FP16 in class schedule.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data and arranges
            them into data and label.
    """

    def __init__(self, data_process_func: Callable = None):
        self.data_process_func = data_process_func
        self._packed_mode = None

    @abstractmethod
    def pre_processing(self, engine: Engine):
        """To perform actions before running the schedule.

        Args:
           engine (internlm.core.Engine): InternLM engine for training and inference.
        """
        pass

    def _load_micro_batch(self, data: Dict, label: torch.Tensor, offset: int, bsz_stride: int):
        """
        For pp, it will cut one fully batch into micro batch in pipeline concept.
        For nopp, it will cut one fully batch into small batch based on gradient accumulate size.

        A special case is that pp uses a 'non-packed-dateset' (such as evaluation dataset),
        so the data of batch is unpacked and 'bsz_stride' is equal to 'micro_bsz'.
        In all other cases 'bsz_stride' should be equal to 1.
        """
        assert isinstance(data, dict) and isinstance(label, torch.Tensor)

        if self.packed_mode:
            micro_batch_data = {k: v[offset : offset + bsz_stride] for k, v in data.items()}
            micro_batch_label = label[offset : offset + bsz_stride]

            assert "cu_seqlens" in micro_batch_data
            assert "indexes" in micro_batch_data
            assert len(micro_batch_data["cu_seqlens"]) == 1
            assert len(micro_batch_data["indexes"]) == 1

            # squeeze
            micro_batch_data["cu_seqlens"] = micro_batch_data["cu_seqlens"].squeeze(0)
            # The indexes are used to indicate the actual position IDs of each token in the packed input.
            indexes = indexes[0]

            # squeeze the dim of micro num.
            micro_batch_data["input_ids"] = micro_batch_data["input_ids"].squeeze(0)
        else:
            micro_batch_data = {k: v[offset] for k, v in data.items()}
            micro_batch_label = label[offset]

            micro_batch_data.pop("cu_seqlens", None)
            micro_batch_data.pop("indexes", None)

        if "DEBUG_DATA_SHAPE" in os.environ:
            attention_mask_shape = (
                micro_batch_data["attention_mask"].shape if "attention_mask" in micro_batch_data else ""
            )
            input_ids_shape = micro_batch_data["input_ids"].shape
            print(
                f"offset: {offset}, bsz_stride:{bsz_stride}, attn_mask: {attention_mask_shape}, input_ids:{input_ids_shape}, label: {micro_batch_label.shape}",
                flush=True,
            )

        return micro_batch_data, micro_batch_label

    @property
    def packed_mode(self):
        return self._packed_mode

    @packed_mode.setter
    def packed_mode(self, packed_mode):
        self._packed_mode = packed_mode

    @abstractmethod
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool,
        return_loss: bool = True,
        return_output_label: bool = True,
    ):
        """The process function over a batch of dataset for training or evaluation.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
            forward_only (bool): If True, the process won't include backward.
            return_loss (bool, optional): If False, the loss won't be returned.
            return_output_label (bool, optional): If False, the output and label won't be returned.
        """
        pass

    @staticmethod
    def _call_engine(engine: Engine, inputs: Any):
        """Calls the engine with the given inputs.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            inputs (Any): The inputs to the engine, can be of type torch.Tensor, list, tuple, or dict.
        """
        if isinstance(inputs, torch.Tensor):
            return engine(inputs)
        elif isinstance(inputs, (list, tuple)):
            return engine(*inputs)
        elif isinstance(inputs, dict):
            return engine(**inputs)
        else:
            raise TypeError(
                f"Expected engine inputs to be of type torch.Tensor, list, tuple, or dict, but got {type(inputs)}"
            )

    @staticmethod
    def _call_engine_criterion(engine: Engine, outputs: Any, labels: Any):
        """Calls the engine's criterion with the given outputs and labels.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            outputs (Any): The outputs from the model, can be of type torch.Tensor, list, tuple, or dict.
            labels (Any): The labels for the outputs, can be of type torch.Tensor, list, tuple, or dict.
        """
        assert isinstance(
            outputs, (torch.Tensor, list, tuple, dict)
        ), f"Expect output of model is (torch.Tensor, list, tuple), got {type(outputs)}"
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        if isinstance(labels, torch.Tensor):
            labels = (labels,)

        if isinstance(outputs, (tuple, list)) and isinstance(labels, (tuple, list)):
            return engine.criterion(*outputs, *labels)
        elif isinstance(outputs, (tuple, list)) and isinstance(labels, dict):
            return engine.criterion(*outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, dict):
            return engine.criterion(**outputs, **labels)
        elif isinstance(outputs, dict) and isinstance(labels, (list, tuple)):
            raise ValueError(f"Expected labels to be a dict when the model outputs are dict, but got {type(labels)}")
        else:
            raise TypeError(
                f"Expected model outputs and labels to be of type torch.Tensor ' \
                '(which is auto-converted to tuple), list, tuple, or dict, ' \
                'but got {type(outputs)} (model outputs) and {type(labels)} (labels)"
            )
