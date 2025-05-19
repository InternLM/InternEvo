#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

import json
import math
import os
import time
from collections import deque
from typing import Iterable, List, Optional

from torch.utils.data import DataLoader

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine
from internlm.core.parallel.comm import ISPCommunicatorSchedulerHook
from internlm.core.scheduler import BaseScheduler, NonPipelineScheduler
from internlm.data.utils import unpack_type_ids
from internlm.model.model_ops.metrics import SchedulerMetricHook
from internlm.monitor import monitor_manager as mm
from internlm.utils.common import SchedulerHook, set_env_var
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.timeout import llm_timeout


class TrainState:
    """
    The TrainState class is used to record the current state of training.

    Args:
        train_dl (DataLoader): The DataLoader object used for training.
    """

    def __init__(self, config, batch_sampler) -> None:
        """
        Args:
            config (Config): internlm config
            batch_sampler (torch.utils.data.Sampler): Because the dataloader loading is
            asynchronous and prefetched, the batch_sampler state maintained inside the
            dataloader are faster then the actual training progress, so we copy the
            batch_sampler as the anchor point of ckpt reload.
        """
        # The number of batches produced by the data iterator
        self.batch_count: int = 0
        # Used to store the number of samples consumed in the current epoch
        self.num_consumed_samples_in_epoch: int = 0
        # Total number of tokens consumed
        self.num_consumed_tokens: int = 0
        # Number of batches skipped due to inf or nan values
        self.inf_nan_skip_batches: int = 0
        # Records the number of updates, skipped batches and inf batches are not counted
        self.step_count: int = 0

        # Total step count
        self.total_steps: int = config.data.total_steps

        # resume tensorboard folder, need load from checkpoint or set manually.
        self.resume_tb_folder = config.resume_tb_folder

        self.tensorboard_folder = config.tensorboard_folder

        # learning rate
        self.lr = config.adam.lr

        # smapler state
        if batch_sampler is not None:
            self.init_batch_sampler(batch_sampler)

        # tgs statistic
        self.tgs_statistic = {
            "sum_step": 0,
            "sum_tg": 0,
            "total_time": 0,
            "sum_last_tg_10": 0,
            "sum_last_time_10": 0,
            "sum_last_tg_50": 0,
            "sum_last_time_50": 0,
            "SMA_tg_50": 0,
            "SMA_time_50": 0,
            "SMA_tg_50_list": deque(),
            "SMA_time_50_list": deque(),
            "sum_tgs": 0,
            "last_tgs_10": 0,
            "last_tgs_50": 0,
        }

    def init_batch_sampler(self, batch_sampler):
        """
        Args:
            batch_sampler (torch.utils.data.Sampler): sampler.
        """
        # make a copy of batch_sampler.
        self.batch_sampler = batch_sampler.copy()
        # Iterator for the batch sampler
        self.batch_sampler_iter = iter(self.batch_sampler)

    def __str__(self) -> str:
        """Returns a string representation of the training state in JSON format."""
        info = {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
        }

        return json.dumps(info, indent=4, sort_keys=True)

    def load_state_dict(self, other_stuffs):
        """
        Resumes training from a checkpoint.

        Args:
            other_stuffs (dict): Other information needed to resume training.
        """
        self.num_consumed_samples_in_epoch = other_stuffs["num_consumed_samples_in_epoch"]
        self.num_consumed_tokens = other_stuffs["num_consumed_tokens"]
        self.inf_nan_skip_batches = other_stuffs["inf_nan_skip_batches"]

        # Because the ckpt save occurs after updating 'step_count',
        # there is no need to increment 'step_count' here (Does our step count start from 0 ?),
        # However, 'batch_count' is updating before ckpt storage, so it need to inc 1 when resume.
        self.batch_count = other_stuffs["batch_count"] + 1  # here you need to shift a batch backward
        self.step_count = other_stuffs.get("step_count", self.batch_count)

        # resume tensorboard from older tensorboard_folder
        self.resume_tb_folder = other_stuffs.get("tensorboard_folder", None)

    def state_dict(self):
        if os.environ.get("CLUSTER_NAME") == "volc" and os.environ.get("petrelfs_tb_path") is not None:
            tensorboard_folder = os.path.join(os.environ["petrelfs_tb_path"], os.environ["MLP_TASK_ID"])
        else:
            tensorboard_folder = self.tensorboard_folder
        return {
            "batch_count": self.batch_count,
            "num_consumed_samples_in_epoch": self.num_consumed_samples_in_epoch,
            "num_consumed_tokens": self.num_consumed_tokens,
            "inf_nan_skip_batches": self.inf_nan_skip_batches,
            "step_count": self.step_count,
            "tensorboard_folder": tensorboard_folder,
        }


class Trainer:
    """This is a class tending for easy deployments of users' training and evaluation instead of
    writing their own scripts.

    Args:
        engine (:class:`Engine`): Engine responsible for the process function.
        schedule (:class:`BaseScheduler`, optional): Runtime schedule. Defaults to None.
    """

    def __init__(
        self,
        engine: Engine,
        schedule: Optional[BaseScheduler] = None,
    ):
        """Initializes the Trainer class.

        Args:
            engine (Engine): The engine responsible for the process function.
            schedule (Optional[BaseScheduler], optional): The runtime schedule. Defaults to None.
        """
        self._engine = engine

        # build schedule
        if schedule is None:
            self._schedule = NonPipelineScheduler()
        else:
            assert isinstance(
                schedule, BaseScheduler
            ), f"expected schedule to be of type BaseSchedule, but got {type(schedule)}"
            self._schedule = schedule

        self._schedule.pre_processing(self._engine)

    @property
    def engine(self):
        """Returns the engine that responsible for managing the training and evaluation process."""
        return self._engine

    @property
    def schedule(self):
        """Returns the runtime scheduler."""
        return self._schedule

    @property
    def uses_pipeline(self):
        """Returns whether the pipeline parallel is used or not."""
        return not isinstance(self._schedule, (NonPipelineScheduler, BaseScheduler))

    def train(self):
        """Sets the model to training mode."""
        self._engine.train()

    def eval(self):
        """Sets the model to evaluation mode."""
        self._engine.eval()

    def zero_grad(self):
        """Sets the gradient of all parameters in the model to zero."""
        self._engine.zero_grad()

    def step(self):
        """Executes the parameter update step."""
        return self._engine.step()

    def execute_schedule(self, data_iter: Iterable, **kwargs):
        """Runs the forward, loss computation, and backward for the model.
        Returns a tuple of (output, label, loss).

        Args:
            data_iter (Iterable): The data iterator.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss, moe_loss).
        """
        return self._schedule.forward_backward_step(self._engine, data_iter, **kwargs)


def get_scheduler_hooks(metric, zero_optim, isp_communicator_wrapper) -> List[SchedulerHook]:
    scheduler_hooks: List[SchedulerHook] = []

    if metric is not None:
        scheduler_hooks.append(
            SchedulerMetricHook(
                metric=metric,
                skip=(
                    gpc.is_using_parallel_mode(ParallelMode.PIPELINE)
                    and hasattr(gpc.config.model, "num_chunks")
                    and gpc.config.model.num_chunks > 1
                    and gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
                ),
            ),
        )

    if isp_communicator_wrapper is not None:
        for isp_communicator in isp_communicator_wrapper.isp_communicators:
            if isp_communicator is not None and isp_communicator.overlap:
                scheduler_hooks.append(ISPCommunicatorSchedulerHook(isp_communicator, zero_optim))

    return scheduler_hooks


@llm_timeout(func_name="load_new_batch")
def load_new_batch(train_dl: DataLoader, train_iter: Iterable, train_state: TrainState):
    """
    Load and return the new batch data based on training data loader.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader for training.
        train_iter (Iterable): Data iterator from which get a batch of data, obtained by calling iter(dataloader).
        train_state (TrainState): Current training state.

    Returns: A batch data and the updated train_iter.
    """

    timer("batch-gen").start()
    try:
        batch = next(train_iter)  # structure is ({'input_ids': Tensor, 'cu_seqlens': Tensor}, Tensor)
        if hasattr(train_state, "batch_sampler_iter"):
            next(train_state.batch_sampler_iter)
    except StopIteration:
        train_iter = iter(train_dl)
        batch = next(train_iter)
        train_state.num_consumed_samples_in_epoch = 0
        if hasattr(train_state, "batch_sampler"):
            train_state.batch_sampler.batch_count = 0
            train_state.batch_sampler.num_consumed_samples_in_epoch = 0
            train_state.batch_sampler_iter = iter(train_state.batch_sampler)
            next(train_state.batch_sampler_iter)
    timer("batch-gen").stop()

    if batch[0].get("type_ids", None) is not None:
        # if use_packed_dataset is False, we need to unpack type_ids
        if not gpc.config.data.use_packed_dataset:
            batch[0]["type_ids"] = unpack_type_ids(batch[0]["type_ids"], batch[0]["cu_seqlens"])

    return batch, train_iter


@llm_timeout(func_name="record_current_batch_training_metrics")
def record_current_batch_training_metrics(
    get_tflops_func,
    logger,
    writer,
    success_update,
    batch_count,
    batch,
    train_state,
    optimizer,
    beta2_scheduler,
    engine,
    start_time,
    very_begining_time,
    loss,
    moe_loss,
    grad_norm,
    metric,
):
    """
    Print some training metrics of current batch.
    """

    set_env_var(key="LAST_ACTIVE_TIMESTAMP", value=int(time.time()))

    timer.store_last_timers()
    if success_update in (0, True):
        train_state.num_consumed_tokens += batch[1].nelement() * gpc.get_world_size(ParallelMode.DATA)
    if gpc.is_no_pp_or_last_stage():
        acc_perplex = metric.get_metric()

    if success_update and gpc.is_rank_for_log():
        lr = optimizer.param_groups[0]["lr"]
        if hasattr(engine.optimizer, "grad_scaler"):
            scaler = engine.optimizer.grad_scaler._scale.item()
        elif hasattr(engine.optimizer.optim, "grad_scaler"):
            scaler = engine.optimizer.optim.grad_scaler._scale.item()

        num_tokens_in_batch = batch[1].nelement()
        real_num_tokens = math.ceil(acc_perplex.pop("real_token_num") / gpc.get_world_size(ParallelMode.GLOBAL))
        num_samples_in_batch = sum([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        max_length_in_batch = max([(b[1:] - b[:-1]).max().item() for b in batch[0]["cu_seqlens"]])
        max_samples_in_batch = max([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        min_samples_in_batch = min([len(b) - 1 for b in batch[0]["cu_seqlens"]])
        time_cost = time.time() - start_time
        tk_per_gpu = round(
            num_tokens_in_batch * gpc.get_world_size(ParallelMode.DATA) / gpc.get_world_size(ParallelMode.GLOBAL),
            4,
        )
        tgs_statistic = train_state.tgs_statistic
        tgs_statistic["sum_step"] += 1
        tgs_statistic["sum_tg"] += tk_per_gpu
        tgs_statistic["total_time"] = time.time() - very_begining_time
        tgs_statistic["sum_last_tg_10"] += tk_per_gpu
        tgs_statistic["sum_last_time_10"] += time_cost
        tgs_statistic["sum_last_tg_50"] += tk_per_gpu
        tgs_statistic["sum_last_time_50"] += time_cost
        tgs_statistic["SMA_tg_50"] += tk_per_gpu
        tgs_statistic["SMA_time_50"] += time_cost
        tgs_statistic["SMA_tg_50_list"].append(tk_per_gpu)
        tgs_statistic["SMA_time_50_list"].append(time_cost)
        if tgs_statistic["sum_step"] > 50:
            tgs_statistic["SMA_tg_50"] -= tgs_statistic["SMA_tg_50_list"][0]
            tgs_statistic["SMA_time_50"] -= tgs_statistic["SMA_time_50_list"][0]
            tgs_statistic["SMA_tg_50_list"].popleft()
            tgs_statistic["SMA_time_50_list"].popleft()

        last_tgs_1 = round(tk_per_gpu / time_cost, 2)
        tgs_statistic["sum_tgs"] += last_tgs_1

        if tgs_statistic["sum_step"] % 10 == 0:
            tgs_statistic["last_tgs_10"] = round(tgs_statistic["sum_last_tg_10"] / tgs_statistic["sum_last_time_10"], 2)
            tgs_statistic["sum_last_tg_10"] = 0
            tgs_statistic["sum_last_time_10"] = 0

        if tgs_statistic["sum_step"] % 50 == 0:
            tgs_statistic["last_tgs_50"] = round(tgs_statistic["sum_last_tg_50"] / tgs_statistic["sum_last_time_50"], 2)
            tgs_statistic["sum_last_tg_50"] = 0
            tgs_statistic["sum_last_time_50"] = 0

        last_tgs_10 = tgs_statistic["last_tgs_10"]
        last_tgs_50 = tgs_statistic["last_tgs_50"]

        tgs_all = round(tgs_statistic["sum_tg"] / tgs_statistic["total_time"], 2)
        tgs_avg = round(tgs_statistic["sum_tgs"] / tgs_statistic["sum_step"], 2)
        tgs_SMA = round(tgs_statistic["SMA_tg_50"] / tgs_statistic["SMA_time_50"], 2)

        tflops = get_tflops_func(time_cost)

        tgs_origin = round(
            num_tokens_in_batch
            * gpc.get_world_size(ParallelMode.DATA)
            / gpc.get_world_size(ParallelMode.GLOBAL)
            / time_cost,
            2,
        )

        real_tgs = round(
            real_num_tokens / time_cost,
            2,
        )

        infos = {
            "tflops": tflops,
            "step": batch_count,
            "loss": loss - moe_loss if moe_loss is not None else loss,
            "real_tgs": real_tgs,
            "tgs (tokens/gpu/second)": tgs_origin,
            "tgs/last_tgs_1": last_tgs_1,
            "tgs/tgs_all": tgs_all,
            "tgs/tgs_avg": tgs_avg,
            "tgs/tgs_SMA": tgs_SMA,
            "tgs/last_tgs_10": last_tgs_10,
            "tgs/last_tgs_50": last_tgs_50,
            "lr": lr,
            "loss_scale": scaler,
            "grad_norm": grad_norm,
        }
        if moe_loss is not None:
            infos["moe_loss"] = moe_loss

        infos["micro_num"] = len(batch[1])
        infos["num_consumed_tokens"] = train_state.num_consumed_tokens
        infos["inf_nan_skip_batches"] = train_state.inf_nan_skip_batches
        infos["num_samples_in_batch"] = num_samples_in_batch  # the number of batches which have the most samples
        infos["largest_length"] = max_length_in_batch  # the longest input
        infos["largest_batch"] = max_samples_in_batch  # the batch with the most samples
        infos["smallest_batch"] = min_samples_in_batch
        infos["adam_beta2"] = beta2_scheduler.get_beta2()

        fwd_bwd_time = round(timer("fwd-bwd").elapsed(), 2)
        infos["fwd_bwd_time"] = fwd_bwd_time
        bwd_time = round(timer("bwd").elapsed(), 2)
        infos["bwd_time"] = bwd_time

        for key, value in acc_perplex.items():
            infos[key] = value

        line = ""
        for key, value in infos.items():
            line += f"{key}={value} "
            if isinstance(value, dict):
                writer.add_scalars(key=key, value=value, step=train_state.step_count)
            else:
                writer.add_scalar(key=key, value=value, step=train_state.step_count)

        logger.info(line)

        # if loss spike occurs, send alert info to feishu
        mm.monitor_loss_spike(
            alert_address=gpc.config.monitor.alert.feishu_alert_address,
            step_count=batch_count,
            cur_step_loss=loss,
        )
