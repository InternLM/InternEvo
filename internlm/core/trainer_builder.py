import gc
import logging
import time
from functools import partial
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from internlm.checkpoint.checkpoint_manager import CheckpointManager
from internlm.core.context import global_context as gpc
from internlm.core.context.process_group_initializer import ParallelMode
from internlm.core.trainer import Trainer
from internlm.data.streaming.utils import hf_simple_resume
from internlm.data.train_state import get_train_state
from internlm.eval.evaluation import evaluate_on_val_dls
from internlm.initialize.initialize_trainer import initialize_trainer
from internlm.model.losses.ce_loss import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex
from internlm.monitor.monitor import send_alert_message
from internlm.train.pipeline import (
    get_scheduler_hooks,
    initialize_llm_profile,
    initialize_optimizer,
    initialize_parallel_communicator,
    load_new_batch,
    record_current_batch_training_metrics,
)
from internlm.utils.common import (
    BatchSkipper,
    enable_pytorch_expandable_segments,
    get_current_device,
    get_megatron_flops,
    launch_time,
)
from internlm.utils.gputest import empty_cache_and_diag
from internlm.utils.logger import get_logger
from internlm.utils.megatron_timers import megatron_timer as timer
from internlm.utils.parallel import get_parallel_log_file_name
from internlm.utils.simple_memory_profiler import SimpleMemoryProfiler
from internlm.utils.writer import Writer

# global llm logger
logger = logging.getLogger(__file__)


class TrainerBuilder(Trainer):
    """
    Manage InternEvo training process.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_dl (torch.utils.data.DataLoader): The training data loader.
        val_dls (Optional[Dict[str, torch.utils.data.DataLoader]]): The validation data loaders.
        kwargs: Additional keyward arguments.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dl: DataLoader,
        val_dls: Optional[Dict[str, DataLoader]] = None,
        **kwargs,
    ):
        """
        Initialize InternEvo TrainerBuilder class.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_dl (torch.utils.data.DataLoader): The training data loader.
            val_dls (Optional[Dict[str, torch.utils.data.DataLoader]]): The validation data loaders.
            kwargs: Additional keyward arguments.
        """

        # record very_begining_time
        very_begining_time = time.time()

        # set torch expandable_segments
        enable_pytorch_expandable_segments()

        # get and broadcast current time
        current_time = launch_time()
        objs = [current_time]
        dist.broadcast_object_list(objs, src=0)
        current_time = objs[0].replace(":", ".")
        global logger
        logger = get_logger(
            __file__, launch_time=current_time, job_name=gpc.config.JOB_NAME, file_name=get_parallel_log_file_name()
        )

        # initialize isp communicator
        isp_communicator = initialize_parallel_communicator(model)

        with open(kwargs["config"], "r") as f:
            config_lines = f.readlines()

        # initialize loss function
        criterion = FlashGPTLMLoss(
            parallel_output=gpc.config.model.parallel_output, label_smoothing=gpc.config.loss.label_smoothing
        )

        # initialize and resume train state
        train_state = get_train_state(train_dl)

        # initialize optimizer
        optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

        # initialize checkpoint manager
        ckpt_manager = CheckpointManager(
            ckpt_config=gpc.config.ckpt,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dl=train_dl,
            model_config=gpc.config.model,
            model_config_file="".join(config_lines),
            feishu_address=gpc.config.monitor.alert.feishu_alert_address,
        )

        # load other persistent training states
        ckpt_manager.try_resume_training(train_state, current_time)

        # initialize customed llm writer
        writer = Writer(
            job_name=gpc.config.JOB_NAME,
            launch_time=current_time,
            file_name=get_parallel_log_file_name(),
            tensorboard_folder=gpc.config.tensorboard_folder,
            resume_tb_folder=train_state.resume_tb_folder,  # resume from ckpt.
            step_count=train_state.step_count,  # resume from ckpt.
            config=config_lines,
            logger=logger,
            enable_tb=gpc.config.enable_tb,
            queue_max_length=gpc.config.tensorboard.queue_max_length,
            total_steps=gpc.config.data.total_steps,
        )

        # initialize metric for calculating accuracy and perplexity
        metric = AccPerplex(
            device=get_current_device(),
            tp_pg=gpc.get_group(ParallelMode.TENSOR),
            dp_pg=gpc.get_group(ParallelMode.DATA),
            dataset_types=kwargs["dataset_types"],
        )

        # initialize simple memory profiler
        if kwargs["profiling"]:
            self.memory_profiler = SimpleMemoryProfiler(
                model,
                optimizer.optim,
                log_folder=f"RUN/{gpc.config.JOB_NAME}/{current_time}/memory_trace/rank{gpc.get_global_rank()}_"
                + f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
                + f"wp{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
                + f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}",
            )
        else:
            self.memory_profiler = None

        # initialize batch skipper
        skip_batches = gpc.config.data.skip_batches
        if gpc.config.data.type == "hf" and gpc.config.ckpt.auto_resume:
            skip_batches = hf_simple_resume(train_state)
        self.batch_skipper = BatchSkipper(skip_batches)

        # set TrainerBuilder attributes
        self.very_begining_time = very_begining_time
        self.profiling = kwargs["profiling"]
        self.current_time = current_time
        self.train_dl = train_dl
        self.val_dls = val_dls
        self.train_state = train_state
        self.optimizer = optimizer
        self.beta2_scheduler = beta2_scheduler
        self.isp_communicator = isp_communicator
        self.writer = writer
        self.ckpt_manager = ckpt_manager
        self.metric = metric

        # initialize trainer
        engine, scheduler = initialize_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            beta2_scheduler=beta2_scheduler,
            scheduler_hooks=get_scheduler_hooks(metric, optimizer, isp_communicator),
        )

        super().__init__(engine, scheduler)

    def fit(self):
        """
        Launch InternEvo TrainerBuilder training process.
        """

        self.train()
        train_iter = iter(self.train_dl)

        with initialize_llm_profile(profiling=self.profiling, start_time=self.current_time) as prof:
            # close automatic garbage collection
            gc.disable()
            # start iterating the train data and begin training
            for batch_count in range(self.train_state.batch_count, gpc.config.data.total_steps):
                empty_cache_and_diag(batch_count, interval=gpc.config.data.empty_cache_and_diag_interval)
                # internlm_accelerator.memory._record_memory_history()
                start_time = time.time()
                timer("one-batch").start()

                # load batch data
                batch, train_iter = load_new_batch(
                    train_dl=self.train_dl, train_iter=train_iter, train_state=self.train_state
                )

                # record the consumed samples in training
                self.train_state.batch_count = batch_count
                self.train_state.num_consumed_samples_in_epoch += len(batch[1])
                if self.batch_skipper(batch_count):  # skip this batch
                    if gpc.is_rank_for_log():
                        logger.info(f"Skip batch count:`{batch_count}`...")
                    timer("one-batch").stop()
                    continue

                # zero the grads of parameters
                self.zero_grad()
                # process data
                if batch[0].get("type_ids", None) is not None:
                    self.metric.set_current_type_ids(type_ids=batch[0].pop("type_ids", None))
                # if batch[0].get("cu_seqlens", None) is not None:
                #     metric.set_cu_seqlens(cu_seqlens=batch[0].pop("cu_seqlens", None))

                # do forward and backward
                timer("fwd-bwd").start()

                moe_loss = None
                if hasattr(gpc.config.model, "num_experts"):
                    _, _, loss, moe_loss = self.execute_schedule(
                        batch,
                        forward_only=False,
                        return_loss=True,
                        return_output_label=False,
                    )
                else:
                    _, _, loss = self.execute_schedule(  # pylint: disable=W0632
                        batch,
                        forward_only=False,
                        return_loss=True,
                        return_output_label=False,
                    )
                timer("fwd-bwd").stop()

                if self.isp_communicator and self.isp_communicator.enable_memory_pool:
                    self.isp_communicator.memory_pool.reset_lazy_pools()

                # update parameters, and returns (success_update, grad_norm)
                trainer_result = self.step()
                assert trainer_result is not None

                success_update, grad_norm_groups = trainer_result
                if success_update:  # update parameters successfully
                    self.train_state.step_count += 1
                else:
                    self.train_state.inf_nan_skip_batches += (
                        1  # record the amount of updating parameters unsuccessfully.
                    )
                    if -1 in grad_norm_groups.values() and gpc.is_rank_for_log():  # -1 encodes a specific failure case
                        logger.warning(f"Warning: skip parameter update at step {batch_count}.")
                        send_alert_message(
                            address=gpc.config.monitor.alert.feishu_alert_address,
                            message=f"Warning: skip parameter update at step {batch_count}.",
                        )

                get_tflops_func = partial(
                    get_megatron_flops,
                    checkpoint=gpc.config.model.checkpoint,
                    seq_len=gpc.config.data["seq_len"],
                    hidden_size=gpc.config.model.hidden_size,
                    num_layers=gpc.config.model.num_layers,
                    vocab_size=gpc.config.model.vocab_size,
                    global_batch_size=gpc.config.data.micro_bsz
                    * gpc.config.data.micro_num
                    * gpc.get_world_size(ParallelMode.DATA),
                    global_world_size=gpc.get_world_size(ParallelMode.GLOBAL),
                    mlp_ratio=gpc.config.model["mlp_ratio"],
                )

                # calculate and record the training metrics, eg. loss, accuracy and so on.
                record_current_batch_training_metrics(
                    get_tflops_func=get_tflops_func,
                    logger=logger,
                    writer=self.writer,
                    success_update=success_update,
                    batch_count=batch_count,
                    batch=batch,
                    train_state=self.train_state,
                    optimizer=self.optimizer,
                    beta2_scheduler=self.beta2_scheduler,
                    trainer=self,
                    start_time=start_time,
                    very_begining_time=self.very_begining_time,
                    loss=loss,
                    moe_loss=moe_loss,
                    grad_norm=grad_norm_groups,
                    metric=self.metric,
                )

                timer("one-batch").stop()

                # evaluate on validation data loaders
                if gpc.config.data.valid_every > 0 and self.train_state.step_count % gpc.config.data.valid_every == 0:
                    evaluate_on_val_dls(
                        self,
                        val_dls=self.val_dls,
                        writer=self.writer,
                        logger=logger,
                        step_count=self.train_state.step_count,
                    )

                # checkpoint the training states in specific steps, which is determined by the args "checkpoint_every"
                # # save batch sampler that tracks the true consumed samples
                now_break = self.ckpt_manager.try_save_checkpoint(self.train_state)
                if now_break:
                    break

                if self.memory_profiler is not None:
                    self.memory_profiler.step()

                if batch_count % 2 == 0:
                    prof.step()

                # internlm_accelerator.memory._dump_snapshot(f"my_snapshot_{gpc.get_global_rank()}.pickle")

        self.ckpt_manager.wait_async_upload_finish()
