import gc
import logging
import time
from contextlib import contextmanager
from functools import partial
from typing import Callable, Iterable, Optional

import torch
import torch.distributed as dist
from torch import nn
from tqdm import tqdm

from internlm.accelerator.abstract_accelerator import get_accelerator
from internlm.checkpoint.checkpoint_manager import CheckpointManager
from internlm.core.context import global_context as gpc
from internlm.core.context.process_group_initializer import ParallelMode
from internlm.core.engine import Engine
from internlm.core.gradient_handler import PipelineSharedModuleGradientHandler
from internlm.core.scheduler import (
    BaseScheduler,
    InterleavedPipelineScheduler,
    NonPipelineScheduler,
    PipelineScheduler,
)
from internlm.core.scheduler.pipeline_scheduler import get_tensor_shape
from internlm.data.train_state import get_train_state
from internlm.data.utils import packed_data_normalizer, unpack_data
from internlm.model.losses.ce_loss import FlashGPTLMLoss
from internlm.model.metrics import AccPerplex, SchedulerMetricHook
from internlm.monitor.monitor import send_alert_message
from internlm.solver.optimizer.hybrid_zero_optim import BaseOptimizer
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


class Trainer:
    """
    Manage the training process.

    Args:
        model: The dmodel to be trained.
        train_dl: The training data loader.
        dataset_types (List[str]): Various data types that will be used in the current training process,
            such as ['en', 'cn', 'code']. The order of the List should be consistent with the type_id specified
            in the dataset. Changed parameters need to be used in conjunction with set_current_type_ids().
        val_dls: Validation data loaders.
        args: Additional arguments and configurations.
    """

    def __init__(
        self,
        model,
        train_dl,
        dataset_types,
        val_dls,
        args,
    ):
        self.very_begining_time = time.time()
        enable_pytorch_expandable_segments()

        # get and broadcast current time
        current_time = launch_time()
        objs = [current_time]
        dist.broadcast_object_list(objs, src=0)
        current_time = objs[0].replace(":", ".")
        self.current_time = current_time
        global logger
        logger = get_logger(
            __file__, launch_time=current_time, job_name=gpc.config.JOB_NAME, file_name=get_parallel_log_file_name()
        )

        train_state = get_train_state(train_dl)
        self.train_state = train_state
        self.train_dl = train_dl

        self.val_dls = val_dls

        # initialize the batch skipper
        if gpc.config.data.type == "hf" and gpc.config.ckpt.auto_resume and train_state.batch_count > 0:
            self.batch_skipper = BatchSkipper(f"0-{train_state.batch_count - 1}")
            train_state.batch_count = 0
            train_state.num_consumed_samples_in_epoch = 0
            if hasattr(train_state, "batch_sampler"):
                train_state.batch_sampler.batch_count = 0
                train_state.batch_sampler.num_consumed_samples_in_epoch = 0
                train_state.batch_sampler_iter = iter(train_state.batch_sampler)
        else:
            self.batch_skipper = BatchSkipper(gpc.config.data.skip_batches)

        self.profiling = args.profiling

        # initialize isp communicator
        isp_communicator = initialize_parallel_communicator(model)

        with open(args.config, "r") as f:
            config_lines = f.readlines()

        optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

        # initialize simple memory profiler
        if args.profiling:
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

        self.optimizer = optimizer
        self.beta2_scheduler = beta2_scheduler
        self.isp_communicator = isp_communicator

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

        # Loading other persistent training states.
        ckpt_manager.try_resume_training(train_state, current_time)

        self.writer = Writer(
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
        self.ckpt_manager = ckpt_manager

        metric = AccPerplex(
            device=get_current_device(),
            tp_pg=gpc.get_group(ParallelMode.TENSOR),
            dp_pg=gpc.get_group(ParallelMode.DATA),
            dataset_types=dataset_types,
        )
        self.metric = metric

        scheduler_hooks = get_scheduler_hooks(metric, optimizer, isp_communicator)
        if isinstance(model, nn.Module):
            # first sync model across dp ranks
            model.to(get_current_device())
        elif isinstance(model, Callable):
            model = model().to(get_current_device())

        # clip grad norm
        clip_grad_norm = gpc.config.hybrid_zero_optimizer.get("clip_grad_norm", 0.0)

        assert isinstance(optimizer, BaseOptimizer), "optimizer must be instance of BaseOptimizer"

        # gradient handler, only support PipelineSharedModuleGradientHandler now
        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            gpc.config.gradient_handler = [dict(type="PipelineSharedModuleGradientHandler")]
        gradient_handler_cfg = gpc.config.get("gradient_handler", [])
        gradient_handlers = []
        assert isinstance(
            gradient_handler_cfg, list
        ), f"gradient_handler must be list but got {type(gradient_handler_cfg)}"
        for config in gradient_handler_cfg:
            if isinstance(config, dict) and config.get("type") == "PipelineSharedModuleGradientHandler":
                handler = PipelineSharedModuleGradientHandler(model=model, optimizer=optimizer)
                gradient_handlers.append(handler)

        if gpc.config.data.use_packed_dataset:
            data_fn = packed_data_normalizer
        elif gpc.config.data.type == "hf":
            data_fn = None
        else:
            data_fn = unpack_data

        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            gpc.config.NUM_MICRO_BATCHES = gpc.config.data.micro_num
            tensor_shape = get_tensor_shape()
            use_interleaved = (
                hasattr(gpc.config, "model")
                and hasattr(gpc.config.model, "num_chunks")
                and gpc.config.model.num_chunks > 1
            )
            scatter_gather = gpc.is_initialized(ParallelMode.TENSOR)
            if use_interleaved:
                if isinstance(model, nn.Sequential):
                    model = nn.ModuleList([model])

                communication_overlap = gpc.config.parallel["pipeline"].get("interleaved_overlap", False)
                scheduler = InterleavedPipelineScheduler(
                    data_process_func=data_fn,
                    num_microbatches=gpc.config.NUM_MICRO_BATCHES,
                    num_chunks=gpc.config.model.num_chunks,
                    dtype=gpc.config.model["dtype"],
                    tensor_shape=tensor_shape,
                    scatter_gather_tensors=scatter_gather,
                    scheduler_hooks=scheduler_hooks,
                    communication_overlap=communication_overlap,
                )
            else:
                scheduler = PipelineScheduler(
                    data_process_func=data_fn,
                    num_microbatches=gpc.config.NUM_MICRO_BATCHES,
                    dtype=gpc.config.model["dtype"],
                    tensor_shape=tensor_shape,
                    scatter_gather_tensors=scatter_gather,
                    scheduler_hooks=scheduler_hooks,
                )
        else:
            scheduler = NonPipelineScheduler(
                data_process_func=data_fn,
                gradient_accumulation_size=gpc.config.data.gradient_accumulation,
                scheduler_hooks=scheduler_hooks,
            )

        criterion = FlashGPTLMLoss(
            parallel_output=gpc.config.model.parallel_output, label_smoothing=gpc.config.loss.label_smoothing
        )
        # initialize engine for trainer
        self._engine = Engine(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            beta2_scheduler=beta2_scheduler,
            criterion=criterion,
            gradient_handlers=gradient_handlers,
            clip_grad_norm=clip_grad_norm,
        )

        # build schedule
        if scheduler is None:
            self._schedule = NonPipelineScheduler()
        else:
            assert isinstance(
                scheduler, BaseScheduler
            ), f"expected schedule to be of type BaseSchedule, but got {type(scheduler)}"
            self._schedule = scheduler

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
        return isinstance(self._schedule, (PipelineScheduler, InterleavedPipelineScheduler))

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

    def fit(self):
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
                    engine=self.engine,
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
                    self.evaluate_on_val_dls(
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

    def evaluate_on_val_dls(
        self,
        val_dls,
        writer,
        logger,
        step_count,
        streaming: bool = False,
    ):
        internlm_accelerator = get_accelerator()
        val_metric = AccPerplex(
            device=get_current_device(),
            tp_pg=gpc.get_group(ParallelMode.TENSOR),
            dp_pg=gpc.get_group(ParallelMode.DATA),
        )
        val_sche_metric_hook = SchedulerMetricHook(metric=val_metric)

        with self.switch_evaluation_mode(metric_hook_list=[val_sche_metric_hook]):
            internlm_accelerator.empty_cache()
            self.eval()
            verbose = gpc.is_rank_for_log()
            data_cfg = gpc.config.data

            for val_name, val_dl in val_dls.items():
                if not streaming and len(val_dl) == 0 and verbose:
                    logger.info(f"Validation dataset: {val_name} is empty")
                    continue

                val_loss = 0
                val_idx = -1
                for val_idx, batch in tqdm(
                    enumerate(val_dl),
                    desc="Val.",
                    total=len(val_dl) if not streaming else None,
                    position=1,
                    disable=not verbose,
                    leave=False,
                ):
                    moe_loss = None
                    with torch.inference_mode():
                        total_val_bsz = len(batch[1])
                        assert total_val_bsz % data_cfg.micro_bsz == 0

                        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
                            with self.switch_evaluation_pipeline_scheduler():
                                # Compatible for non-moe
                                if hasattr(gpc.config.model, "num_experts"):
                                    _, _, loss, moe_loss = self.execute_schedule(
                                        batch, forward_only=True, return_loss=True, return_output_label=False
                                    )
                                else:
                                    _, _, loss = self.execute_schedule(  # pylint: disable=W0632
                                        batch, forward_only=True, return_loss=True, return_output_label=False
                                    )
                        else:
                            if hasattr(gpc.config.model, "num_experts"):
                                _, _, loss, moe_loss = self.execute_schedule(
                                    batch, forward_only=True, return_loss=True, return_output_label=False
                                )
                            else:
                                _, _, loss = self.execute_schedule(  # pylint: disable=W0632
                                    batch, forward_only=True, return_loss=True, return_output_label=False
                                )
                    if verbose:
                        val_loss += loss.item() - moe_loss.item() if moe_loss is not None else loss.item()

                assert val_idx != -1
                dist.barrier()

                val_res = val_metric.get_metric()
                if verbose and (streaming or len(val_dl) != 0):
                    val_loss = val_loss / (val_idx + 1 + 1e-6)
                    infos = {
                        "step": step_count,
                        f"val/{val_name}_loss": val_loss,
                        f"val/{val_name}_acc": val_res["acc"],
                        f"val/{val_name}_plex": val_res["perplexity"],
                    }

                    for key, value in infos.items():
                        writer.add_scalar(key=key, value=value, step=step_count)

                    logger.info(
                        f"Validation on {val_name}: " + " ".join([f"{key}={value}" for key, value in infos.items()])
                    )

            self.train()
            internlm_accelerator.empty_cache()
            dist.barrier()

    @contextmanager
    def switch_evaluation_pipeline_scheduler(self):
        if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
            prev_tensor_shape = self.schedule.tensor_shape
            try:
                self.schedule.tensor_shape = get_tensor_shape()
                yield
            finally:
                self.schedule.tensor_shape = prev_tensor_shape

    @contextmanager
    def switch_evaluation_mode(self, metric_hook_list):
        prev_eval = gpc.is_evaluating
        pre_data_process_func = self.schedule.data_process_func
        prev_metric_hooks = self.schedule._hooks
        try:
            gpc.is_evaluating = True
            self.schedule.data_process_func = None
            self.schedule._hooks = metric_hook_list

            yield
        finally:
            gpc.is_evaluating = prev_eval
            self.schedule.data_process_func = pre_data_process_func
            self.schedule._hooks = prev_metric_hooks


class DeprecatedTrainer:
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
        return isinstance(self._schedule, (PipelineScheduler, InterleavedPipelineScheduler))

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
