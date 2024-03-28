from contextlib import contextmanager

import torch
import torch.distributed as dist
from tqdm import tqdm

from internlm.accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.scheduler.pipeline_scheduler import get_tensor_shape
from internlm.model.metrics import AccPerplex, SchedulerMetricHook
from internlm.utils.common import get_current_device

internlm_accelerator = get_accelerator()


@contextmanager
def switch_evaluation_pipeline_scheduler(trainer):
    if gpc.is_using_parallel_mode(ParallelMode.PIPELINE):
        prev_tensor_shape = trainer.schedule.tensor_shape
        try:
            trainer.schedule.tensor_shape = get_tensor_shape()
            yield
        finally:
            trainer.schedule.tensor_shape = prev_tensor_shape


@contextmanager
def switch_evaluation_mode(trainer, metric_hook_list):
    prev_eval = gpc.is_evaluating
    pre_data_process_func = trainer.schedule.data_process_func
    prev_metric_hooks = trainer.schedule._hooks
    try:
        gpc.is_evaluating = True
        trainer.schedule.data_process_func = None
        trainer.schedule._hooks = metric_hook_list

        yield
    finally:
        gpc.is_evaluating = prev_eval
        trainer.schedule.data_process_func = pre_data_process_func
        trainer.schedule._hooks = prev_metric_hooks


def evaluate_on_val_dls(
    trainer,
    val_dls,
    writer,
    logger,
    step_count,
    update_panel: bool = False,
    streaming: bool = False,
):
    val_metric = AccPerplex(
        device=get_current_device(),
        tp_pg=gpc.get_group(ParallelMode.TENSOR),
        dp_pg=gpc.get_group(ParallelMode.DATA),
    )
    val_sche_metric_hook = SchedulerMetricHook(metric=val_metric)

    with switch_evaluation_mode(trainer, metric_hook_list=[val_sche_metric_hook]):
        internlm_accelerator.empty_cache()
        trainer.eval()
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
                        with switch_evaluation_pipeline_scheduler(trainer=trainer):
                            # Compatible for non-moe
                            if hasattr(gpc.config.model, "num_experts"):
                                _, _, loss, moe_loss = trainer.execute_schedule(
                                    batch, forward_only=True, return_loss=True, return_output_label=False
                                )
                            else:
                                _, _, loss = trainer.execute_schedule(
                                    batch, forward_only=True, return_loss=True, return_output_label=False
                                )
                    else:
                        if hasattr(gpc.config.model, "num_experts"):
                            _, _, loss, moe_loss = trainer.execute_schedule(
                                batch, forward_only=True, return_loss=True, return_output_label=False
                            )
                        else:
                            _, _, loss = trainer.execute_schedule(
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

                if update_panel:
                    logger.info(
                        f"Validation on {val_name}: " + " ".join([f"{key}={value}" for key, value in infos.items()]),
                        extra={
                            "step": step_count,
                            "val_loss": val_loss,
                            "val_acc": val_res["acc"],
                            "val_perplexity": val_res["perplexity"],
                        },
                    )
                else:
                    logger.info(
                        f"Validation on {val_name}: " + " ".join([f"{key}={value}" for key, value in infos.items()])
                    )

        trainer.train()
        internlm_accelerator.empty_cache()
        dist.barrier()
