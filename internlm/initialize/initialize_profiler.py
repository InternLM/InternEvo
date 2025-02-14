import torch

from internlm.accelerator import AcceleratorType
from internlm.accelerator.abstract_accelerator import get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.common import DummyProfile
from internlm.utils.logger import get_logger

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()

try:
    import torch_npu
except (ModuleNotFoundError, ImportError):
    pass


def initialize_llm_profile(profiling: bool = False, start_time: str = None):
    """Initialize and return the profiler context manager instance."""

    if profiling and gpc.get_local_rank(ParallelMode.DATA) == 0 and gpc.get_local_rank(ParallelMode.TENSOR) == 0:
        schedule_config = {"wait": 1, "warmup": 1, "active": 1, "repeat": 1, "skip_first": 3}
        trace_path = (
            f"RUN/{gpc.config.JOB_NAME}/{start_time}/traces/rank{gpc.get_global_rank()}_"
            f"dp{gpc.get_local_rank(ParallelMode.DATA)}_"
            f"wp{gpc.get_local_rank(ParallelMode.WEIGHT)}_"
            f"tp{gpc.get_local_rank(ParallelMode.TENSOR)}"
        )
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            llm_profile = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(**schedule_config),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(trace_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
            logger.info(f"Do profiling for NPU on rank {gpc.get_global_rank()}!")
        else:
            llm_profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(**schedule_config),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_path),
                with_stack=True,
                with_modules=True,
                profile_memory=True,
            )
            logger.info(f"Do profiling for GPU on rank {gpc.get_global_rank()}!")
    else:
        llm_profile = DummyProfile()

    return llm_profile
