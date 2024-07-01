#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import os
import shutil
import socket
import traceback
from functools import wraps

from internlm.accelerator import get_accelerator
from internlm.core.context import global_context as gpc
from internlm.monitor.monitor import initialize_monitor_manager
from internlm.monitor.monitor import monitor_manager as mm

# global llm logger
logger = logging.getLogger(__file__)
internlm_accelerator = get_accelerator()

def internevo_monitor(feishu_alert=True, clean_run=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if feishu_alert:
                with initialize_monitor_manager(
                    job_name=gpc.config.JOB_NAME, alert_address=gpc.config.monitor.alert.feishu_alert_address
                ):
                    return execute_with_exception_handling(func, *args, **kwargs)
            else:
                return execute_with_exception_handling(func, *args, **kwargs)

        def execute_with_exception_handling(func, *args, **kwargs):
            if not clean_run:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                hostname = socket.gethostname()
                logger.error(
                    f"Raise exception from {hostname} with rank id: {gpc.get_global_rank()}\n{traceback.format_exc()}",
                )
            finally:
                devices_per_node = internlm_accelerator.device_count()
                local_rank = gpc.get_global_rank() % devices_per_node
                if gpc.config.data.use_shm and local_rank == 0:
                    if os.path.exists(gpc.config.data.shm_path):
                        shutil.rmtree(gpc.config.data.shm_path)

        return wrapper
    return decorator