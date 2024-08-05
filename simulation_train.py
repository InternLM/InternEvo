#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import logging
import os
import socket
import time

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import internlm
from internlm.core.context import Config, ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context.random import reset_seed
from internlm.core.trainer import TrainState
from internlm.initialize.launch import launch
from internlm.model.losses import FlashGPTLMLoss
from internlm.simulator.common import AlgoType, CommOp
from internlm.simulator.tracker.comm_tracker import get_gloabl_comm_tracker
from internlm.simulator.tracker.mem_tracker import get_global_allocator

# from internlm.simulator.elements.tensor import FakeTensor
from internlm.simulator.utils import PPIter, SPIter, get_bsz_approximate, get_bsz_strict
from internlm.train import (
    get_scheduler_hooks,
    initialize_llm_profile,
    initialize_model,
    initialize_optimizer,
    initialize_parallel_communicator,
)
from internlm.utils.common import (
    enable_pytorch_expandable_segments,
    launch_time,
    parse_args,
)

# global llm logger
logger = logging.getLogger(__file__)


gloab_allocator = get_global_allocator()
global_comm_tracker = get_gloabl_comm_tracker()
from internlm.initialize.launch import args_sanity_check


class WaitHandler:
    def wait(self):
        return


def dummy_broadcast(tensor, src, group=None, async_op=False):
    global_comm_tracker.cal_comm_cost(
        comm_op=CommOp.BROADCAST, comm_volume=tensor.numel() * tensor.element_size(), dtype=tensor.dtype
    )
    if async_op is True:
        return WaitHandler()


def dummy_allreduce(tensor, op, group=None, async_op=False):
    global_comm_tracker.cal_comm_cost(
        comm_op=CommOp.ALLREDUCE, comm_volume=tensor.numel() * tensor.element_size(), dtype=tensor.dtype
    )
    if async_op is True:
        return WaitHandler()


def dummy_allgahter(tensor_list, tensor, group=None, async_op=False):
    if async_op is True:
        return WaitHandler()


def dummy_reduce_scatter(output, input_list, op, group=None, async_op=False):
    if async_op is True:
        return WaitHandler()


def dummy_reduce_scatter(output, input_list, op, group=None, async_op=False):
    if async_op is True:
        return WaitHandler()


def dummy_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
    if async_op is True:
        return WaitHandler()


def dummy_batch_isend_irecv(p2p_op_list):
    return [WaitHandler() for _ in range(len(p2p_op_list))]


def dummy_barrier(group=None, async_op=False, device_ids=None):
    if async_op is True:
        return WaitHandler()


old_bcast = dist.broadcast
old_all_reduce = dist.all_reduce
old_all_gahter = dist.all_gather
old_reduce_scatter = dist.reduce_scatter
old_all_to_all = dist.all_to_all
old_batch_isend_irecv = dist.batch_isend_irecv
old_barrier = dist.barrier

dist.broadcast = dummy_broadcast
dist.all_reduce = dummy_allreduce
dist.all_gather = dummy_allgahter
dist.reduce_scatter = dummy_reduce_scatter
dist.all_to_all = dummy_all_to_all
dist.batch_isend_irecv = dummy_batch_isend_irecv
dist.barrier = dummy_barrier


def main(args):
    very_begining_time = time.time()
    enable_pytorch_expandable_segments()

    # init setting
    skip_batches = gpc.config.data.skip_batches
    total_steps = gpc.config.data.total_steps
    valid_every = gpc.config.data.valid_every
    label_smoothing = gpc.config.loss.label_smoothing

    # initialize model
    model = initialize_model()
    model = model.to("cuda")
    # print(model)
    # for prefix, module in model.named_modules():
    #     print(f"prefix: {prefix}, module: {module}", flush=True)
    # for prefix, param in model.named_parameters():
    #     print(f"prefix: {prefix}, param: {param}", flush=True)

    # initialize isp communicator
    isp_communicator = initialize_parallel_communicator(model)

    # initialize loss function
    criterion = FlashGPTLMLoss(parallel_output=gpc.config.model.parallel_output, label_smoothing=label_smoothing)

    # initialize the train and validation data loader
    # initialize and resume train state
    train_state = TrainState(gpc.config, None)

    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(model, isp_communicator)

    # initialize trainer
    trainer, train_dl, _, _ = internlm.initialize_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=None,
        lr_scheduler=lr_scheduler,
        beta2_scheduler=beta2_scheduler,
        scheduler_hooks=get_scheduler_hooks(None, optimizer, isp_communicator),
    )

    trainer.train()
    total_steps = 10

    S = gpc.config.data["seq_len"]
    micro_num = gpc.config.data["micro_num"]
    micro_bsz = gpc.config.data["micro_bsz"]

    batch = [
        {
            "input_ids": torch.tensor(micro_num * [list(range(micro_bsz * S))], dtype=torch.int64),
            "cu_seqlens": torch.tensor(micro_num * [[0, S]], dtype=torch.int64),
            "indexes": torch.tensor(micro_num * [list(range(S))], dtype=torch.int64),
            # 'type_ids': torch.tensor(micro_num* [list(range(S))], dtype=torch.int32 ),
        },
        torch.tensor(micro_num * [list(range(micro_bsz * S))], dtype=torch.int64),
    ]
    print(batch)
    with initialize_llm_profile(profiling=True, start_time=launch_time()) as prof:
        for batch_count in range(train_state.batch_count, total_steps):
            s = time.time()
            # record the consumed samples in training
            train_state.batch_count = batch_count
            train_state.num_consumed_samples_in_epoch += len(batch[1])

            # zero the grads of parameters
            trainer.zero_grad()

            if hasattr(gpc.config.model, "num_experts"):
                trainer.execute_schedule(
                    batch,
                    forward_only=False,
                    return_loss=True,
                    return_output_label=False,
                )
            else:
                trainer.execute_schedule(
                    batch,
                    forward_only=False,
                    return_loss=True,
                    return_output_label=False,
                )

            if isp_communicator and isp_communicator.enable_memory_pool:
                isp_communicator.memory_pool.reset_lazy_pools()

            trainer_result = trainer.step()
            print(f"ont step use time: {time.time() -s :.3f} s", flush=True)
            prof.step()
            import pdb

            pdb.set_trace()


def run_loop(
    global_bsz,
    world_size,
    args,
    use_fixed_micro_bsz=False,
    use_strict_bsz=True,
    global_bsz_max=1,
    global_bsz_min=1,
    debug=True,
):
    gpc.load_config(config=Config.from_file(args.config))
    gpc.set_fake_mode(True)

    L = gpc.config.model["num_layers"]
    KV_H = gpc.config.model["num_kv_attention_heads"]
    S = gpc.config.data["seq_len"]
    H = gpc.config.model["hidden_size"]
    MICRO_BSZ = gpc.config.data["micro_bsz"]
    MICRO_NUM = gpc.config.data["micro_num"]

    pp_search_range = PPIter(world_size, L)
    sp_search_range = SPIter(world_size, KV_H)
    wp_search_ranges = SPIter(world_size, world_size)
    # zp_search_ranges_max = SPIter(world_size, world_size)
    solutions_list = []
    algo_list = [AlgoType.ISP, AlgoType.MSP, AlgoType.FSP]

    for pp_i, pp in enumerate(pp_search_range):
        for sp_i, sp in enumerate(sp_search_range):
            if not use_fixed_micro_bsz:
                if use_strict_bsz:
                    bs_bns = get_bsz_strict(global_bsz, world_size, pp, sp, S)
                else:
                    bs_bns = get_bsz_approximate(global_bsz_max, global_bsz_min, world_size, pp, sp, S)
                if bs_bns is None or len(bs_bns) == 0:
                    if debug:
                        print(
                            f"NO solu: pp:{pp} , sp:{sp} can't find micro_bsz/micro_num for"
                            f"world_size:{world_size}, seq_len:{S}, global bsz range: [{global_bsz_min}-{global_bsz_max}]!",
                            flush=True,
                        )
                    continue
            else:
                bs_bns = [(MICRO_BSZ, MICRO_NUM)]

            for micro_bsz, micro_num in bs_bns:
                for algo_type in algo_list:
                    for activation_ckpt in [0, 1]:
                        for wp_i, wp in enumerate(wp_search_ranges):
                            if algo_type in [AlgoType.MSP, AlgoType.FSP]:
                                if wp > 1:
                                    if debug:
                                        print("NO solu: msp, fsp not support wp>1 !", flush=True)
                                    continue  # msp, fsp禁掉fsdp，我们目前还不支持
                                # zp的搜索空间是被wp限制的，同时他不是按照8的倍数变化的，是,1,2,3, ...这样递增的
                                zp_search_range = world_size // pp // sp // wp  # 这里的sp对于msp和fsp来说是tp
                            else:
                                zp_search_range = world_size // pp // wp  # internlm实现的zp和deepspeed不一样，zp是在切wp的基础上再切的

                            try:
                                assert H % sp == 0, f"embed_dim:{H} must be divisible by sp: {sp}"
                                assert KV_H % sp == 0, f"num_heads: {KV_H} must be divisible by sp: {sp}"
                                assert KV_H >= sp, f"num_heads: {KV_H} must bigger then sp: {sp}"
                            except AssertionError as e:
                                if debug:
                                    print(f"NO solu: head assert {e}", flush=True)
                                continue

                            for zp_i, zp in enumerate(range(1, zp_search_range + 1)):
                                # set config
                                print(
                                    f"activation_ckpt: {activation_ckpt}, micro_num: {micro_num}, micro_bsz: {micro_bsz}, pp: {pp}, wp: {wp}, zp: {zp}, sp: {sp}, {str(algo_type)}",
                                    flush=True,
                                )
                                gpc.config.model["checkpoint"] = activation_ckpt
                                gpc.config.parallel["zero1"]["size"] = zp
                                gpc.config.parallel["tensor"]["size"] = sp
                                gpc.config.parallel["tensor"]["mode"] = str(algo_type)
                                gpc.config.parallel["pipeline"]["size"] = pp
                                gpc.config.parallel["weight"]["size"] = wp

                                gpc.config.data["micro_num"] = micro_num
                                gpc.config.data["micro_bsz"] = micro_bsz

                                gpc.destroy()
                                reset_seed()

                                launch(
                                    config=gpc.config,
                                    local_rank=0,
                                    rank=0,
                                    world_size=world_size,
                                    host="127.0.0.1",
                                    port=12345,
                                    backend="nccl",
                                    seed=0,
                                    fake_mode=fake_mode,
                                )
                                args_sanity_check()
                                assert hasattr(gpc, "config") and gpc.config is not None

                                with FakeTensorMode():
                                    main(args)


if __name__ == "__main__":
    args = parse_args()
    hostname = socket.gethostname()
    world_size = args.world_size

    fake_mode = "fake_mode" in os.environ

    # initialize distributed environment
    print(f"fake_mode: {fake_mode}", flush=True)

    gloab_allocator.init_capcity = 80 * 1024**3
    gloab_allocator.capcity = 80 * 1024**3

    run_loop(global_bsz=4096 * 1024, world_size=world_size, args=args)
