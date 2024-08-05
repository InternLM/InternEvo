import math
import os

import torch
import torch.distributed as dist
from torch.distributed import GroupMember


# TODO: 这里需要增加一个broadcast
class CommOp:
    ALL2ALL = "all2all"
    ALLREDUCE = "all_reduce"
    REDUCESCATTER = "reduce_scatter"
    ALLGATHER = "all_gather"
    LINEAR = "linear"
    BROADCAST = "broadcast"
    P2P = "p2p"
    FLASH_ATTN = "flash_attn"


class AlgoType:
    ISP = "isp"
    MSP = "msp"
    FSP = "fsp"
    MTP = "mtp"
    NONE = "none"


class BW:
    IB = 100 * 1024**3
    A800_NVL = 150 * 1024**3  # 满速是 200 GB/s
    A100_NVL = 250 * 1024**3  # 满速是 300 GB/s


BENCH_TYPE_LIST = [CommOp.ALL2ALL, CommOp.ALLREDUCE, CommOp.REDUCESCATTER, CommOp.ALLGATHER, CommOp.LINEAR]
# BENCH_TYPE_LIST = [CommOp.ALL2ALL, CommOp.ALLREDUCE, CommOp.REDUCESCATTER, CommOp.ALLGATHER, CommOp.LINEAR]

K = 1024

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

MS = 1000
US = 1000 * MS

_75GB = 75 * GB
_100GB = 100 * GB

GLOBAL_BYTE_SIZES_LIST = [512 * KB, 1 * MB, 4 * MB, 64 * MB, 128 * MB, 256 * MB, 512 * MB, 1 * GB, 2 * GB, 4 * GB]
# GLOBAL_BYTE_SIZES_LIST = [512 * KB, 1 * MB, 4 * MB] # , 64 * MB, 128 * MB, 256 * MB]
GLOBAL_ELEM_SIZES_LIST = [dsize // 2 for dsize in GLOBAL_BYTE_SIZES_LIST]
WORLD_SIZE_LIST = [2, 4, 8, 16, 32, 64, 128]
TP_SIZE_RANGE = [1] + list(range(2, 80 + 1, 2))

OUT_OF_MEM_LATENCY = 10**9


def cal_block_p_elem(h, multiple_of, mlp_ratio):
    norm1_p_elem = h
    norm2_p_elem = h
    MHA = h * 3 * h
    out_proj = h * h
    mlp_hidden_features = multiple_of * ((int(h * mlp_ratio) + multiple_of - 1) // multiple_of)
    mlp_p_elem = (h * mlp_hidden_features) * 3
    dropout1 = 0
    dropout2 = 0
    return norm1_p_elem + norm2_p_elem + MHA + out_proj + mlp_p_elem + dropout1 + dropout2


def cal_model_p_elem(h, l, vocab_size, multiple_of, mlp_ratio):
    embedding_p_elem = vocab_size * h
    block_p_elem = l * cal_block_p_elem(h, multiple_of, mlp_ratio)
    norm_p_elem = h
    head_p_elem = vocab_size * h
    return embedding_p_elem + block_p_elem + norm_p_elem + head_p_elem


def get_model_config(model_size):
    if model_size == 7:
        h = 4096
        a = 32
        l = 32
    elif model_size == 13:
        h = 5120
        a = 40
        l = 40
    elif model_size == 20:
        h = 5120
        a = 40
        l = 60
    elif model_size == 30:
        h = 6144
        a = 48
        l = 60
    elif model_size == 65:
        h = 8192
        a = 64
        l = 80
    elif model_size == 104:
        h = 10240
        a = 80
        l = 82
    else:
        raise ValueError(f"unsupport modesize: {model_size}")

    vocab_size = 103168
    mlp_ratio = 8 / 3
    multiple_of = 256

    model_p_elem = cal_model_p_elem(h=h, l=l, vocab_size=vocab_size, multiple_of=multiple_of, mlp_ratio=mlp_ratio)

    return h, a, l, mlp_ratio, multiple_of, model_p_elem


def pretty_print_size(x):
    if x < KB:
        return f"{x} B"
    elif x >= KB and x < MB:
        return f"{x/KB:.3f} KB"
    elif x >= MB and x < GB:
        return f"{x/MB:.3f} MB"
    else:
        return f"{x/GB:.3f} GB"


def pretty_print_latency(x):
    if x >= 1:
        return f"{x:.3f} s"
    elif x >= 1 / MS and x < 1:
        return f"{x*MS:.3f} ms"
    else:
        return f"{x*US:.3f} us"


def get_local_rank():
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"]) % 8
    else:
        return 0


def get_world_size():
    if "SLURM_NPROCS" in os.environ:
        return int(os.environ["SLURM_NPROCS"])
    else:
        return 1


def sync_all():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def get_bw(comm_op, size, duration, args):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all":
        tput = size / duration
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather" or comm_op == "reduce_scatter":
        size *= n
        tput = size / duration
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = size * 2 / duration
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt" or comm_op == "broadcast":
        tput = size / duration
        busbw = tput
    else:
        print("wrong comm_op specified")
        exit(0)

    if args.bw_unit == "Gbps":
        tput *= 8
        busbw *= 8

    return tput, busbw


sub_process_groups = {}
TORCH_DISTRIBUTED_DEFAULT_PORT = 12349


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def init_torch_distributed(backend):
    global dist

    # discover rank/size info from env
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(TORCH_DISTRIBUTED_DEFAULT_PORT)
    if "MASTER_ADDR" not in os.environ:
        import subprocess

        result = subprocess.check_output('scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1', shell=True)
        master_addr = result.decode("utf8").strip()
        if master_addr == "":
            master_addr = "127.0.0.1"
        os.environ["MASTER_ADDR"] = master_addr
    local_rank = env2int(
        ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"]
    )
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(local_rank)
    rank = env2int(["RANK", "MPI_RANKID", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID"])
    if "RANK" not in os.environ:
        os.environ["RANK"] = str(rank)
    world_size = env2int(["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NPROCS"])
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(world_size)

    torch.distributed.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


def build_process_gourp(max_world_size):
    global sub_process_groups
    if max_world_size > 1:
        init_torch_distributed("nccl")
        sub_process_groups[str(dist.get_world_size())] = GroupMember.WORLD

        if dist.is_initialized():
            world_size = dist.get_world_size()
            node_nums = world_size // 8
            base_num = [2, 4, 6] + [8 * i for i in range(1, node_nums)]

            for gpu_nums in base_num:
                ranks = [j for j in range(gpu_nums)]
                print(ranks, flush=True)
                sub_process_groups[f"{gpu_nums}"] = dist.new_group(ranks)
                # dist.get_process_group_ranks()


def get_global_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0
