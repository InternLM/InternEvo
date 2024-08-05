from typing import Dict

import torch

from internlm.simulator.common import BW, CommOp
from internlm.simulator.predict_cost_model import SplineModel

cost_model = None
scale_ratio = [1.415134488, 1.208864145, 1.1, 1]


def coll_comm_lat(comm_op, size, n):
    if comm_op == CommOp.ALL2ALL:
        if n <= 8:
            return size * (n - 1) / n
        else:
            # intra_parts = 8
            one_part = size / n
            return 8 * one_part * (n - 8 / n)
    elif comm_op == CommOp.ALLREDUCE:
        return size * 2 * (n - 1) / n
    elif comm_op == CommOp.REDUCESCATTER:
        return size * (n - 1) / n
    elif comm_op == CommOp.ALLGATHER:
        return size * (n - 1) / n
    elif comm_op == CommOp.BROADCAST:
        return size * (n - 1) / n
    elif comm_op == CommOp.P2P:
        return size

    raise ValueError(f"unknown comm_op: {comm_op}")


def coll_bus_bw(comm_op, size):
    if comm_op == CommOp.ALL2ALL:
        return size
    elif comm_op == CommOp.ALLREDUCE:
        return size * 2
    elif comm_op == CommOp.REDUCESCATTER:
        return size
    elif comm_op == CommOp.ALLGATHER:
        return size
    elif comm_op == CommOp.BROADCAST:
        return size
    elif comm_op == CommOp.P2P:
        return size

    raise ValueError(f"unknown comm_op: {comm_op}")


# 需要判断是否打满带宽
def get_scale_ratio(scale):
    # 通信扩展惩罚系数
    if scale <= 16:
        return 1
    elif 16 < scale <= 32:
        return 1.1
    elif 32 < scale <= 64:
        return 1.2
    elif 64 < scale <= 256:
        return 1.3
    elif 256 < scale <= 512:
        return 1.4
    else:
        return 1.5


class SingleCommMetric:
    def __init__(self) -> None:
        self.dur = 0
        self.volume = 0
        self.count = 0
        self.bw = 0

    def add_new_comm(self, dur, volume, bw):
        self.dur += dur
        self.volume += volume
        self.bw += bw
        self.count += 1

    def __repr__(self) -> str:
        return f"dur: {self.dur}, volume: {self.volume}, avg bw: {self.bw/self.count:.3f} GB/s"

    def __str__(self) -> str:
        return self.__repr__()


class CommType:
    WP_PREFETCH_ALLGATHER = "wp_preftch_allgaher"
    WP_WDP = "wp_wdp"
    DP_ALLREDUCE = "dp_allreduce"
    MSP_REDUCE_SCATTER = "msp_reduce_scatter"
    MSP_ALLGATHER = "msp_allgahter"
    MTP_ALLREDUCE = "mtp_allreduce"

    SP_NORM_ALLREDUCE = "sp_norm_allreduce"


coom_type_list = [
    CommType.WP_PREFETCH_ALLGATHER,
    CommType.WP_WDP,
    CommType.DP_ALLREDUCE,
    CommType.MSP_ALLGATHER,
    CommType.MSP_REDUCE_SCATTER,
    CommType.MSP_ALLGATHER,
    CommType.MSP_ALLGATHER,
]


class WPCommCost:
    """
    WP的通信开销包括:
        1. pre-fetch allgahter
        2. wdp
    """

    def __init__(self) -> None:
        pass


class CommTracker:
    def __init__(self) -> None:
        self.next_comm_type = None
        self.next_parallel_mode = None

        self.comm_cost_dict: Dict[CommType, SingleCommMetric] = {}
        for comm_type in coom_type_list:
            self.comm_cost_dict[comm_type] = SingleCommMetric()

    def add_comm_meta(self, comm_type: CommType, parallel_mode, can_overlap):
        self.next_comm_type = comm_type
        self.next_parallel_mode = parallel_mode
        self.can_overlap = can_overlap

    def cal_comm_cost(self, comm_op, comm_volume=1, dtype=torch.bfloat16):
        """根据通信量获得近似的通信延迟,这个函数考虑了跨节点带宽content的情景
        所以为了正确计算延迟，传入的 comm_volume 必须是以单个rank视角下的通信量
        (即代码中实际传入的通信量)

        Args:
            comm_volume (int): 通信量, 单位B
            parallel_mode (ParallelMode): gpc并行模式
            comm_op (CommOp, optional): 通信算子

        Returns:
            int: 通信延迟,是乘以10**4后并取整后的数值
        """

        from internlm.core.context import ParallelMode
        from internlm.core.context import global_context as gpc

        comm_type = self.next_comm_type
        parallel_mode = self.next_parallel_mode

        if comm_type is None:
            return

        scale = gpc.get_world_size(parallel_mode)

        if parallel_mode == ParallelMode.PIPELINE:
            scale = 2

        if scale <= 1:
            return 0

        is_intra = gpc.check_pg_is_intra(parallel_mode)
        if not is_intra:
            num_partner = gpc.same_group_in_one_node(parallel_mode)
            assert num_partner <= 8, f"num_partner: {num_partner}"
            if parallel_mode == ParallelMode.WEIGHT:
                assert num_partner == 1
            if parallel_mode == ParallelMode.TENSOR:
                assert num_partner == 1
            comm_volume *= num_partner

        global cost_model
        try:
            if cost_model is None:
                cost_model = SplineModel()

            lat = cost_model.predict(comm_type, scale, comm_volume)
        except FileNotFoundError:
            # if comm_op == CommOp.P2P:
            bw = BW.A800_NVL if is_intra else (BW.IB / get_scale_ratio(scale))

            lat = coll_comm_lat(comm_op, comm_volume, scale) / bw  # 转换成ms小数点保留两位

        self.comm_cost_dict[comm_type].add_new_comm(lat, comm_volume, bw)


comm_tracker = CommTracker()


def get_gloabl_comm_tracker() -> CommTracker:
    return comm_tracker
