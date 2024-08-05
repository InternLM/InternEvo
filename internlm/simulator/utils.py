import math

from internlm.simulator.common import GB, AlgoType, cal_block_p_elem, get_model_config


class LinsSolutionNoZ3:
    def __init__(
        self,
        pp,
        sp,
        wp,
        zp,
        seq_len,
        micro_bsz,
        micro_num,
        algo_type,
        pp_comm_cost,
        activation,
        zp_comm_cost,
        wp_comm_cost,
        sp_comm_cost,
        os_mm_cost,
        p_g_mm_cost,
        fwd_bwd_cost,
        mem_cost,
        comp_wp,
        comp_attn,
        world_size,
        activation_ckpt,
        tgs,
        mem_pool_mm,
        norm_activation,
        head_input_activation,
        head_output_activation,
        block_output_activation,
        wdp_comm_cost,
        all_fwd_bwd_cost,
        g_bsz,
        pp_p2p_buffer,
        rotary_emb_sincos_cache_mm,
        modelsize,
        backward_mem_peak,
        blocks_activation,
        overlap_latency,
        total_latency,
    ):
        self.pp = pp
        self.sp = sp
        self.seq_len = seq_len
        self.micro_bsz = micro_bsz
        self.micro_num = micro_num
        self.algo_type = algo_type
        self.pp_comm_cost = pp_comm_cost
        self.activation = activation
        self.activation_ckpt = activation_ckpt

        self.wp_size = wp
        self.zp_size = zp
        self.zp_comm_cost = zp_comm_cost
        self.wp_comm_cost = wp_comm_cost
        self.os_mm_cost = os_mm_cost
        self.p_g_mm_cost = p_g_mm_cost
        self.sp_comm_cost = sp_comm_cost
        self.total_mm_cost = mem_cost
        self.fwd_bwd_cost = fwd_bwd_cost
        self.comp_wp = comp_wp
        self.comp_attn = comp_attn
        self.world_size = world_size
        self.tgs = tgs

        self.mem_pool_mm = mem_pool_mm
        self.norm_activation = norm_activation
        self.head_input_activation = head_input_activation
        self.head_output_activation = head_output_activation
        self.block_output_activation = block_output_activation

        self.wdp_comm_cost = wdp_comm_cost
        self.all_fwd_bwd_cost = all_fwd_bwd_cost
        self.g_bsz = g_bsz
        self.pp_p2p_buffer = pp_p2p_buffer
        self.rotary_emb_sincos_cache_mm = rotary_emb_sincos_cache_mm
        self.modelsize = modelsize
        self.backward_mem_peak = backward_mem_peak
        self.blocks_activation = blocks_activation
        self.overlap_latency = overlap_latency
        self.total_latency = total_latency

    def __str__(self):
        return self.__repr__()

    # Begin: world_size: 128, pp:1, sp:16, micro_bsz:1, micro_num:2, algo_type:isp, wp:16, zp:4 ckpt:1
    def __repr__(self):
        return (
            f" world_size: {self.world_size}"
            f" tgs: {self.tgs}, total_latency:{self.total_latency*10**3:.3f} ms"
            f" global bsz: {self.g_bsz} \n"
            f" activation ckpt: {self.activation_ckpt}"
            f" seq_len: {self.seq_len}"
            f" micro_bsz: {self.micro_bsz}"
            f" micro_num: {self.micro_num}, \n"
            f" modelsize: {self.modelsize}, algo_type: {self.algo_type}, pp_size: {self.pp}, sp_size: {self.sp}, wp_size: {self.wp_size}, zp_size: {self.zp_size}, \n"
            f" one micro step fwd_bwd_cost: {self.fwd_bwd_cost*10**3:.2f} ms, all_fwd_bwd_cost: {self.all_fwd_bwd_cost*10**3:.2f} ms, overlap_latency: {self.overlap_latency*10**3:.2f} ms\n"
            f" COMP: comp_wp: {self.comp_wp*10**3:.2f} ms, comp_attn: {self.comp_attn*10**3:.2f} ms, \n"
            f" COMM: pp_comm_cost: {self.pp_comm_cost*10**3:.2f} ms, zp_comm_cost: {self.zp_comm_cost*10**3:.2f} ms, one layer wp_comm_cost: {self.wp_comm_cost*10**3:.2f} ms, one layer sp_comm_cost: {self.sp_comm_cost*10**3:.2f} ms, wdp_comm_cost: {self.wdp_comm_cost*10**3:.2f} ms \n"
            f" total mem_cost: {self.total_mm_cost /GB:.2f} GB \n"
            f" Not evictable MEM: os_mm_cost: {self.os_mm_cost/GB:.2f} GB, p_g_mm_cost: {self.p_g_mm_cost/GB:.2f} GB, isp_mem_pool: {self.mem_pool_mm/GB:.2f} GB, sincos_cache_mm: {self.rotary_emb_sincos_cache_mm/GB:.2f} GB,pp_p2p_buffer: {self.pp_p2p_buffer/GB:.2f} GB\n"
            f" Activation MEM: total activation: {self.activation/GB:.2f} GB, blocks_activation: {self.blocks_activation/GB:.2f} GB, norm_activation: {self.norm_activation/GB:.2f} GB,backward_mem_peak: {self.backward_mem_peak/GB:.2f} GB \n"
            f" head_input_activation: {self.head_input_activation/GB:.2f} GB, head_output_activation: {self.head_output_activation/GB:.2f} GB, block_output_activation(enable ckpt): {self.block_output_activation/GB:.2f} GB \n"
        )


class SPIter:
    def __init__(self, gpu_nums, head_nums):
        assert head_nums % 2 == 0
        stop = min(gpu_nums, head_nums)
        if gpu_nums <= 8:
            self.num_list = [1] + list(range(2, stop + 1, 2))
        else:
            self.num_list = [1] + list(range(2, 8, 2)) + list(range(8, stop + 1, 8))

    def __iter__(self):
        return iter(self.num_list)

    def __len__(self):
        return len(self.num_list)


class PPIter:
    def __init__(self, gpu_nums, layer_nums):
        # assert layer_nums % 2 == 0
        stop = int(math.log2(min(gpu_nums, layer_nums)))
        self.num_list = [2**i for i in range(stop + 1)]

    def __iter__(self):
        return iter(self.num_list)

    def __len__(self):
        return len(self.num_list)


def get_bsz_strict(global_bsz: int, world_size: int, pp_size: int, sp_size: int, seq_len: int):
    """
    严格的按照 global_bsz 限制返回满足要求的 micro_bsz 和 micro_num
    Args:
        pp_size (int)
        sp_size (int)
        seq_len (int)

    Returns:
        List[(int, int)]: micro_bsz, micro_num
    """
    if pp_size * sp_size > world_size:
        return None

    dp_world_size = world_size // pp_size // sp_size
    if world_size % pp_size != 0 or world_size % sp_size != 0 or world_size % (pp_size * sp_size) != 0:
        return None

    if global_bsz % dp_world_size != 0:
        return None
    if global_bsz % seq_len != 0:
        return None
    if global_bsz % (dp_world_size * seq_len) != 0:
        return None

    bsz = global_bsz // dp_world_size // seq_len

    micro_bsz_num = []
    for micro_bsz in range(1, bsz + 1):
        if bsz % micro_bsz == 0:
            micro_num = bsz // micro_bsz
            if micro_num >= pp_size:  # 我们暂时不考虑 micro_num < pp_size 的情况
                micro_bsz_num.append((micro_bsz, micro_num))
    return micro_bsz_num


def get_bsz_approximate(
    global_bsz_max: int, global_bsz_min: int, world_size: int, pp_size: int, sp_size: int, seq_len: int
):
    """
    允许global bsz在 min_bsz 和 max_bsz 之间松弛
    Args:
        pp_size (int)
        sp_size (int)
        seq_len (int)

    Returns:
        List[(int, int)]: micro_bsz, micro_num
    """
    if pp_size * sp_size > world_size:
        return None

    dp_world_size = world_size // pp_size // sp_size
    if world_size % pp_size != 0 or world_size % sp_size != 0 or world_size % (pp_size * sp_size) != 0:
        return None

    bsz_max = global_bsz_max // dp_world_size // seq_len
    bsz_min = global_bsz_min // dp_world_size // seq_len

    micro_bsz_num = []
    for micro_bsz in range(1, int(bsz_max**0.5) + 1):
        for micro_num in range(1, int(bsz_max**0.5) + 1):
            if micro_bsz * micro_num >= bsz_min:
                if micro_num >= pp_size:  # 我们暂时不考虑 micro_num < pp_size 的情况
                    assert micro_bsz * micro_num >= bsz_min and micro_bsz * micro_num <= bsz_max
                    micro_bsz_num.append((micro_bsz, micro_num))
    return micro_bsz_num
