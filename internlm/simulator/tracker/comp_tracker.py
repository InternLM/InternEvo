class CompTracker:
    def __init__(self) -> None:
        self.next_comm_type = None
        self.next_parallel_mode = None

    # def add_comm_meta(self, comm_type: CommType, parallel_mode, can_overlap):
    #     self.next_comm_type = comm_type
    #     self.next_parallel_mode = parallel_mode
    #     self.can_overlap = can_overlap

    # def cal_comm_cost(self, comm_op, comm_volume=1, dtype=torch.bfloat16):
    #     pass
