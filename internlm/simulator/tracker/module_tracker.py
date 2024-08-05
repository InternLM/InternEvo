# from typing import Self


# from internlm.simulator.tracker.global_var import get_pre_module_tracker
from typing import TypeVar

_ModuleTracker = TypeVar("_ModuleTracker", bound="ModuleTracker")

pre_module_tracker = None

now_comm_tracker = None
now_comp_tracker = None
now_mem_tracker = None


def set_now_comm_tracker(tracker):
    global now_comm_tracker
    now_comm_tracker = tracker


def set_now_comp_tracker(tracker):
    global now_comp_tracker
    now_comp_tracker = tracker


def set_now_mem_tracker(tracker):
    global now_mem_tracker
    now_mem_tracker = tracker


def get_now_comm_tracker():
    return now_comm_tracker


def get_now_comp_tracker():
    return now_comp_tracker


def get_pre_module_tracker() -> _ModuleTracker:
    return pre_module_tracker


class ModuleTracker:
    def __init__(self, name: str) -> None:

        from internlm.simulator.tracker.comm_tracker import CommTracker
        from internlm.simulator.tracker.comp_tracker import CompTracker
        from internlm.simulator.tracker.mem_tracker import TensorTracker

        self.name = name
        self.father_module = get_pre_module_tracker()
        if self.father_module is not None:
            self.father_module.register_submodule_tracker(self)

        self.comm_tracker = CommTracker()
        self.comp_tracker = CompTracker()
        self.mem_tracker = TensorTracker()
        self.sub_tracker = []

    def fwd_pre_hook(self, module, args, kwargs):
        set_now_comm_tracker(self.comm_tracker)
        set_now_comp_tracker(self.comp_tracker)
        set_now_mem_tracker(self.mem_tracker)
        print(f"[DEBUG]: call {self.name} fwd_pre_hook !", flush=True)

    def bwd_pre_hook(self, module, grad_input, grad_output):
        set_now_comm_tracker(self.comm_tracker)
        set_now_comp_tracker(self.comp_tracker)
        set_now_mem_tracker(self.mem_tracker)
        print(f"[DEBUG]: call {self.name} bwd_pre_hook !", flush=True)

    def register_submodule_tracker(self, module_tracker: _ModuleTracker):
        self.sub_tracker.append(module_tracker)
