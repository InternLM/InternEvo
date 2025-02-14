from abc import ABCMeta, abstractmethod

from torch import nn

from internlm.model.model_ops.utils import load_src_states, merge_pp_src_states


class BaseTransformerModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for InternEvo transformer models.
    """

    @staticmethod
    @abstractmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        raise NotImplementedError

    @staticmethod
    def load_sharded_states(src):
        states = merge_pp_src_states(load_src_states(src))
        num_shards = len(states)
        return states, num_shards
