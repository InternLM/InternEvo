from abc import ABCMeta, abstractmethod

from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        raise NotImplementedError
