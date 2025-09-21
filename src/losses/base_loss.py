from abc import ABC, abstractmethod


class BaseLoss(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, outputs, targets):
        """Compute loss value"""
        pass