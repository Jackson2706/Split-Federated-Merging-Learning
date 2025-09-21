from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    @abstractmethod
    def step(self):
        """Perform one optimization step"""
        pass

    @abstractmethod
    def zero_grad(self):
        """Reset gradients"""
        pass