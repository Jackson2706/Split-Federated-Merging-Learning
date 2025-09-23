import time
from abc import ABC, abstractmethod

import torch

from src.utils.logger import Logger


class BaseEdge(ABC):
    def __init__(self, edge_id, clients, config):
        self.edge_id = edge_id
        self.clients = clients
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def aggregate(self, updates):
        pass
    
    @abstractmethod
    def send_update(self):
        """Send local model update"""
        pass

    @abstractmethod
    def send_smashed_data(self, batch):
        """Send smashed data (SplitNN style)"""
        pass

    def log_aggregation(self, round_id, loss=0.0, acc=0.0):
        step_start = time.time()
        self.logger.log(step=f"round_{round_id}", loss=loss, acc=acc, step_start=step_start)
