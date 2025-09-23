import copy

import torch

from src.edge.base_edge import BaseEdge
from src.utils.logger import Logger


class HierFedAvgEdge(BaseEdge):
    def __init__(self, edge_id, clients, config):
        super().__init__(edge_id, clients, config)
        self.logger = Logger(log_dir=f"logs/HierFedAvg", name=f"edge_{edge_id}")
        self.weight = None

    def aggregate(self, updates):
        """HierFedAvg: simple average of client weights"""
        new_state = copy.deepcopy(updates[0])
        for key in new_state.keys():
            for i in range(1, len(updates)):
                new_state[key] += updates[i][key]
            new_state[key] = torch.div(new_state[key], len(updates))
        self.weight = new_state
    
    def send_smashed_data(self):
        pass
    def send_update(self):
        """Send local weights to cloud"""
        return copy.deepcopy(self.weight)