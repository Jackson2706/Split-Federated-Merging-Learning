from src.edge.base_edge import BaseEdge
import torch
import copy
from src.utils.logger import Logger
class HierFedAvgEdge(BaseEdge):
    def __init__(self, edge_id, clients, config):
        super().__init__(edge_id, clients, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger(log_dir=f"logs/HierFedAvg", name=f"client_{edge_id}")

    def aggregate(self, updates):
        """HierFedAvg: simple average of client weights"""
        new_state = copy.deepcopy(updates[0])
        for key in new_state.keys():
            for i in range(1, len(updates)):
                new_state[key] += updates[i][key]
            new_state[key] = torch.div(new_state[key], len(updates))
        return new_state
    
    def send_smashed_data(self):
        pass
    def send_update(self):
        """Send local weights to cloud"""
        return copy.deepcopy(self.model.state_dict())