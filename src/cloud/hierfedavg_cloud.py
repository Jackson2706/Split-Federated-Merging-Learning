import copy
import time

import torch

from src.edge.hierfedavg_edge import HierFedAvgEdge
from src.utils.logger import Logger

from .base_cloud import BaseCloud


def split_into_clusters(lst, n):
    k, m = divmod(len(lst), n)
    return [
        lst[i*k + min(i, m):(i+1)*k + min(i+1, m)]
        for i in range(n)
    ]

class HierFedAvgCloud(BaseCloud):
    def __init__(self, global_model, clients, edges, config=None, test_dataset=None):
        super().__init__(global_model, clients, edges, config, test_dataset)
        self.logger = Logger(log_dir="logs/HierFedAvg", name="cloud")
        clusters = split_into_clusters(self.clients, self.edges)
        self.silos = [
            HierFedAvgEdge(edge_id=i, clients=silo, config=config)
            for i, silo in enumerate(clusters)
        ]

    def run(self, rounds, test_dataset=None, **kwargs):
        history = []
        self.first_distribute()
        for r in range(rounds):
            round_start = time.time()
            print(f"--- Round {r+1}/{rounds} ---")

            updates = []
            for silo in self.silos:
                silo_updates = []
                for client in silo.clients:
                    client.local_train(
                        epochs=self.config["training"]["local_epochs"],
                        lr=self.config["training"]["lr"]
                    )
                    silo_updates.append(client.send_update())
                silo.aggregate(silo_updates)
                updates.append(silo.send_update())
            new_weights = self.aggregate(updates)
            self.global_model.load_state_dict(new_weights)
            self.distribute()

            loss, acc = self.evaluate()

            # cloud-level logging
            self.logger.log(step=f"round_{r+1}", loss=loss, acc=acc, step_start=round_start)

            history.append(copy.deepcopy(new_weights))

        self.logger.log_final()
        return history

    def aggregate(self, updates):
        """FedAvg: simple average of client weights"""
        new_state = copy.deepcopy(updates[0])
        for key in new_state.keys():
            for i in range(1, len(updates)):
                new_state[key] += updates[i][key]
            new_state[key] = torch.div(new_state[key], len(updates))
        return new_state

    def first_distribute(self):
        """Send initial model to clients"""
        for client in self.clients:
            client.model = self.global_model

    def distribute(self):
        """Send global model to clients"""
        for client in self.clients:
            client.model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))