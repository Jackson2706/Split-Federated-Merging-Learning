# In hfedprotogencl_cloud.py

from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cloud.base_cloud import BaseCloud
from src.edge.hfedprotogencl_edge import HFedProtoGenCLEdge
from src.utils.logger import Logger
import time
import copy


def split_into_clusters(lst, n):
    k, m = divmod(len(lst), n)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    ]


class HFedProtoGenCLCloud(BaseCloud):
    def __init__(
        self,
        global_model,
        clients,
        edges=None,
        config=None,
        test_dataset=None,
        embed_dim=128,
        tau=0.1,
    ):
        super().__init__(global_model, clients, edges, config, test_dataset)
        self.logger = Logger(log_dir="logs/HFedProtoGenCL", name="cloud")
        self.embed_dim = embed_dim
        self.tau = tau
        self.generator = None
        clusters = split_into_clusters(self.clients, self.edges)
        self.silos = [
            HFedProtoGenCLEdge(edge_id=i, clients=silo, config=config)
            for i, silo in enumerate(clusters)
        ]
        self.new_prototype = None

    def run(self, rounds, test_dataset=None, **kwargs):
        history = []
        self.first_distribute()
        for r in range(rounds):
            round_start = time.time()
            print(f"--- Round {r+1}/{rounds} ---")

            updates = []
            for silo in self.silos:
                # The silo returns the aggregated prototypes from its clients
                silo_updates = []
                for client in silo.clients:
                    client.local_train(
                        epochs=self.config["training"]["local_epochs"],
                        lr=self.config["training"]["lr"],
                    )

                    silo_updates.append(client.send_update())
                silo.aggregate(silo_updates)
                for client in silo.clients:
                    client.align_loss_at_edge = silo.send_alignment_loss()
                updates.append(silo.send_update())
            self.new_prototype = self.aggregate(updates)
            self.distribute()

            max_acc = -1
            global_loss = -1
            for client in self.clients:
                self.global_model = client.model
                loss, acc = self.evaluate()
                if max_acc < acc:
                    max_acc = acc
                    global_loss = loss

            self.logger.log(
                step=f"round_{r+1}",
                loss=global_loss,
                acc=max_acc,
                step_start=round_start,
            )
            history.append(copy.deepcopy(self.global_model.state_dict()))

        self.logger.log_final()
        return history

    def first_distribute(self):
        for client in self.clients:
            client.model = self.global_model

    # In your HFedProtoGenCLCloud class

    def aggregate(self, updates):
        """
        Aggregates a list of dictionaries, where each dictionary contains
        prototype vectors for different classes. Instead of averaging, it
        selects the most central prototype (medoid) per class, i.e., the one
        closest to all others in cosine similarity space.

        Args:
            updates (list): A list of dictionaries, with each dictionary
                            containing prototype vectors from a silo.
                            e.g., [{'class1': tensor, 'class2': tensor}, ...]

        Returns:
            dict: A dictionary of representative (central/medoid) prototype
                vectors per class.
        """
        if not updates:
            return {}

        # Collect all prototypes for each class
        class_protos = {}
        for protos_dict in updates:
            for label, proto_vector in protos_dict.items():
                proto_vector = proto_vector.to(self.device)
                if label not in class_protos:
                    class_protos[label] = [proto_vector]
                else:
                    class_protos[label].append(proto_vector)

        # For each class, find the central prototype (medoid)
        central_protos = {}
        for label, protos in class_protos.items():
            # Stack all prototypes for this class into [N, D]
            protos_tensor = torch.stack(protos, dim=0)  # shape [N, D]

            # Normalize for cosine similarity
            protos_norm = F.normalize(protos_tensor, p=2, dim=1)

            # Compute cosine similarity matrix [N, N]
            sim_matrix = protos_norm @ protos_norm.T

            # Sum similarities per prototype
            sim_sums = sim_matrix.sum(dim=1)

            # Pick the prototype with maximum total similarity
            center_idx = torch.argmax(sim_sums)
            central_protos[label] = protos_tensor[center_idx]

        return central_protos

    def distribute(self):
        for client in self.clients:
            client.global_protos = self.new_prototype
