import copy
import logging
import os
import time
from collections import deque
import sys
import numpy as np
import psutil
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from utils import *

class HierarchicalFL:
    def __init__(
        self,
        args,
        client_model,
        client_weights,
        edge_model,
        edge_weights,
        cloud_model,
        cloud_weight,
        test_dataset=None,
    ):
        self.args = args
        self.client_model = client_model
        self.edge_model = edge_model
        self.cloud_model = cloud_model

        self.client_weight = client_weights
        self.edge_weight = edge_weights
        self.cloud_weight = cloud_weight

        self.structure, self.connectivity = self._build_hierarchy()
        self.total_layers = len(args["mid_server"]) + 1
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.comm_tracker = {
            "client_upload_smashed_MB": 0.0,
            "edge_upload_smashed_MB": 0.0,
            "client_model_upload_MB": 0.0,
            "client_model_download_MB": 0.0,
            "edge_model_upload_MB": 0.0,
            "edge_model_download_MB": 0.0,
            "cloud_download_grad_MB": 0.0,
            "edge_download_grad_MB": 0.0,
        }
        self.client_cache = deque(maxlen=20)

    def _build_hierarchy(self):
        structure = {}
        connectivity = {}

        def build_layer(layer_idx):
            layer_dict = {}
            conn_dict = {}

            if layer_idx == -1:
                num_clients = self.args["num_users"]
                num_edges = self.args["mid_server"][0]
                clients_per_edge = num_clients // num_edges
                all_clients = list(range(num_clients))

                for edge_id in range(num_edges):
                    assigned = (
                        all_clients
                        if edge_id == num_edges - 1
                        else list(
                            np.random.choice(
                                all_clients, clients_per_edge, replace=False
                            )
                        )
                    )
                    for cid in assigned:
                        layer_dict[cid] = copy.deepcopy(self.client_model)
                        conn_dict[cid] = edge_id
                    all_clients = list(set(all_clients) - set(assigned))

            elif layer_idx == len(self.args["mid_server"]):
                layer_dict[0] = copy.deepcopy(self.cloud_model)
                for mid_id in range(self.args["mid_server"][-1]):
                    conn_dict[mid_id] = 0

            else:
                num_servers = self.args["mid_server"][layer_idx]
                prev_layer_count = (
                    self.args["num_users"]
                    if layer_idx == 0
                    else self.args["mid_server"][layer_idx - 1]
                )
                servers_per_layer = prev_layer_count // num_servers
                all_prev = list(range(prev_layer_count))

                for sid in range(num_servers):
                    assigned = (
                        all_prev
                        if sid == num_servers - 1
                        else list(
                            np.random.choice(
                                all_prev, servers_per_layer, replace=False
                            )
                        )
                    )
                    for nid in assigned:
                        conn_dict[nid] = sid
                    all_prev = list(set(all_prev) - set(assigned))
                    layer_dict[sid] = copy.deepcopy(self.edge_model)

            structure[layer_idx] = layer_dict
            if layer_idx > -1:
                connectivity[layer_idx - 1] = conn_dict

            if layer_idx < len(self.args["mid_server"]):
                build_layer(layer_idx + 1)

        build_layer(-1)
        return structure, connectivity

    def print_structure(self):
        for layer_idx in sorted(self.structure.keys()):
            layer_nodes = self.structure[layer_idx]
            layer_type = (
                "Client Layer"
                if layer_idx == -1
                else (
                    "Cloud Layer"
                    if layer_idx == len(self.args["mid_server"])
                    else f"Edge Layer {layer_idx}"
                )
            )
            logging.info(f"\n=== {layer_type} (Layer {layer_idx}) ===")
            for node_id, model in layer_nodes.items():
                model_type = (
                    "Client Model"
                    if model.__class__ == self.client_model.__class__
                    else (
                        "Edge Model"
                        if model.__class__ == self.edge_model.__class__
                        else "Cloud Model"
                    )
                )
                logging.info(f"  Node ID {node_id}: {model_type}")

    def get_model_size(self, state_dict):
        return (
            sum(param.numel() for param in state_dict.values()) * 4 / 1e6
        )  # MB

    def average_state_dicts(self, state_dicts):
        avg_dict = {}
        # Pick the device of the first state_dict (assuming they're all supposed to be the same)
        device = next(iter(state_dicts[0].values())).device

        for key in state_dicts[0].keys():
            tensors = [d[key].to(device) for d in state_dicts]
            avg_dict[key] = sum(tensors) / len(tensors)
        return avg_dict

    def edge_server_aggregation(self):
        pass

    def cloud_aggregation(self):
        pass

    def print_comm_report(self):
        print("\n=== Communication Report ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

    def initialize_optimizers(self):
        self.optimizers = {}
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {
                nid: torch.optim.Adam(
                    model.parameters(),
                    lr=self.args["lr"],
                    weight_decay=self.args["weight_decay"],
                )
                for nid, model in nodes.items()
            }

    def compute_prototype(self, feature_dicts):
        """
           Compute prototypes (mean feature map) for each key.

           Args:
               feature_dicts (dict[int, list[torch.Tensor]]):
               Dictionary mapping key -> list of feature tensors.
           Returns:
               dict[int, torch.Tensor]: Dictionary mapping key -> prototype tensor.
           """
        prototypes = {}
        for key, feature_list in feature_dicts.items():
            if not feature_list:  # skip empty lists
                continue
            # Stack and average along batch dimension
            stacked = torch.stack(feature_list).cpu()
            proto = stacked.mean(dim=0)
            prototypes[key] = proto
        return prototypes
    def train_end_to_end(
        self,
        train_dataset,
        test_dataset,
        user_groups,
        config,
        epochs,
    ):
        self.initialize_optimizers()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        criterion = nn.CrossEntropyLoss().to(device)
        evalcriterion = nn.NLLLoss().to(device)
        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]
        train_f1, train_loss = [], []
        client_cpu_list = []
        client_ram_list = []
        client_gpu_ram_list = []

        for epoch in tqdm(range(1, epochs + 1)):
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)

            start_time = time.time()
            client_cpu_usages = []
            client_ram_usages = []
            client_gpu_ram_usages = []
            client_outputs = {}


            '''
                Client-side process.
            '''
            # inside your training loop
            for cid in idxs_users:
                # Track client IDs for caching
                self.client_cache.append(cid)

                # Measure CPU + RAM before training client
                cpu_before = psutil.cpu_percent(interval=None)
                mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)  # MB

                # Reset GPU memory tracker
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

                # Prepare local dataset
                if self.args["iid"]:
                    local_data = DatasetSplit(train_dataset, user_groups[cid])
                else:
                    local_data = user_groups[cid]

                loader = DataLoader(local_data, batch_size=local_bs, shuffle=True)

                # Get client model
                model = self.structure[-1][cid]
                supcon_loss_fn = SupConLoss()
                updated_model, local_proto = client_step(model, loader, device=self.device,
                                                supcon_loss_fn=supcon_loss_fn,
                                                optimizer=self.optimizers[-1][cid],
                                                proj_head=None)
                client_outputs[cid] = local_proto
                self.structure[-1][cid] = updated_model
                # Track communication (MB)
                self.comm_tracker["client_upload_smashed_MB"] += sys.getsizeof(client_outputs[cid]) / 1024

                # Measure CPU + RAM after training client
                cpu_after = psutil.cpu_percent(interval=None)
                mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024**2)  # MB

                # GPU memory usage (MB)
                mem_gpu_used = torch.cuda.max_memory_allocated(device) / (1024**2)

                # Average CPU over before/after
                avg_cpu = (cpu_before + cpu_after) / 2
                mem_used = mem_after - mem_before

                # Record usage
                client_ram_usages.append(mem_used)
                client_cpu_usages.append(avg_cpu)
                client_gpu_ram_usages.append(mem_gpu_used)

                # Free model from GPU
                model.cpu()
                torch.cuda.empty_cache()

            # Compute averages across all clients in this round
            avg_ram = sum(client_ram_usages) / len(client_ram_usages) if client_ram_usages else 0
            avg_cpu = sum(client_cpu_usages) / len(client_cpu_usages) if client_cpu_usages else 0
            avg_gpu_ram = sum(client_gpu_ram_usages) / len(client_gpu_ram_usages) if client_gpu_ram_usages else 0

            # Log results using loguru
            logger.info(f"Average CPU usage per client: {avg_cpu:.2f}%")
            logger.info(f"Average RAM usage per client: {avg_ram:.2f} MB")
            logger.info(f"Average GPU RAM usage per client: {avg_gpu_ram:.2f} MB")

            # Append to global trackers
            client_ram_list.append(avg_ram)
            client_cpu_list.append(avg_cpu)
            client_gpu_ram_list.append(avg_gpu_ram)

