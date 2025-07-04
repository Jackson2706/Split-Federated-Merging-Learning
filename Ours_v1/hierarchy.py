import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import deque
import psutil
import os
import time


class FullPipelineModel(nn.Module):
    def __init__(self, client_model, edge_model, cloud_model):
        super().__init__()
        self.client = client_model
        self.edge = edge_model
        self.cloud = cloud_model

    def forward(self, x):
        x = self.client(x)
        x = self.edge(x)
        x = self.cloud(x)
        return x


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image.clone(), torch.tensor(label)


def estimate_gradient_size_MB(model, input_shape, device="cpu"):
    """
    Ước lượng kích thước gradient truyền về (tức kích thước output cuối của model).

    Args:
        model: nn.Module (client, edge, or cloud model)
        input_shape: tuple, ví dụ (3, 32, 32)
        device: 'cuda' hoặc 'cpu'

    Returns:
        size_MB: float - kích thước output cuối cùng theo MB
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    numel = output.numel()
    element_size = output.element_size()  # thường là 4 bytes (float32)
    size_MB = (numel * element_size) / (1024**2)
    return size_MB


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
        client_layer = self.structure[-1]
        edge_layer = self.structure[0]
        client_to_edge = self.connectivity[-1]

        edge_to_clients = {}
        for cid in self.client_cache:
            eid = client_to_edge[cid]
            edge_to_clients.setdefault(eid, []).append(cid)

        self.edge_cache = {}  # Reset edge cache

        for eid, cids in edge_to_clients.items():
            # Aggregate client models
            client_models = [client_layer[cid].state_dict() for cid in cids]
            avg_client_model = self.average_state_dicts(client_models)

            # Save edge model and aggregated client model
            self.edge_cache[eid] = {
                "client_model": avg_client_model,
                "edge_model": edge_layer[eid].state_dict(),
            }

            # Estimate upload cost from clients to edge
            size_MB = self.get_model_size(avg_client_model)
            self.comm_tracker["client_model_upload_MB"] += len(cids) * size_MB

            # Distribute aggregated client model to all clients under this edge
            for cid in cids:
                client_layer[cid].load_state_dict(avg_client_model)
            self.comm_tracker["client_model_download_MB"] += len(cids) * size_MB

        # self.client_cache = []

    def cloud_aggregation(self):
        edge_layer = self.structure[0]
        edge_to_cloud = self.connectivity[0]
        client_layer = self.structure[-1]
        client_to_edge = self.connectivity[-1]

        # Reverse mapping: cloud → list of edge servers
        cloud_to_edges = {}
        for eid, cid in edge_to_cloud.items():
            cloud_to_edges.setdefault(cid, []).append(eid)

        for cid, edge_ids in cloud_to_edges.items():
            edge_models = []
            client_models = []

            for eid in edge_ids:
                cache = self.edge_cache.get(eid)
                if cache:
                    edge_models.append(cache["edge_model"])
                    client_models.append(cache["client_model"])

            # Aggregate
            avg_edge_model = self.average_state_dicts(edge_models)
            avg_client_model = self.average_state_dicts(client_models)

            # Communication tracking
            size_edge_MB = self.get_model_size(avg_edge_model)
            size_client_MB = self.get_model_size(avg_client_model)
            self.comm_tracker["edge_model_upload_MB"] += len(edge_ids) * (
                size_edge_MB + size_client_MB
            )
            self.comm_tracker["edge_model_download_MB"] += len(edge_ids) * (
                size_edge_MB + size_client_MB
            )

            # Send back aggregated models
            for eid in edge_ids:
                # Update edge server with aggregated edge model
                edge_layer[eid].load_state_dict(avg_edge_model)

                # === ✅ Send back aggregated client model to relevant clients ===
                # Find all clients connected to this edge server
                for client_id, edge_id in client_to_edge.items():
                    if edge_id == eid:
                        client_layer[client_id].load_state_dict(
                            avg_client_model
                        )
                self.comm_tracker["client_model_download_MB"] += (
                    sum(
                        1
                        for edge_id in client_to_edge.values()
                        if edge_id == eid
                    )
                    * size_client_MB
                )

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

    def train_end_to_end(
        self, train_dataset, valid_dataset, user_groups, config, epochs
    ):
        self.initialize_optimizers()
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]
        train_f1, train_loss = [], []
        client_cpu_list = []
        client_ram_list = []
        client_gpu_ram_list = []
        edge_cpu_list = []
        edge_ram_list = []
        edge_gpu_ram_list = []
        cloud_cpu_list = []
        cloud_ram_list = []
        cloud_gpu_ram_list = []
        best_f1 = 0
        client_compute_times = []
        input_shape_edge = None
        input_shape_client = None
        for epoch in tqdm(range(1, epochs + 1)):
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)

            start_time = time.time()
            client_cpu_usages = []
            client_ram_usages = []
            client_gpu_ram_usages = []
            client_outputs = {}

            for cid in idxs_users:
                self.client_cache.append(cid)
                cpu_before = psutil.cpu_percent(interval=None)
                mem_before = psutil.Process(os.getpid()).memory_info().rss / (
                    1024**2
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(
                    local_data, batch_size=local_bs, shuffle=True
                )

                model = self.structure[-1][cid]
                model.to(device).eval()

                feats, labels = [], []
                with torch.no_grad():
                    for data, target in loader:
                        if input_shape_client is None:
                            input_shape_client = data.shape
                        data, target = data.to(device), target.to(device)
                        out = model(data)
                        feats.append(out.cpu())
                        labels.append(target.cpu())

                fx, fy = torch.cat(feats), torch.cat(labels)
                client_outputs[cid] = (fx, fy)
                cpu_after = psutil.cpu_percent(interval=None)
                mem_after = psutil.Process(os.getpid()).memory_info().rss / (
                    1024**2
                )

                mem_gpu_used = torch.cuda.max_memory_allocated(device) / (
                    1024**2
                )
                avg_cpu = (cpu_before + cpu_after) / 2
                mem_used = mem_after - mem_before
                client_ram_usages.append(mem_used)
                client_cpu_usages.append(avg_cpu)
                client_gpu_ram_usages.append(mem_gpu_used)
                model.cpu()
                torch.cuda.empty_cache()

            avg_ram = (
                sum(client_ram_usages) / len(client_ram_usages)
                if sum(client_ram_usages) / len(client_ram_usages) > 0
                else 0
            )
            avg_cpu = sum(client_cpu_usages) / len(client_cpu_usages)
            avg_gpu_ram = sum(client_gpu_ram_usages) / len(
                client_gpu_ram_usages
            )

            print(f"Average CPU usage / client: {avg_cpu:.2f} %")
            print(f"Average RAM usage/ client: {avg_ram:.2f} MB")
            print(f"Average GPU RAM usage/ client: {avg_gpu_ram:.2f} MB")

            client_ram_list.append(avg_ram)
            client_cpu_list.append(avg_cpu)
            client_gpu_ram_list.append(avg_gpu_ram)

            edge_outputs = {}
            edge_to_clients = {}
            for cid in client_outputs:
                eid = self.connectivity[-1][cid]
                edge_to_clients.setdefault(eid, []).append(cid)

            edge_cpu_usages = []
            edge_ram_usages = []
            edge_gpu_ram_usages = []
            for eid, cids in edge_to_clients.items():
                cpu_before = psutil.cpu_percent(interval=None)
                mem_before = psutil.Process(os.getpid()).memory_info().rss / (
                    1024**2
                )
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                model = self.structure[0][eid]
                model.to(device).train()
                X = torch.cat(
                    [client_outputs[cid][0] for cid in cids], dim=0
                ).to(device)
                Y = torch.cat(
                    [client_outputs[cid][1] for cid in cids], dim=0
                ).to(device)
                if input_shape_edge is None:
                    input_shape_edge = X.shape
                out = model(X)
                size_MB = (X.numel() + Y.numel()) * 4 / (1024**2)
                self.comm_tracker["client_upload_smashed_MB"] += size_MB
                edge_outputs[eid] = (out.detach().cpu(), Y.detach().cpu())
                model.cpu()

                mem_after = psutil.Process(os.getpid()).memory_info().rss / (
                    1024**2
                )
                cpu_after = psutil.cpu_percent(interval=None)
                mem_gpu_used = torch.cuda.max_memory_allocated(device) / (
                    1024**2
                )
                avg_cpu = (cpu_before + cpu_after) / 2
                mem_used = mem_after - mem_before

                edge_ram_usages.append(mem_used)
                edge_cpu_usages.append(avg_cpu)
                edge_gpu_ram_usages.append(mem_gpu_used)
                del X, Y, out
                torch.cuda.empty_cache()

            del client_outputs
            avg_edge_ram = (
                sum(edge_ram_usages) / len(edge_ram_usages)
                if sum(edge_ram_usages) / len(edge_ram_usages) > 0
                else 0
            )
            avg_edge_cpu = sum(edge_cpu_usages) / len(edge_cpu_usages)
            avg_edge_gpu_ram = sum(edge_gpu_ram_usages) / len(
                edge_gpu_ram_usages
            )
            edge_cpu_list.append(avg_edge_cpu)
            edge_ram_list.append(avg_edge_ram)
            edge_gpu_ram_list.append(avg_edge_gpu_ram)

            print(f"Average Edge CPU usage: {avg_edge_cpu:.2f} %")
            print(f"Average Edge RAM usage: {avg_edge_ram:.2f} MB")
            print(f"Average Edge GPU RAM usage: {avg_edge_gpu_ram:.2f} MB")

            cloud_cpu_usage = psutil.cpu_percent(interval=None)
            cloud_ram_usage = psutil.Process(os.getpid()).memory_info().rss / (
                1024**2
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            cloud_model = self.structure[len(self.args["mid_server"])][0]
            cloud_model.to(device).train()

            all_X = torch.cat(
                [fx for fx, _ in edge_outputs.values()], dim=0
            ).to(device)
            all_Y = torch.cat(
                [fy for _, fy in edge_outputs.values()], dim=0
            ).to(device)
            size_MB = (all_X.numel() + all_Y.numel()) * 4 / (1024**2)
            self.comm_tracker["edge_upload_smashed_MB"] += size_MB
            pred = cloud_model(all_X)
            loss = criterion(pred, all_Y)

            self.optimizers[len(self.args["mid_server"])][0].zero_grad()
            loss.backward()
            self.optimizers[len(self.args["mid_server"])][0].step()

            # === Track gradient sent from cloud → edge ===
            for eid in edge_outputs:
                edge_model = self.structure[0][eid].to(device)
                grad_to_edge_MB = estimate_gradient_size_MB(edge_model, input_shape_edge)
                self.comm_tracker["cloud_download_grad_MB"] += grad_to_edge_MB

            # === Track gradient sent from edge → client ===
            for cid in idxs_users:
                client_model = self.structure[-1][cid].to(device)
                grad_to_client_MB = estimate_gradient_size_MB(client_model, input_shape_client)
                self.comm_tracker["edge_download_grad_MB"] += grad_to_client_MB
            cloud_model.cpu()
            end_time = time.time()
            cloud_ram_usage = (
                (
                    psutil.Process(os.getpid()).memory_info().rss / (1024**2)
                    - cloud_ram_usage
                )
                if psutil.Process(os.getpid()).memory_info().rss / (1024**2)
                - cloud_ram_usage
                > 0
                else 0
            )
            cloud_cpu_usage = (
                psutil.cpu_percent(interval=None) + cloud_cpu_usage
            ) / 2
            mem_gpu_used = torch.cuda.max_memory_allocated(device) / (1024**2)
            time_taken = end_time - start_time
            cloud_cpu_list.append(cloud_cpu_usage)
            cloud_ram_list.append(cloud_ram_usage)
            cloud_gpu_ram_list.append(mem_gpu_used)
            client_compute_times.append(time_taken)
            print(f"Cloud CPU usage: {cloud_cpu_usage:.2f} %")
            print(f"Cloud RAM usage: {cloud_ram_usage:.2f} MB")
            print(f"Cloud GPU RAM usage: {mem_gpu_used:.2f} MB")
            print(f"Time taken for cloud aggregation: {time_taken:.2f} seconds")
            del all_X, all_Y, pred, loss
            torch.cuda.empty_cache()

            for eid in edge_outputs:
                self.optimizers[0][eid].zero_grad()
                self.optimizers[0][eid].step()

            for cid in idxs_users:
                self.optimizers[-1][cid].zero_grad()
                self.optimizers[-1][cid].step()

            del edge_outputs
            torch.cuda.empty_cache()
            if epoch % int(config["t1"]) == 0:
                print("Edge server aggregation...")
                self.edge_server_aggregation()
            elif epoch == 1:
                pass
            else:
                continue

            if epoch % int(config["t2"]) == 0:
                print("Edge server aggregation...")
                self.edge_server_aggregation()
                print("Cloud aggregation...")
                self.cloud_aggregation()
            elif epoch == 1:
                pass
            else:
                continue
            list_f1, list_loss = [], []
            client_model = self.structure[-1][cid]
            eid = self.connectivity[-1][0]
            edge_model = self.structure[0][eid]
            cloud_model = self.structure[len(self.args["mid_server"])][0]
            client_model.to(device).eval()
            edge_model.to(device).eval()
            cloud_model.to(device).eval()
            loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

            from sklearn.metrics import f1_score

            all_preds = []
            all_targets = []
            total_loss = 0.0
            criterion = nn.NLLLoss()
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    out_c = client_model(data)
                    out_e = edge_model(out_c)
                    out_cl = cloud_model(out_e)
                    pred = out_cl.argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

                    out_cl = nn.functional.log_softmax(out_cl, dim=1)
                    loss = criterion(out_cl, target)

                    total_loss += loss.item() * data.size(0)
            # Compute F1 score (macro, micro, or weighted depending on your task)
            f1 = f1_score(
                all_targets, all_preds, average="macro"
            )  # change 'macro' if needed
            avg_loss = total_loss / len(loader.dataset)
            list_f1.append(f1)
            list_loss.append(avg_loss)
            torch.cuda.empty_cache()
            train_f1.append(sum(list_f1) / len(list_f1))
            train_loss.append(sum(list_loss) / len(list_loss))
            if best_f1 < f1:
                print(
                    f"Save best weight at epoch {epoch} with f1: {f1 * 100:.2f} %"
                )
                pipeline_model = FullPipelineModel(
                    client_model=copy.deepcopy(client_model),
                    edge_model=copy.deepcopy(edge_model),
                    cloud_model=copy.deepcopy(cloud_model),
                )
                best_f1 = f1
            else:
                continue

            print("\n=== Communication Accuracy Summary ===")
            print(f"F1: {f1 * 100} %")
            for k, v in self.comm_tracker.items():
                print(f"{k}: {v:.2f} MB")
        print("\n=== Communication Summary ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

        return {
            "train_accuracy": train_f1,
            "train_loss": train_loss,
            "client_cpu": client_cpu_list,
            "client_ram": client_ram_list,
            "client_gpu_ram": client_gpu_ram_list,
            "edge_cpu": edge_cpu_list,
            "edge_ram": edge_ram_list,
            "edge_gpu_ram": edge_gpu_ram_list,
            "cloud_cpu": cloud_cpu_list,
            "cloud_ram": cloud_ram_list,
            "cloud_gpu_ram": cloud_gpu_ram_list,
            "client_time_list": client_compute_times,
            "best_weight": pipeline_model,
        }
