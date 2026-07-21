import copy
import logging
import os
import time
from collections import deque

import numpy as np
import psutil
import torch
from ehsfp.communication import add_communication, mb_of, new_communication_tracker
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


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
    Estimate the size of the gradient sent back (i.e. the model's final output size).
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    numel = output.numel()
    element_size = output.element_size()  # usually 4 bytes (float32)
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

        self.comm_tracker = new_communication_tracker()
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
                # camera-ready: honor a partitioner-provided client->edge mapping
                # (two-level Dirichlet) when present; otherwise random assignment.
                provided = self.args.get("_client_to_edge")
                if provided is not None:
                    for cid in range(num_clients):
                        layer_dict[cid] = copy.deepcopy(self.client_model)
                        conn_dict[cid] = int(provided[cid])
                else:
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
        return mb_of(state_dict)

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
                "edge_model": edge_layer[eid].state_dict(),
            }

            # Estimate upload cost from clients to edge
            size_MB = self.get_model_size(avg_client_model)
            add_communication(self.comm_tracker, "client_to_edge_MB", mb=size_MB, copies=len(cids))

            # Distribute aggregated client model to all clients under this edge
            for cid in cids:
                client_layer[cid].load_state_dict(avg_client_model)
            add_communication(self.comm_tracker, "edge_to_client_MB", mb=size_MB, copies=len(cids))

    def cloud_aggregation(self):
        edge_layer = self.structure[0]
        edge_to_cloud = self.connectivity[0]

        # Reverse mapping: cloud -> list of edge servers
        cloud_to_edges = {}
        for eid, cid in edge_to_cloud.items():
            cloud_to_edges.setdefault(cid, []).append(eid)

        for cid, edge_ids in cloud_to_edges.items():
            edge_models = []

            for eid in edge_ids:
                cache = getattr(self, "edge_cache", {}).get(eid)
                if cache:
                    edge_models.append(cache["edge_model"])
                else:
                    edge_models.append(edge_layer[eid].state_dict())

            if not edge_models:
                continue

            # Aggregate
            avg_edge_model = self.average_state_dicts(edge_models)

            # Communication tracking
            size_edge_MB = self.get_model_size(avg_edge_model)
            add_communication(self.comm_tracker, "edge_to_cloud_MB", mb=size_edge_MB, copies=len(edge_ids))
            add_communication(self.comm_tracker, "cloud_to_edge_MB", mb=size_edge_MB, copies=len(edge_ids))

            # Send back aggregated models
            for eid in edge_ids:
                edge_layer[eid].load_state_dict(avg_edge_model)

    def print_comm_report(self):
        print("\n=== Communication Report ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

    def _make_optimizer(self, model):
        """Build an optimizer matching the configured strategy."""
        lr = self.args["lr"]
        wd = self.args.get("weight_decay", 0.0)
        if self.args.get("optimizer", "adam") == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=self.args.get("momentum", 0.9)
        )

    def _snapshot_pipeline(self):
        """Return a CPU FullPipelineModel from a representative client path."""
        client_model = self.structure[-1][0]
        eid = self.connectivity[-1][0]
        edge_model = self.structure[0][eid]
        cloud_model = self.structure[len(self.args["mid_server"])][0]
        return FullPipelineModel(
            copy.deepcopy(client_model).cpu(),
            copy.deepcopy(edge_model).cpu(),
            copy.deepcopy(cloud_model).cpu(),
        )

    def _validate(self, valid_dataset, device, best_val_top1, epoch):
        """Evaluate the end-to-end pipeline (client 0 -> its edge -> cloud)."""
        client_model = self.structure[-1][0].to(device).eval()
        eid = self.connectivity[-1][0]
        edge_model = self.structure[0][eid].to(device).eval()
        cloud_model = self.structure[len(self.args["mid_server"])][0].to(device).eval()

        loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
        all_preds, all_targets = [], []
        with torch.no_grad():
            for data, target in loader:
                data = data.to(device)
                logits = cloud_model(edge_model(client_model(data)))
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_targets.extend(target.numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        snap = None
        # Preserve the historical accuracy-based checkpoint selection.
        if accuracy >= best_val_top1:
            best_val_top1 = accuracy
            snap = FullPipelineModel(
                copy.deepcopy(client_model).cpu(),
                copy.deepcopy(edge_model).cpu(),
                copy.deepcopy(cloud_model).cpu(),
            )
            print(f"Save best weight at epoch {epoch} with Acc: {accuracy * 100:.2f} %")

        client_model.train()
        edge_model.train()
        cloud_model.train()
        return f1, accuracy, best_val_top1, snap

    def train_end_to_end(
        self,
        train_dataset,
        valid_dataset,
        test_dataset,
        user_groups,
        config,
        epochs,
    ):
        """Hierarchical Split FL training.

        Each selected client performs split-learning forward/backward through the
        connected client -> edge -> cloud pipeline, so *all three tiers* receive
        gradients (the previous implementation detached every tier and only ever
        trained the cloud). Client models are FedAvg-aggregated at the edge every
        t1 rounds and edge models are aggregated at the cloud every t2 rounds.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss().to(device)

        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]
        local_ep = int(config.get("local_ep", 1))
        cloud_layer = len(self.args["mid_server"])

        # Server-side models (edges + cloud) are persistent; keep them on-device
        # with persistent optimizers (their parameter tensors survive in-place
        # load_state_dict aggregation, so the optimizer state stays valid).
        cloud_model = self.structure[cloud_layer][0].to(device)
        cloud_opt = self._make_optimizer(cloud_model)
        edge_opts = {}
        for eid, edge_model in self.structure[0].items():
            edge_model.to(device)
            edge_opts[eid] = self._make_optimizer(edge_model)

        train_f1, train_accuracy, train_loss = [], [], []
        client_cpu_list, client_ram_list, client_gpu_ram_list = [], [], []
        edge_cpu_list, edge_ram_list, edge_gpu_ram_list = [], [], []
        cloud_cpu_list, cloud_ram_list, cloud_gpu_ram_list = [], [], []
        client_compute_times = []
        best_f1 = 0.0
        best_val_top1 = 0.0
        best_pipeline_model = None

        for epoch in tqdm(range(1, epochs + 1)):
            start_time = time.time()
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)

            cloud_model.train()
            client_cpu_usages, client_ram_usages, client_gpu_ram_usages = [], [], []
            epoch_losses = []

            for cid in idxs_users:
                self.client_cache.append(cid)
                eid = self.connectivity[-1][cid]

                cpu_before = psutil.cpu_percent(interval=None)
                mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()

                client_model = self.structure[-1][cid].to(device).train()
                edge_model = self.structure[0][eid].to(device).train()
                client_opt = self._make_optimizer(client_model)
                edge_opt = edge_opts[eid]

                loader = DataLoader(
                    DatasetSplit(train_dataset, user_groups[cid]),
                    batch_size=local_bs,
                    shuffle=True,
                )

                client_losses = []
                for _ in range(local_ep):
                    for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        client_opt.zero_grad(set_to_none=True)
                        edge_opt.zero_grad(set_to_none=True)
                        cloud_opt.zero_grad(set_to_none=True)

                        # Connected split-learning forward through all 3 tiers
                        smashed_c = client_model(data)
                        smashed_e = edge_model(smashed_c)
                        logits = cloud_model(smashed_e)
                        loss = criterion(logits, target)
                        loss.backward()

                        client_opt.step()
                        edge_opt.step()
                        cloud_opt.step()
                        client_losses.append(loss.item())

                        # Communication: smashed activations up; equal-size grads down
                        c_MB = smashed_c.numel() * smashed_c.element_size() / (1024**2)
                        e_MB = smashed_e.numel() * smashed_e.element_size() / (1024**2)
                        add_communication(self.comm_tracker, "client_to_edge_MB", payload=(smashed_c, target))
                        add_communication(self.comm_tracker, "edge_to_cloud_MB", payload=(smashed_e, target))
                        add_communication(self.comm_tracker, "cloud_to_edge_MB", mb=e_MB)
                        add_communication(self.comm_tracker, "edge_to_client_MB", mb=c_MB)

                epoch_losses.append(
                    float(np.mean(client_losses)) if client_losses else 0.0
                )
                client_model.cpu()

                cpu_after = psutil.cpu_percent(interval=None)
                mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
                client_cpu_usages.append((cpu_before + cpu_after) / 2)
                client_ram_usages.append(max(mem_after - mem_before, 0))
                client_gpu_ram_usages.append(
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else 0.0
                )
                torch.cuda.empty_cache()

            train_loss.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
            client_cpu_list.append(float(np.mean(client_cpu_usages)))
            client_ram_list.append(float(np.mean(client_ram_usages)))
            client_gpu_ram_list.append(float(np.mean(client_gpu_ram_usages)))
            # Edge/cloud are the same physical device here; report coarse round stats
            edge_cpu_list.append(client_cpu_list[-1])
            edge_ram_list.append(client_ram_list[-1])
            edge_gpu_ram_list.append(client_gpu_ram_list[-1])
            cloud_cpu_list.append(psutil.cpu_percent(interval=None))
            cloud_ram_list.append(
                psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            )
            cloud_gpu_ram_list.append(
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            client_compute_times.append(time.time() - start_time)

            # --- Hierarchical FedAvg aggregation of model tiers ---
            if epoch % int(config["t1"]) == 0:
                print("Edge server aggregation...")
                self.edge_server_aggregation()
            if epoch % int(config["t2"]) == 0:
                print("Cloud aggregation...")
                self.cloud_aggregation()

            # --- Validation: after a cloud aggregation, or on the final epoch ---
            if epoch % int(config["t2"]) == 0 or epoch == epochs:
                f1, accuracy, best_val_top1, snap = self._validate(
                    valid_dataset, device, best_val_top1, epoch
                )
                if snap is not None:
                    best_pipeline_model = snap
                train_f1.append(f1)
                train_accuracy.append(accuracy)
                best_f1 = max(best_f1, f1)

                print(f"\n=== Epoch {epoch} | F1: {f1 * 100:.2f} % ===")
                for k, v in self.comm_tracker.items():
                    print(f"{k}: {v:.2f} MB")

                if wandb is not None and wandb.run is not None:
                    wandb.log({
                        "epoch": epoch,
                        "f1": f1,
                        "best_f1": best_f1,
                        "accuracy": accuracy,
                        "best_val_top1": best_val_top1,
                        "train_loss": train_loss[-1],
                        "avg_client_cpu_pct": client_cpu_list[-1],
                        "avg_client_ram_MB": client_ram_list[-1],
                        "avg_client_gpu_ram_MB": client_gpu_ram_list[-1],
                        **{k: v for k, v in self.comm_tracker.items()},
                    })

            torch.cuda.empty_cache()

        if best_pipeline_model is None:
            best_pipeline_model = self._snapshot_pipeline()
        if not train_f1:
            train_f1.append(0.0)
            train_accuracy.append(0.0)

        print("\n=== Communication Summary ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

        return {
            "train_accuracy": train_accuracy,
            "validation_f1": train_f1,
            "best_f1": best_f1,
            "best_val_top1": best_val_top1,
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
            "best_weight": best_pipeline_model,
        }
