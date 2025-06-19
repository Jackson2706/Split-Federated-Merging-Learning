import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image.clone(), torch.tensor(label)


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
        }
        self.client_cache = []

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


    def compute_model_delta(self, local_state, global_state):
        return {k: local_state[k] - global_state[k] for k in global_state}

    def fednova_aggregate(self, deltas, steps, global_state):
        total = sum(steps)
        delta_sum = {
            k: sum(d[k] * (steps[i] / total) for i, d in enumerate(deltas))
            for k in global_state
        }
        return {k: global_state[k] + delta_sum[k] for k in global_state}

    def aggregate_with_weights(self, models, weights):
        return {
            k: sum(weights[i] * m[k] for i, m in enumerate(models))
            for k in models[0]
        }

    def get_model_size(self, state_dict):
        return sum(p.numel() for p in state_dict.values()) * 4 / 1e6  # MB

    def edge_server_aggregation(self):
        client_layer = self.structure[-1]
        edge_layer = self.structure[0]
        client_to_edge = self.connectivity[-1]

        edge_to_clients = {}
        for cid in self.client_cache:
            eid = client_to_edge[cid]
            edge_to_clients.setdefault(eid, []).append(cid)

        self.edge_cache = {}

        for eid, cids in edge_to_clients.items():
            client_deltas = []
            local_steps = []
            global_state = self.client_weight

            for cid in cids:
                local_state = client_layer[cid].state_dict()
                delta = self.compute_model_delta(local_state, global_state)
                client_deltas.append(delta)
                local_steps.append(
                    self.args.get("local_steps", 10)
                )  # fake value

            avg_client_model = self.fednova_aggregate(
                client_deltas, local_steps, global_state
            )
            self.client_weight = avg_client_model
            self.edge_cache[eid] = {
                "client_model": avg_client_model,
                "edge_model": edge_layer[eid].state_dict(),
            }

            size_MB = self.get_model_size(avg_client_model)
            self.comm_tracker["client_model_upload_MB"] += len(cids) * size_MB

            for cid in cids:
                client_layer[cid].load_state_dict(avg_client_model)
            self.comm_tracker["client_model_download_MB"] += len(cids) * size_MB

        self.client_cache = []

    def shaa_cloud_aggregation(self, edge_val_scores):
        edge_layer = self.structure[0]
        cloud_layer = self.structure[1][0]
        client_layer = self.structure[-1]
        edge_to_cloud = self.connectivity[0]
        client_to_edge = self.connectivity[-1]

        cloud_to_edges = {}
        for eid, cid in edge_to_cloud.items():
            cloud_to_edges.setdefault(cid, []).append(eid)

        for cid, edge_ids in cloud_to_edges.items():
            edge_models = []
            client_models = []
            scores = []

            for eid in edge_ids:
                cache = self.edge_cache.get(eid)
                if cache:
                    edge_models.append(cache["edge_model"])
                    client_models.append(cache["client_model"])
                    scores.append(edge_val_scores.get(eid, 1.0))

            weights = F.softmax(
                torch.tensor(scores, dtype=torch.float32) ** 2, dim=0
            ).tolist()

            avg_edge_model = self.aggregate_with_weights(edge_models, weights)
            avg_client_model = self.aggregate_with_weights(
                client_models, weights
            )

            size_edge_MB = self.get_model_size(avg_edge_model)
            size_client_MB = self.get_model_size(avg_client_model)
            self.comm_tracker["edge_model_upload_MB"] += len(edge_ids) * (
                size_edge_MB + size_client_MB
            )
            self.comm_tracker["edge_model_download_MB"] += len(edge_ids) * (
                size_edge_MB + size_client_MB
            )

            for eid in edge_ids:
                edge_layer[eid].load_state_dict(avg_edge_model)
                for cid, eid_check in client_to_edge.items():
                    if eid_check == eid:
                        client_layer[cid].load_state_dict(avg_client_model)
                self.comm_tracker["client_model_download_MB"] += (
                    sum(
                        1
                        for eid_check in client_to_edge.values()
                        if eid_check == eid
                    )
                    * size_client_MB
                )

    def initialize_optimizers(self):
        self.optimizers = {}
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {
                nid: torch.optim.Adam(model.parameters(), lr=self.args["lr"])
                for nid, model in nodes.items()
            }

    def train_end_to_end(
        self, train_dataset, valid_dataset, user_groups, config, epochs
    ):
        self.initialize_optimizers()
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]
        train_accuracy, train_loss = [], []

        for epoch in tqdm(range(1, epochs + 1)):
            print(f"\n=== Epoch {epoch} ===")
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)

            client_outputs = {}
            for cid in idxs_users:
                self.client_cache.append(cid)
                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(
                    local_data, batch_size=local_bs, shuffle=True
                )

                model = self.structure[-1][cid]
                model.to(device).eval()

                feats, labels = [], []
                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        out = model(data)
                        feats.append(out.cpu())
                        labels.append(target.cpu())

                fx, fy = torch.cat(feats), torch.cat(labels)
                client_outputs[cid] = (fx, fy)

                model.cpu()
                torch.cuda.empty_cache()

            edge_outputs = {}
            edge_to_clients = {}
            for cid in client_outputs:
                eid = self.connectivity[-1][cid]
                edge_to_clients.setdefault(eid, []).append(cid)

            for eid, cids in edge_to_clients.items():
                model = self.structure[0][eid]
                model.to(device).train()
                X = torch.cat(
                    [client_outputs[cid][0] for cid in cids], dim=0
                ).to(device)
                Y = torch.cat(
                    [client_outputs[cid][1] for cid in cids], dim=0
                ).to(device)
                out = model(X)
                size_MB = (X.numel() + Y.numel()) * 4 / (1024**2)
                self.comm_tracker["client_upload_smashed_MB"] += size_MB
                edge_outputs[eid] = (out.detach().cpu(), Y.detach().cpu())
                model.cpu()
                del X, Y, out
                torch.cuda.empty_cache()

            del client_outputs

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
            print(f"Cloud Loss: {loss.item():.4f}")
            train_loss.append(loss)
            self.optimizers[len(self.args["mid_server"])][0].zero_grad()
            loss.backward()
            self.optimizers[len(self.args["mid_server"])][0].step()

            cloud_model.cpu()
            del all_X, all_Y, pred, loss
            torch.cuda.empty_cache()

            if epoch % int(config["t1"]) == 0:
                print("Edge aggregation via FedNova...")
                self.edge_server_aggregation()

            if epoch % int(config["t2"]) == 0:
                print("Cloud aggregation via SHAA...")
                edge_val_scores = self.evaluate_edge_models(valid_dataset, batch_size=256)
                average_acc = sum(edge_val_scores.values()) / len(edge_val_scores)
                train_accuracy.append(average_acc)
                self.shaa_cloud_aggregation(edge_val_scores)
                print(f"AvgAcc: {average_acc}")
        print("\nTraining completed.")
        client_model = self.structure[-1][0]
        eid = self.connectivity[-1][0]
        edge_model = self.structure[0][eid]
        cloud_model = self.structure[len(self.args["mid_server"])][0]
        client_model.to(device).eval()
        edge_model.to(device).eval()
        cloud_model.to(device).eval()
        fullmodel = FullPipelineModel(client_model, edge_model, cloud_model)
        return {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "best_weight": fullmodel
        }

    def evaluate_edge_models(self, valid_dataset, batch_size=256):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        client_layer = self.structure[-1]
        edge_layer = self.structure[0]
        cloud_layer = self.structure[1][0].to(device)
        cloud_layer.eval()

        scores = {}
        for eid, edge_model in edge_layer.items():
            clients = [cid for cid, e in self.connectivity[-1].items() if e == eid]
            if not clients:
                continue
            cid = clients[0]  # use one representative client
            client_model = client_layer[cid].to(device).eval()
            edge_model = edge_model.to(device).eval()

            loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    out = cloud_layer(edge_model(client_model(data)))
                    pred = out.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

            scores[eid] = correct / total if total > 0 else 0.0
            client_model.cpu()
            edge_model.cpu()
        cloud_layer.cpu()
        return scores
