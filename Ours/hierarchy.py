import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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
        client_to_edge = self.connectivity[-1]

        edge_to_clients = {}
        for cid, eid in client_to_edge.items():
            edge_to_clients.setdefault(eid, []).append(cid)

        for cid in self.client_cache:
            client_models = [client_layer[cid].state_dict()]

        avg_model = self.average_state_dicts(client_models)

        size_MB = self.get_model_size(avg_model)
        self.comm_tracker["client_model_upload_MB"] += (
            len(self.client_cache) * size_MB
        )
        
        for _, client in client_layer.items():
            client.load_state_dict(avg_model)
        self.comm_tracker["client_model_download_MB"] += (
            len(client_layer.items()) * size_MB
        )
        self.client_cache = []

    def cloud_aggregation(self):
        edge_layer = self.structure[0]
        edge_to_cloud = self.connectivity[0]

        cloud_to_edges = {}
        for eid, cid in edge_to_cloud.items():
            cloud_to_edges.setdefault(cid, []).append(eid)

        for cid, edge_ids in cloud_to_edges.items():
            edge_models = [edge_layer[eid].state_dict() for eid in edge_ids]
            avg_model = self.average_state_dicts(edge_models)

            size_MB = self.get_model_size(avg_model)
            self.comm_tracker["edge_model_upload_MB"] += len(edge_ids) * size_MB
            self.comm_tracker["edge_model_download_MB"] += (
                len(edge_ids) * size_MB
            )

            for eid in edge_ids:
                edge_layer[eid].load_state_dict(avg_model)

    def print_comm_report(self):
        print("\n=== Communication Report ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

    def initialize_optimizers(self):
        self.optimizers = {}
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {
                nid: torch.optim.SGD(
                    model.parameters(), lr=self.args.get("lr", 0.01)
                )
                for nid, model in nodes.items()
            }

    def train_end_to_end(
        self, train_dataset, valid_dataset, user_groups, config, epochs
    ):
        self.initialize_optimizers()
        criterion = torch.nn.NLLLoss()
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

            self.optimizers[len(self.args["mid_server"])][0].zero_grad()
            loss.backward()
            self.optimizers[len(self.args["mid_server"])][0].step()

            cloud_model.cpu()
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
            else:
                continue
            if epoch % int(config["t2"]) == 0:
                print("Cloud aggregation...")
                self.cloud_aggregation()
            else:
                continue
            metrics_by_cid = {}
            list_acc, list_loss = [], []
            for cid in range(config["num_users"]):
                client_model = self.structure[-1][cid]
                eid = self.connectivity[-1][cid]
                edge_model = self.structure[0][eid]
                cloud_model = self.structure[len(self.args["mid_server"])][0]

                client_model.to(device).eval()
                edge_model.to(device).eval()
                cloud_model.to(device).eval()

                loader = DataLoader(
                    valid_dataset, batch_size=256, shuffle=False
                )

                correct, total, total_loss = 0, 0, 0.0
                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        out_c = client_model(data)
                        out_e = edge_model(out_c)
                        out_cl = cloud_model(out_e)

                        loss = criterion(out_cl, target)
                        pred = out_cl.argmax(dim=1)

                        total_loss += loss.item() * data.size(0)
                        correct += pred.eq(target).sum().item()
                        total += data.size(0)

                acc = correct / total
                avg_loss = total_loss / total
                list_acc.append(acc)
                list_loss.append(avg_loss)
                metrics_by_cid[cid] = (acc, avg_loss)
                client_model.cpu()
                edge_model.cpu()
                cloud_model.cpu()
                torch.cuda.empty_cache()
            train_accuracy.append(sum(list_acc) / len(list_acc))
            train_loss.append(sum(list_loss) / len(list_loss))
            best_cid = max(
                metrics_by_cid, key=lambda cid: metrics_by_cid[cid][0]
            )
            print(
                f"Best model pipeline from client {best_cid} with accuracy {metrics_by_cid[best_cid][0]:.4f}"
            )
            best_client = self.structure[-1][best_cid]
            best_edge = self.structure[0][self.connectivity[-1][best_cid]]
            best_cloud = self.structure[len(self.args["mid_server"])][
                0
            ]  # assuming 1 cloud
            best_client.cpu()
            best_edge.cpu()
            best_cloud.cpu()
            pipeline_model = FullPipelineModel(
                client_model=copy.deepcopy(best_client),
                edge_model=copy.deepcopy(best_edge),
                cloud_model=copy.deepcopy(best_cloud),
            )
            print("\n=== Communication Summary ===")
            for k, v in self.comm_tracker.items():
                print(f"{k}: {v:.2f} MB")
        print("\n=== Communication Summary ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

        return {
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "best_weight": pipeline_model,
        }
