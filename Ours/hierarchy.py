import numpy as np
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging  # Make sure this is at the top of your script/module

from clients import test_inference
from servers import FedAvgAggregator


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
    def __init__(self, args, client_model, client_weights, edge_model, edge_weights,
                 cloud_model, cloud_weight, test_dataset=None):
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
        self.aggregator = FedAvgAggregator(args)

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
                    assigned = all_clients if edge_id == num_edges - 1 else list(
                        np.random.choice(all_clients, clients_per_edge, replace=False)
                    )
                    for cid in assigned:
                        layer_dict[cid] = copy.deepcopy(self.client_model)
                        conn_dict[cid] = edge_id
                    all_clients = list(set(all_clients) - set(assigned))

            elif layer_idx == len(self.args["mid_server"]):
                layer_dict[0] = copy.deepcopy(self.cloud_model)
                for mid_id in range(self.args["mid_server"][-1]):
                    conn_dict[mid_id] = 0  # one cloud

            else:
                num_servers = self.args["mid_server"][layer_idx]
                prev_layer_count = (
                    self.args["num_users"] if layer_idx == 0
                    else self.args["mid_server"][layer_idx - 1]
                )
                servers_per_layer = prev_layer_count // num_servers
                all_prev = list(range(prev_layer_count))

                for sid in range(num_servers):
                    assigned = all_prev if sid == num_servers - 1 else list(
                        np.random.choice(all_prev, servers_per_layer, replace=False)
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
        """
        Pretty-print the hierarchical FL structure using logging.
        """
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
                        else (
                            "Cloud Model"
                            if model.__class__ == self.cloud_model.__class__
                            else "Unknown"
                        )
                    )
                )
                logging.info(f"  Node ID {node_id}: {model_type}")

    def initialize_optimizers(self):
        self.optimizers = {}
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {
                nid: torch.optim.SGD(model.parameters(), lr=self.args.get("lr", 0.01))
                for nid, model in nodes.items()
            }

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
        for cid, eid in client_to_edge.items():
            edge_to_clients.setdefault(eid, []).append(cid)

        for eid, client_ids in edge_to_clients.items():
            client_models = [client_layer[cid].state_dict() for cid in client_ids]
            avg_model = self.average_state_dicts(client_models)



            # Push to clients
            for cid in client_ids:
                client_layer[cid].load_state_dict(avg_model)

    def cloud_aggregation(self):
        edge_layer = self.structure[0]
        cloud_layer = self.structure[len(self.args["mid_server"])]
        edge_to_cloud = self.connectivity[0]

        cloud_to_edges = {}
        for eid, cid in edge_to_cloud.items():
            cloud_to_edges.setdefault(cid, []).append(eid)

        for cid, edge_ids in cloud_to_edges.items():
            edge_models = [edge_layer[eid].state_dict() for eid in edge_ids]
            avg_model = self.average_state_dicts(edge_models)



            # Push to edge
            for eid in edge_ids:
                edge_layer[eid].load_state_dict(avg_model)

    def train_end_to_end(self, train_dataset, user_groups, config, epochs):
        self.initialize_optimizers()
        criterion = torch.nn.NLLLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]

        for epoch in tqdm(range(1, epochs + 1)):
            print(f"\n=== Epoch {epoch} ===")

            # 1. Sample users
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)

            client_outputs = {}

            # 2. Clients → Edge (Forward Pass)
            for cid in idxs_users:
                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(local_data, batch_size=local_bs, shuffle=True)

                model = self.structure[-1][cid]
                model.to(device)
                model.eval()

                feats, labels = [], []
                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        out = model(data)
                        feats.append(out.cpu())    # move to CPU
                        labels.append(target.cpu())

                client_outputs[cid] = (torch.cat(feats), torch.cat(labels))
                model.cpu()
                torch.cuda.empty_cache()

            # 3. Edge → Cloud
            edge_outputs = {}
            edge_to_clients = {}
            for cid in client_outputs:
                eid = self.connectivity[-1][cid]
                edge_to_clients.setdefault(eid, []).append(cid)

            for eid, cids in edge_to_clients.items():
                model = self.structure[0][eid]
                model.to(device)
                model.train()

                X = torch.cat([client_outputs[cid][0] for cid in cids], dim=0).to(device)
                Y = torch.cat([client_outputs[cid][1] for cid in cids], dim=0).to(device)

                out = model(X)
                edge_outputs[eid] = (out.detach().cpu(), Y.detach().cpu())

                model.cpu()
                del X, Y, out
                torch.cuda.empty_cache()

            del client_outputs  # free memory

            # 4. Cloud aggregation
            cloud_model = self.structure[len(self.args["mid_server"])][0]
            cloud_model.to(device)
            cloud_model.train()

            all_X = torch.cat([fx for fx, _ in edge_outputs.values()], dim=0).to(device)
            all_Y = torch.cat([fy for _, fy in edge_outputs.values()], dim=0).to(device)

            pred = cloud_model(all_X)
            loss = criterion(pred, all_Y)
            print(f"Cloud Loss: {loss.item():.4f}")

            # 5. Backpropagation
            self.optimizers[len(self.args["mid_server"])][0].zero_grad()
            loss.backward()
            self.optimizers[len(self.args["mid_server"])][0].step()

            cloud_model.cpu()
            del all_X, all_Y, pred, loss
            torch.cuda.empty_cache()

            # Optional optimizer steps (no backprop)
            for eid in edge_outputs:
                self.optimizers[0][eid].zero_grad()
                self.optimizers[0][eid].step()

            for cid in idxs_users:
                self.optimizers[-1][cid].zero_grad()
                self.optimizers[-1][cid].step()

            del edge_outputs
            torch.cuda.empty_cache()

            # 6. Aggregation
            if epoch % int(config["t1"]) == 0:
                print("Edge server aggregation...")
                self.edge_server_aggregation()

            if epoch % int(config["t2"]) == 0:
                print("Cloud aggregation...")
                self.cloud_aggregation()
