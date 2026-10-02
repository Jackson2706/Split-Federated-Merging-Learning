import copy

import numpy as np
import torch
from ehsfp.communication import add_communication, new_communication_tracker
from clients import test_inference
from servers import FedAvgAggregator


class HierarchicalFL:
    """
    Create hierarchical federated learning structure:
    - Clients at the bottom
    - Edge servers in the middle
    - Cloud server at the top

    This class manages weight updates, downloads, and aggregation in a hierarchical FL system.
    """

    def __init__(self, args, global_weights, global_model, test_dataset=None):
        self.args = args
        self.model_template = global_model  # model architecture
        self.global_weights = global_weights
        self.structure = self._build_hierarchy()
        self.total_layers = len(args["mid_server"]) + 1  # +1 for cloud layer
        self.test_dataset = test_dataset
        self.aggregator = FedAvgAggregator(args)
        self.comm_cost_dict = new_communication_tracker()

    def print_structure(self):
        print("\n--- Hierarchical Federated Learning Structure ---")
        for layer_idx, layer in self.structure.items():
            if layer_idx == -1:
                print(f"Layer {layer_idx} - Clients:")
                for client_id in sorted(layer.keys()):
                    print(f"  Client {client_id}")
            elif layer_idx == 0:
                print(f"\nLayer {layer_idx} - Edge Servers:")
                for server_id, (client_set, _) in layer.items():
                    print(
                        f"  Edge Server {server_id} serves Clients: {sorted(list(client_set))}"
                    )
            elif layer_idx == self.total_layers - 1:
                print(f"\nLayer {layer_idx} - Cloud Server:")
                for cloud_id, (edge_set, _) in layer.items():
                    print(
                        f"  Cloud Server {cloud_id} connects Edge Servers: {sorted(list(edge_set))}"
                    )
            else:
                print(f"\nLayer {layer_idx} - Intermediate Servers:")
                for server_id, (sub_set, _) in layer.items():
                    print(
                        f"  Mid Server {server_id} connects to: {sorted(list(sub_set))}"
                    )

    def _build_hierarchy(self):
        """
        Build a dictionary representing the hierarchical structure:
        - Layer -1: clients
        - Layer 0: edge servers
        - Layer N: cloud server
        Returns:
            structure (dict): Hierarchical FL layout
        """
        structure = {}

        def build_layer(layer_idx):
            layer_dict = {}

            if layer_idx == -1:
                # Initialize clients
                for client_id in range(self.args["num_users"]):
                    layer_dict[client_id] = self.global_weights

            elif layer_idx == len(self.args["mid_server"]):
                # Cloud layer connected to all edge servers from the previous layer
                all_edge_servers = list(
                    range(self.args["mid_server"][layer_idx - 1])
                )
                layer_dict[0] = [all_edge_servers, self.global_weights]

            else:
                num_servers = self.args["mid_server"][layer_idx]

                if layer_idx == 0:
                    # Edge servers connected to clients
                    # camera-ready: honor a partitioner-provided client->edge
                    # mapping (two-level Dirichlet) when present.
                    provided = self.args.get("_client_to_edge")
                    if provided is not None:
                        edge_members = {s: set() for s in range(num_servers)}
                        for cid, eid in provided.items():
                            edge_members[int(eid)].add(int(cid))
                        for server_id in range(num_servers):
                            layer_dict[server_id] = [
                                edge_members[server_id],
                                self.global_weights,
                            ]
                    else:
                        all_clients = list(range(self.args["num_users"]))
                        clients_per_server = self.args["num_users"] // num_servers

                        for server_id in range(num_servers):
                            if server_id == num_servers - 1:
                                assigned_clients = set(all_clients)
                            else:
                                assigned_clients = set(
                                    np.random.choice(
                                        all_clients,
                                        clients_per_server,
                                        replace=False,
                                    )
                                )
                            layer_dict[server_id] = [
                                assigned_clients,
                                self.global_weights,
                            ]
                            all_clients = list(set(all_clients) - assigned_clients)

                else:
                    # Middle servers connected to previous layer's servers
                    prev_layer_count = self.args["mid_server"][layer_idx - 1]
                    servers_per_server = prev_layer_count // num_servers
                    all_servers = list(range(prev_layer_count))

                    for server_id in range(num_servers):
                        if server_id == num_servers - 1:
                            assigned_servers = set(all_servers)
                        else:
                            assigned_servers = set(
                                np.random.choice(
                                    all_servers,
                                    servers_per_server,
                                    replace=False,
                                )
                            )
                        layer_dict[server_id] = [
                            assigned_servers,
                            self.global_weights,
                        ]
                        all_servers = list(set(all_servers) - assigned_servers)

            structure[layer_idx] = layer_dict
            if layer_idx < len(self.args["mid_server"]):
                build_layer(layer_idx + 1)

        build_layer(-1)  # Start from clients
        return structure

    def get_model_for_client(self, client_id, download_from_edge=True):
        """
        Get the model for a client from its associated edge server.

        Args:
            client_id (int): Client index
            download_from_edge (bool): If False, return client's current model

        Returns:
            model (torch.nn.Module): Model to use
            edge_server_id (int): ID of the associated edge server
        """
        edge_servers = self.structure[0]
        model = copy.deepcopy(self.model_template)

        for server_id, (client_set, weights) in edge_servers.items():
            if client_id in client_set:
                model_weights = copy.deepcopy(
                    weights
                    if download_from_edge
                    else self.structure[-1][client_id]
                )
                model.load_state_dict(model_weights)

                
                if download_from_edge:
                    add_communication(self.comm_cost_dict, "edge_to_client_MB", payload=model_weights)
                return model, server_id

    def upload_client_weights(self, client_weights):
        """
        Upload weights from clients to edge servers and perform hierarchical aggregation.
        Track communication cost in MB by role and direction.
        """

        def _get_weight_size_mb(weights):
            return sum(torch.numel(v) for v in weights.values()) * 4 / (1024**2)

        def aggregate_upward(layer_idx):
            if layer_idx == self.total_layers:
                return

            if layer_idx == 0:
                # Clients → Edge Servers
                for edge_server_id, weight_list in client_weights.items():
                    self.structure[layer_idx][edge_server_id][1] = (
                        self.aggregator.aggregate(None, None, weight_list)
                    )

                    for client_weights_dict in weight_list:
                        add_communication(self.comm_cost_dict, "client_to_edge_MB", payload=client_weights_dict)

            else:
                # Edge Servers → Next Tier (or Cloud)
                for server_id in self.structure[layer_idx]:
                    lower_ids = self.structure[layer_idx][server_id][0]
                    lower_weights = [
                        self.structure[layer_idx - 1][lower_id][1]
                        for lower_id in lower_ids
                    ]
                    self.structure[layer_idx][server_id][1] = (
                        self.aggregator.aggregate(None, None, lower_weights)
                    )

                    for w in lower_weights:
                        add_communication(self.comm_cost_dict, "edge_to_cloud_MB", payload=w)
            aggregate_upward(layer_idx+1)
        aggregate_upward(0)

    def manage_models_top_down(self):
        """
        Propagate best-performing models from cloud to edge (top-down).
        """

        def is_better(w1, w2):
            model = self.model_template
            model.load_state_dict(w1)
            acc1, _ = test_inference(self.args, model, self.test_dataset)
            model.load_state_dict(w2)
            acc2, _ = test_inference(self.args, model, self.test_dataset)
            return acc1 >= acc2

        def propagate(layer_idx):
            if layer_idx == 0:
                return

            for server_id in self.structure[layer_idx]:
                top_weights = self.structure[layer_idx][server_id][1]
                for lower_id in self.structure[layer_idx][server_id][0]:
                    lower_weights = self.structure[layer_idx - 1][lower_id][1]
                    if is_better(top_weights, lower_weights):
                        self.structure[layer_idx - 1][lower_id][1] = (
                            copy.deepcopy(top_weights)
                        )
                        add_communication(self.comm_cost_dict, "cloud_to_edge_MB", payload=top_weights)

            propagate(layer_idx - 1)

        propagate(self.total_layers - 1)

    def save_all_models(self):
        """
        Save the current model state for all clients and servers.
        """
        for layer_idx in range(-1, self.total_layers):
            if layer_idx == -1:
                for client_id in self.structure[layer_idx]:
                    for edge_server_id, (clients, _) in self.structure[
                        0
                    ].items():
                        if client_id in clients:
                            break
                    model_path = f"../save/trained_models/edge{edge_server_id}_client{client_id}.pth"
                    torch.save(self.structure[layer_idx][client_id], model_path)
            else:
                for server_id in self.structure[layer_idx]:
                    name = (
                        "cloud"
                        if layer_idx == self.total_layers - 1
                        else f"edge{layer_idx}"
                    )
                    model_path = (
                        f"../save/trained_models/{name}_server{server_id}.pth"
                    )
                    torch.save(
                        self.structure[layer_idx][server_id][1], model_path
                    )

    def get_communication_status(self):
        return self.comm_cost_dict
