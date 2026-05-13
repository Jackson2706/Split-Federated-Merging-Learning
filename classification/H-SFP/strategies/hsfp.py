"""
H-SFP: Hierarchical Split-Federated Prototype strategy.

This module implements the core H-SFP training pipeline:
  Phase 1: Client SSL training + prototype extraction
  Phase 2: Edge SSL training + prototype extraction
  Phase 3: Cloud supervised training
  Phase 4: Aggregation (FedAvg at edge and cloud levels)
  Phase 5: Validation
"""

import copy
import gc
import logging
import os
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from core.aggregation import average_state_dicts
from core.metrics import get_model_size_MB, get_proto_dist_size_MB
from core.prototype import (
    aggregate_prototypes_and_generate_data,
    calculate_prototypes_and_distribution,
)
from core.ssl import (
    build_client_ssl_transforms,
    build_edge_ssl_transforms,
)
from core.losses import supervised_contrastive_loss
from data.datasets import DatasetSplit
from models.pipeline import FullPipelineModel
from optimizers.lars import LARS
from strategies.base import BaseStrategy


class HierarchicalFL(BaseStrategy):
    """
    Hierarchical Split-Federated Learning with Prototype-based communication.

    Architecture: Client (SSL) → Edge (SSL) → Cloud (Supervised)
    Communication: Prototypes + distributions instead of raw activations.
    Aggregation: FedAvg at edge and cloud tiers on configurable intervals.
    """

    def __init__(self, config, client_model, edge_model, cloud_model, test_dataset=None):
        super().__init__(config, client_model, edge_model, cloud_model, test_dataset)

        self.structure, self.connectivity = self.build_hierarchy()
        self.total_layers = len(config["mid_server"]) + 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_transforms = build_client_ssl_transforms().to(self.device)
        self.ssl_transforms_edge = build_edge_ssl_transforms().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Communication cost tracker
        self.comm_tracker = {
            "client_to_edge_data_MB": 0.0,
            "edge_to_cloud_data_MB": 0.0,
            "client_model_upload_MB": 0.0,
            "client_model_download_MB": 0.0,
            "edge_model_upload_MB": 0.0,
            "edge_model_download_MB": 0.0,
            "total_comm_MB": 0.0,
        }

        self.client_cache = deque(maxlen=20)
        self.optimizers = {}

        self.input_shape_client = None
        self.input_shape_edge = None
        self.input_shape_cloud = None

        self.num_workers = 2 if os.name != 'nt' else 0
        print(f"Using {self.num_workers} workers for DataLoader.")

    # =========================================================================
    # Hierarchy Construction
    # =========================================================================

    def build_hierarchy(self):
        """Build the client → edge → cloud topology from config."""
        structure = {}
        connectivity = {}

        def build_layer(layer_idx):
            layer_dict = {}
            conn_dict = {}

            if layer_idx == -1:
                num_clients = self.config["num_users"]
                num_edges = self.config["mid_server"][0]
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

            elif layer_idx == len(self.config["mid_server"]):
                layer_dict[0] = copy.deepcopy(self.cloud_model)
                for mid_id in range(self.config["mid_server"][-1]):
                    conn_dict[mid_id] = 0

            else:
                num_servers = self.config["mid_server"][layer_idx]
                prev_layer_count = (
                    self.config["num_users"]
                    if layer_idx == 0
                    else self.config["mid_server"][layer_idx - 1]
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

            if layer_idx < len(self.config["mid_server"]):
                build_layer(layer_idx + 1)

        build_layer(-1)
        return structure, connectivity

    def print_structure(self):
        """Print the hierarchy topology."""
        for layer_idx in sorted(self.structure.keys()):
            layer_nodes = self.structure[layer_idx]
            layer_type = (
                "Client Layer"
                if layer_idx == -1
                else (
                    "Cloud Layer"
                    if layer_idx == len(self.config["mid_server"])
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

    # =========================================================================
    # Aggregation (Phase 4)
    # =========================================================================

    def edge_server_aggregation(self):
        """FedAvg aggregation at the edge level."""
        print("Edge Aggregation Started")
        client_layer = self.structure[-1]
        edge_layer = self.structure[0]
        client_to_edge = self.connectivity[-1]

        edge_to_clients = {}
        for cid in self.client_cache:
            eid = client_to_edge[cid]
            edge_to_clients.setdefault(eid, []).append(cid)

        self.edge_cache = {}

        for eid, cids in edge_to_clients.items():
            client_models_states = [client_layer[cid].state_dict() for cid in cids]
            if not client_models_states:
                continue

            avg_client_model = average_state_dicts(client_models_states)
            self.edge_cache[eid] = {"edge_model": edge_layer[eid].state_dict()}

            size_MB = get_model_size_MB(avg_client_model)
            self.comm_tracker["client_model_upload_MB"] += len(cids) * size_MB

            for cid in cids:
                client_layer[cid].load_state_dict(avg_client_model)
            self.comm_tracker["client_model_download_MB"] += len(cids) * size_MB

    def cloud_aggregation(self):
        """FedAvg aggregation at the cloud level."""
        print("Cloud Aggregation Started")
        edge_layer = self.structure[0]
        edge_to_cloud = self.connectivity[0]
        cloud_to_edges = {}
        for eid, cid in edge_to_cloud.items():
            cloud_to_edges.setdefault(cid, []).append(eid)

        for cid, edge_ids in cloud_to_edges.items():
            edge_models_states = []
            for eid in edge_ids:
                if eid in self.edge_cache:
                    edge_models_states.append(self.edge_cache[eid]["edge_model"])

            if not edge_models_states:
                continue

            avg_edge_model = average_state_dicts(edge_models_states)
            size_edge_MB = get_model_size_MB(avg_edge_model)
            self.comm_tracker["edge_model_upload_MB"] += len(edge_ids) * size_edge_MB
            self.comm_tracker["edge_model_download_MB"] += len(edge_ids) * size_edge_MB

            for eid in edge_ids:
                edge_layer[eid].load_state_dict(avg_edge_model)

    def print_comm_report(self):
        """Print communication cost report and compute total."""
        total = 0.0
        for k, v in self.comm_tracker.items():
            if k != "total_comm_MB":
                total += v
        self.comm_tracker["total_comm_MB"] = total

        print("\n=== Communication Report ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

    # =========================================================================
    # Optimizer Initialization
    # =========================================================================

    def initialize_optimizers(self):
        """Initialize optimizers and GradScalers for all models."""
        self.optimizers = {}
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {}
            for nid, model in nodes.items():
                if layer == 1:
                    self.optimizers[layer][nid] = {
                        'optimizer': LARS(
                            model.parameters(),
                            lr=0.3 * self.config["local_bs"] / 256,
                            weight_decay=self.config["weight_decay"],
                        ),
                        'scaler': torch.amp.GradScaler(
                            enabled=(self.device.type == 'cuda')
                        ),
                    }
                else:
                    self.optimizers[layer][nid] = {
                        'optimizer': torch.optim.Adam(
                            model.parameters(),
                            lr=self.config["lr"],
                            weight_decay=self.config["weight_decay"],
                        ),
                        'scaler': torch.amp.GradScaler(
                            enabled=(self.device.type == 'cuda')
                        ),
                    }

    # =========================================================================
    # Phase 1: Client SSL + Prototype Extraction
    # =========================================================================

    def _client_ssl_extraction_phase(self, cid, loader, ssl_epochs):
        """Run SSL training and extract prototypes for a single client."""
        model = self.structure[-1][cid].to(self.device)
        opt_dict = self.optimizers[-1][cid]
        optimizer, scaler = opt_dict['optimizer'], opt_dict['scaler']

        # Supervised Contrastive Training (Paper Eq. 1, Alg. 1)
        model.train()
        for _ in range(ssl_epochs):
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.type):
                    v1 = self.ssl_transforms(data)
                    v2 = self.ssl_transforms(data)
                    z1 = model(v1)
                    z2 = model(v2)
                    # Pool 4D feature maps to 2D vectors for SupCon
                    z1_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), start_dim=1)
                    z2_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), start_dim=1)
                    # SupCon: positives = same class (Paper Eq. 1)
                    all_features = torch.cat([z1_flat, z2_flat], dim=0)
                    all_labels = torch.cat([target, target], dim=0)
                    loss = supervised_contrastive_loss(all_features, all_labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Prototype Extraction
        model.eval()
        all_protos, all_stds = {}, {}
        feats_by_cls = {}

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                out = model(data)
                target_cpu = target.numpy()
                for i, cls_id in enumerate(target_cpu):
                    if cls_id not in feats_by_cls:
                        feats_by_cls[cls_id] = []
                    feats_by_cls[cls_id].append(out[i])

        for cls_id, tensors in feats_by_cls.items():
            stacked = torch.stack(tensors)
            all_protos[cls_id] = stacked.mean(0).cpu()
            all_stds[cls_id] = stacked.std(0, unbiased=False).cpu()

        del feats_by_cls
        gc.collect()
        return all_protos, all_stds

    # =========================================================================
    # Phase 2: Edge SSL + Prototype Extraction
    # =========================================================================

    def _edge_ssl_extraction_phase(self, eid, cids, client_outputs,
                                   ssl_epochs, syn_samples_per_class):
        """Run SSL training and extract prototypes for a single edge server."""
        model = self.structure[0][eid]
        opt_dict = self.optimizers[0][eid]
        optimizer = opt_dict['optimizer']
        scaler = opt_dict['scaler']

        # Collect and generate L1 synthetic data
        edge_specific_outputs = {
            cid: client_outputs[cid]
            for cid in cids if cid in client_outputs
        }
        syn_features_L1, syn_labels_L1 = aggregate_prototypes_and_generate_data(
            input_outputs=edge_specific_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device,
        )

        if syn_features_L1.shape[0] == 0:
            print(f"Edge {eid}: No prototypes from clients, skipping.")
            return None

        print(f"Edge {eid}: Generated {syn_features_L1.shape[0]} L1 samples. Starting SSL...")

        # SSL Training
        model.to(self.device).train()
        syn_dataset_L1 = torch.utils.data.TensorDataset(
            syn_features_L1.cpu(), syn_labels_L1.cpu()
        )
        del syn_features_L1, syn_labels_L1

        syn_loader_L1 = DataLoader(
            syn_dataset_L1,
            batch_size=self.config['local_bs'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        for epoch in range(ssl_epochs):
            total_loss = 0
            for features, labels in syn_loader_L1:
                features = features.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast(
                    device_type='cuda',
                    enabled=(self.device.type == 'cuda')
                ):
                    view_1 = self.ssl_transforms_edge(features)
                    view_2 = self.ssl_transforms_edge(features)
                    z1 = model(view_1)
                    z2 = model(view_2)
                    # Pool 4D feature maps to 2D vectors for SupCon
                    z1_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), start_dim=1)
                    z2_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), start_dim=1)
                    # SupCon with synthetic labels (Paper Eq. 1)
                    all_features = torch.cat([z1_flat, z2_flat], dim=0)
                    all_labels = torch.cat([labels, labels], dim=0)
                    loss = supervised_contrastive_loss(all_features, all_labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

        print(f"Edge {eid}: SSL done, final loss: {total_loss / len(syn_loader_L1):.4f}")

        # Prototype Extraction for L2
        model.to(self.device).eval()
        edge_feats_L2 = []

        with torch.no_grad(), torch.amp.autocast(
            device_type='cuda', enabled=(self.device.type == 'cuda')
        ):
            syn_loader_L1_eval = DataLoader(
                syn_dataset_L1,
                batch_size=self.config['local_bs'],
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            for features, _ in syn_loader_L1_eval:
                features = features.to(self.device)
                out_4d = model(features)
                out_2d = torch.flatten(
                    nn.AdaptiveAvgPool2d((1, 1))(out_4d), start_dim=1
                )
                if self.input_shape_cloud is None:
                    self.input_shape_cloud = out_2d.shape[1:]
                edge_feats_L2.append(out_2d)

        edge_features_L2 = torch.cat(edge_feats_L2, dim=0)
        syn_labels_L1_cpu = syn_dataset_L1.tensors[1]

        edge_protos_dists = calculate_prototypes_and_distribution(
            edge_features_L2, syn_labels_L1_cpu
        )

        del syn_loader_L1, syn_loader_L1_eval, edge_features_L2
        del syn_labels_L1_cpu, syn_dataset_L1
        return edge_protos_dists

    # =========================================================================
    # Phase 3: Cloud Supervised Training
    # =========================================================================

    def _cloud_supervised_phase(self, cloud_id, edge_outputs,
                                syn_epochs, syn_samples_per_class):
        """Run supervised training for the cloud model."""
        model = self.structure[len(self.config["mid_server"])][cloud_id]
        opt_dict = self.optimizers[len(self.config["mid_server"])][cloud_id]
        optimizer = opt_dict['optimizer']
        scaler = opt_dict['scaler']

        # Generate L2 synthetic data from edge prototypes
        syn_features_L2, syn_labels_L2 = aggregate_prototypes_and_generate_data(
            input_outputs=edge_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device,
        )

        if syn_features_L2.shape[0] == 0:
            print(f"Cloud {cloud_id}: No prototypes from Edge, skipping.")
            return 0.0

        print(
            f"Cloud {cloud_id}: Generated {syn_features_L2.shape[0]} L2 samples. "
            f"Starting supervised training..."
        )

        # Supervised Training (CrossEntropy)
        model.to(self.device).train()
        syn_dataset_L2 = torch.utils.data.TensorDataset(
            syn_features_L2.cpu(), syn_labels_L2.cpu()
        )
        del syn_features_L2, syn_labels_L2

        syn_loader_L2 = DataLoader(
            syn_dataset_L2,
            batch_size=self.config['local_bs'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )

        total_loss = 0
        for epoch in range(syn_epochs):
            epoch_loss = 0
            for features, labels in syn_loader_L2:
                features = features.to(self.device)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type='cuda',
                    enabled=(self.device.type == 'cuda')
                ):
                    logits = model(features)
                    loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            total_loss = epoch_loss / len(syn_loader_L2)

        print(f"Cloud {cloud_id}: Supervised done, final loss: {total_loss:.4f}")
        del syn_dataset_L2, syn_loader_L2
        return total_loss

    # =========================================================================
    # Phase 5: Validation
    # =========================================================================

    def _run_validation(self, valid_dataset, best_f1, epoch):
        """Evaluate the full pipeline on the validation set."""
        client_model = self.structure[-1][0]
        eid = self.connectivity[-1][0]
        edge_model = self.structure[0][eid]
        cloud_model = self.structure[len(self.config["mid_server"])][0]

        client_model.to(self.device).eval()
        edge_model.to(self.device).eval()
        cloud_model.to(self.device).eval()

        loader = DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        all_preds_tensors, all_targets_tensors = [], []
        nan_detected = False
        inf_detected = False

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                out = FullPipelineModel(client_model, edge_model, cloud_model)(data)

                if not nan_detected and torch.isnan(out).any():
                    print("!!! WARNING: NaN detected in model output during validation!")
                    nan_detected = True
                if not inf_detected and torch.isinf(out).any():
                    print("!!! WARNING: Inf detected in model output during validation!")
                    inf_detected = True

                pred = out.argmax(dim=1)
                all_preds_tensors.append(pred.cpu())
                all_targets_tensors.append(target.cpu())

        all_preds = torch.cat(all_preds_tensors).numpy()
        all_targets = torch.cat(all_targets_tensors).numpy()

        # Debug info
        print("\n--- Validation Debug Info ---")
        print(f"Targets - Unique: {np.unique(all_targets)[:20]}, Shape: {all_targets.shape}")
        unique_preds = np.unique(all_preds)
        print(f"Predictions - Unique (first 20): {unique_preds[:20]}, Shape: {all_preds.shape}")
        print("--- End Validation Debug Info ---")

        f1 = 0.0
        pipeline_model = None
        try:
            f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

            if best_f1 < f1:
                print(f"Saving best model at epoch {epoch} with F1: {f1 * 100:.2f}%")
                best_f1 = f1
                pipeline_model = FullPipelineModel(
                    client_model=copy.deepcopy(client_model),
                    edge_model=copy.deepcopy(edge_model),
                    cloud_model=copy.deepcopy(cloud_model),
                )
        except ValueError as e:
            print(f"!!! ERROR calculating F1 score: {e}")

        return f1, best_f1, pipeline_model

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def train(self, train_dataset, valid_dataset, test_dataset,
              user_groups, epochs, checkpoint_path="checkpoint_hfl.pt"):
        """
        Run the full H-SFP training pipeline.

        Returns:
            Dict with validation_f1, cloud_loss, best_f1, and comm_report.
        """
        self.initialize_optimizers()

        num_users = self.config["num_users"]
        frac = self.config["frac"]
        local_bs = self.config["local_bs"]
        t1, t2 = int(self.config["t1"]), int(self.config["t2"])

        validation_f1_list, cloud_loss_list = [], []
        best_f1 = 0
        start_epoch = 1

        dl_kwargs = {
            "batch_size": local_bs,
            "num_workers": self.num_workers,
            "pin_memory": True if torch.cuda.is_available() else False,
            "persistent_workers": True if self.num_workers > 0 else False,
            "drop_last": True,
        }

        for epoch in range(start_epoch, epochs + 1):
            print(f"\n{'=' * 20} EPOCH {epoch}/{epochs} {'=' * 20}")
            epoch_start_time = time.time()

            # 1. Select clients
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)
            client_outputs = {}

            # Phase 1: Client Processing
            print(f"-> Phase 1: Clients Processing ({len(idxs_users)} nodes)...")
            for cid in idxs_users:
                self.client_cache.append(cid)
                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(local_data, shuffle=True, **dl_kwargs)

                client_outputs[cid] = self._client_ssl_extraction_phase(
                    cid, loader,
                    ssl_epochs=self.config.get("ssl_epochs_client", 10),
                )
                cost = get_proto_dist_size_MB(client_outputs[cid])
                self.comm_tracker["client_to_edge_data_MB"] += cost

            torch.cuda.empty_cache()
            gc.collect()

            # Phase 2: Edge Processing
            print(f"-> Phase 2: Edge Processing...")
            edge_to_clients = {}
            for cid in idxs_users:
                eid = self.connectivity[-1][cid]
                edge_to_clients.setdefault(eid, []).append(cid)

            edge_outputs = {}
            for eid, cids in edge_to_clients.items():
                edge_outputs[eid] = self._edge_ssl_extraction_phase(
                    eid, cids, client_outputs,
                    ssl_epochs=self.config.get("ssl_epochs_edge", 10),
                    syn_samples_per_class=self.config.get("syn_samples_per_class", 50),
                )
                cost = get_proto_dist_size_MB(edge_outputs[eid])
                self.comm_tracker["edge_to_cloud_data_MB"] += cost

            del client_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # Phase 3: Cloud Processing
            print(f"-> Phase 3: Cloud Supervised Training...")
            cloud_loss = self._cloud_supervised_phase(
                0, edge_outputs,
                syn_epochs=self.config.get("syn_epochs_cloud", 10),
                syn_samples_per_class=self.config.get("syn_samples_per_class", 50),
            )
            cloud_loss_list.append(cloud_loss)

            del edge_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # Phase 4: Aggregation & Validation
            if epoch % t1 == 0:
                self.edge_server_aggregation()

            if epoch % t2 == 0:
                self.cloud_aggregation()

                f1, current_best_f1, model_snapshot = self._run_validation(
                    valid_dataset, best_f1, epoch
                )
                validation_f1_list.append(f1)

                if model_snapshot is not None:
                    best_f1 = current_best_f1
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_snapshot.state_dict(),
                        'best_f1': best_f1,
                        'config': self.config,
                    }, checkpoint_path)
                    print(
                        f"*** Checkpoint saved: {checkpoint_path} "
                        f"(F1: {best_f1 * 100:.2f}%)"
                    )

            print(f"Epoch {epoch} completed in {time.time() - epoch_start_time:.2f}s")

        # Final report
        print("\n" + "=" * 50)
        print("TRAINING FINISHED. Loading best model for testing...")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded best model from epoch {checkpoint['epoch']}")

        self.print_comm_report()

        return {
            "validation_f1": validation_f1_list,
            "cloud_loss": cloud_loss_list,
            "best_f1": best_f1,
            "comm_report": self.comm_tracker,
        }
