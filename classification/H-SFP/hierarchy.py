import copy
import logging
import math
import os
import time
from collections import deque
import gc
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import kornia.augmentation as K
from sklearn.metrics import accuracy_score

try:
    import wandb
except ImportError:
    wandb = None

# E-HSFP extensions
from ehsfp import (
    get_ehsfp_config,
    EpisodicPrototypeMemory,
    mix_current_and_memory,
    PrototypeReliabilityNetwork,
    reliability_weighted_aggregate,
    train_reliability_bootstrap,
    prototype_replay_consistency_loss,
    dropout_consistency_loss,
    PrototypeDropout,
    ServerlessMetricsTracker,
    ResidualPrototypeGenerator,
    EHSFPMetricsLogger,
)

# camera-ready: optional fine-grained phase profiling (gated by config["profile"])
try:
    from camera_ready.profiling import ProfileLog
except Exception:  # pragma: no cover - keeps legacy contexts working
    ProfileLog = None
# =============================================================================
# SECTION 1: UTILITY FUNCTIONS
# =============================================================================

@torch.no_grad()
def generate_synthetic_data(prototypes, distributions_std, num_samples_per_class):
    """
    Memory-efficient: generate data directly on the prototypes' device.
    """
    num_classes = prototypes.shape[0]
    shape = prototypes.shape[1:]
    device = prototypes.device
    
    # Use broadcasting to avoid materializing an overly large epsilon tensor
    # Shape: [C, N, ...]
    means = prototypes.unsqueeze(1) 
    stds = distributions_std.unsqueeze(1)
    
    # Create directly on GPU with a matching dtype to save VRAM
    epsilon = torch.randn(num_classes, num_samples_per_class, *shape, 
                          device=device, dtype=prototypes.dtype)
    
    # In-place operation to save memory
    epsilon.mul_(stds).add_(means)
    
    return epsilon.flatten(0, 1) # [C*N, ...]

def _aggregate_prototypes_and_generate_data(input_outputs, num_samples_per_class, device):
    """
    Group and synthesize data faster by reducing Python loops.
    """
    if not input_outputs:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    # Group by label using a dictionary (faster)
    merged = {}
    for sid, outputs in input_outputs.items():
        if outputs is None: continue
        proto_dict, dist_dict = outputs
        for label, proto in proto_dict.items():
            if label not in merged:
                merged[label] = {'p': [], 'd': []}
            merged[label]['p'].append(proto)
            merged[label]['d'].append(dist_dict[label])

    if not merged:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    final_labels = sorted(merged.keys())
    final_protos = torch.stack([torch.stack(merged[l]['p']).mean(0) for l in final_labels]).to(device)
    # Paper Eq. 5: average variance (σ²), then sqrt to get σ
    final_dists = torch.stack([
        torch.sqrt(torch.stack([d**2 for d in merged[l]['d']]).mean(0))
        for l in final_labels
    ]).to(device)

    # Free the dictionary immediately
    del merged
    
    features = generate_synthetic_data(final_protos, final_dists, num_samples_per_class)
    labels = torch.tensor(final_labels, device=device).repeat_interleave(num_samples_per_class)
    
    return features, labels

    
def calculate_prototypes_and_distribution(fx: torch.Tensor, fy: torch.Tensor):
    """
    Compute per-class prototype and distribution.
    (fx (features) on GPU, fy (labels) on CPU).
    """
    unique_classes = torch.unique(fy) 
    prototypes = {}
    distributions_std = {}

    for cls in unique_classes:
        cls_label = cls.item() 
        mask = (fy == cls).to(fx.device)
        class_features = fx[mask]
        
        if class_features.shape[0] > 0:
            # Avoid FutureWarning
            with torch.amp.autocast(device_type='cuda', enabled=(fx.device.type == 'cuda')):
                prototypes[cls_label] = torch.mean(class_features, dim=0)
                distributions_std[cls_label] = torch.std(class_features, dim=0, unbiased=False)
        else:
            print(f"Warning: class {cls_label} has no samples in the processed data.")

    return prototypes, distributions_std

# --- SSL (Self-Supervised Learning) helpers ---

def build_client_ssl_transforms(input_size):
    """Build client SSL augmentations adaptive to input image size."""
    return nn.Sequential(
        K.RandomResizedCrop(size=(input_size, input_size), scale=(0.5, 1.0)),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        K.RandomGrayscale(p=0.2)
    )

def build_edge_ssl_transforms(spatial_size):
    """Build edge SSL augmentations adaptive to feature map spatial dims."""
    return nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomResizedCrop(size=(spatial_size, spatial_size), scale=(0.8, 1.0)),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5)
    )

def info_nce_loss_4d(z1, z2, temperature=0.5):
    """Compute InfoNCE loss for 4D output (feature map)."""
    z1 = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(z1), start_dim=1)
    z2 = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(z2), start_dim=1)
    
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    sim_matrix = torch.matmul(z1, z2.mT) / temperature # batched transpose via .mT
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.mT, labels) # batched transpose via .mT
    
    return (loss_a + loss_b) / 2

def info_nce_loss_2d(z1, z2, temperature=0.5):
    """Compute InfoNCE loss for 2D input (feature vector)."""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    sim_matrix = torch.matmul(z1, z2.mT) / temperature # batched transpose via .mT
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.mT, labels) # batched transpose via .mT
    
    return (loss_a + loss_b) / 2

def supervised_contrastive_loss(features, labels, temperature=0.5):
    """
    Supervised Contrastive Loss (SupCon) — Paper Eq. 1.
    Positive pairs P(i) are samples sharing the same class label.
    """
    features = nn.functional.normalize(features, dim=1)
    batch_size = features.shape[0]
    device = features.device

    sim_matrix = torch.matmul(features, features.T) / temperature

    labels_col = labels.view(-1, 1)
    positive_mask = torch.eq(labels_col, labels_col.T).float()
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - self_mask

    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    exp_logits = torch.exp(logits) * (1 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    num_positives = positive_mask.sum(dim=1)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    valid = (num_positives > 0).float()
    loss = -(valid * mean_log_prob).sum() / (valid.sum() + 1e-8)

    return loss

# --- Class utility functions (HFL utils) ---

def average_state_dicts(state_dicts):
    if not state_dicts: return {}
    avg_dict = {}
    device = next(iter(state_dicts[0].values())).device
    for key in state_dicts[0].keys():
        tensors = [d[key].to(device) for d in state_dicts]
        avg_dict[key] = sum(tensors) / len(tensors)
    return avg_dict

def get_model_size_MB(state_dict):
    """Compute model size (MB) from a state_dict."""
    return (sum(param.numel() for param in state_dict.values()) * 4 / 1e6)

def get_proto_dist_size_MB(proto_dist_tuple: tuple) -> float:
    """Compute the size (MB) of a (proto_dict, dist_dict) tuple."""
    if proto_dist_tuple is None:
        return 0.0
        
    proto_dict, dist_dict = proto_dist_tuple
    total_bytes = 0
    
    # Size of all tensors in the prototypes dict
    for tensor in proto_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
        
    # Size of all tensors in the distributions dict
    for tensor in dist_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
        
    return total_bytes / (1024**2)

# =============================================================================
# SECTION 2: DATA AND MODEL DEFINITIONS
# =============================================================================

class FullPipelineModel(nn.Module):
    def __init__(self, client_model, edge_model, cloud_model):
        super().__init__()
        self.client = client_model
        self.edge = edge_model
        self.cloud = cloud_model

    def forward(self, x):
        with torch.amp.autocast(device_type='cuda', enabled=(x.device.type == 'cuda')):
            x = self.client(x)
            x = self.edge(x)
            # Apply GAP + flatten to match the prototype extraction pipeline
            # (edge extraction does AdaptiveAvgPool2d + flatten before cloud)
            x = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(x), start_dim=1)
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


# =============================================================================
# SECTION 3: MAIN HIERARCHICAL FL CLASS
# =============================================================================

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
        # Build client SSL transforms adaptive to input image size
        input_size = args.get("input_size", 32)
        self.ssl_transforms = build_client_ssl_transforms(input_size).to(self.device)
        # Edge SSL transforms built lazily when we first see feature map dims
        self.ssl_transforms_edge = None
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        # SupCon temperature. Khosla et al. (2020) report the optimum near 0.1;
        # a lower temperature yields tighter, better-separated class clusters ->
        # more separable per-class prototypes for every downstream tier.
        # Config-gated so it is applied identically to H-SFP and E-HSFP (fair).
        self.supcon_temp = float(args.get("supcon_temperature", 0.1))
        
        # --- Communication tracker ---
        self.comm_tracker = {
            "client_to_edge_data_MB": 0.0,  # cost of sending (proto, dist)
            "edge_to_cloud_data_MB": 0.0,   # cost of sending (proto, dist)
            "client_model_upload_MB": 0.0,  # FedAvg model cost
            "client_model_download_MB": 0.0,
            "edge_model_upload_MB": 0.0,
            "edge_model_download_MB": 0.0,
            "total_comm_MB": 0.0           # total cost
        }
        # --- end ---
        
        self.client_cache = deque(maxlen=20)
        self.optimizers = {}
        
        self.input_shape_client = None
        self.input_shape_edge = None
        self.input_shape_cloud = None

        self.num_workers = 2 if os.name != 'nt' else 0
        print(f"Using {self.num_workers} workers for the DataLoader.")

        # --- E-HSFP components ---
        self.ecfg = get_ehsfp_config(args)
        self.ehsfp_logger = EHSFPMetricsLogger()

        # Episodic memory (client-level and edge-level)
        if self.ecfg["use_episodic_memory"]:
            self.client_memory = EpisodicPrototypeMemory(
                max_size=self.ecfg["memory_size"],
                max_age=self.ecfg["max_prototype_age"],
            )
            self.edge_memory = EpisodicPrototypeMemory(
                max_size=self.ecfg["memory_size"],
                max_age=self.ecfg["max_prototype_age"],
            )
        else:
            self.client_memory = None
            self.edge_memory = None

        # Reliability network
        if self.ecfg["aggregation_mode"] == "learnable_reliability":
            self.reliability_net = PrototypeReliabilityNetwork(
                hidden_dim=self.ecfg["reliability_hidden_dim"],
            ).to(self.device)
            self.reliability_optimizer = torch.optim.Adam(
                self.reliability_net.parameters(),
                lr=self.ecfg["reliability_lr"],
                weight_decay=self.ecfg["reliability_weight_decay"],
            )
        else:
            self.reliability_net = None
            self.reliability_optimizer = None

        # Prototype dropout
        if self.ecfg["use_prototype_dropout"]:
            self.proto_dropout = PrototypeDropout(
                rate=self.ecfg["prototype_dropout_rate"],
                mode=self.ecfg["dropout_mode"],
            )
        else:
            self.proto_dropout = None

        # Serverless simulator
        if self.ecfg["use_serverless_simulation"]:
            self.serverless_tracker = ServerlessMetricsTracker(self.ecfg)
        else:
            self.serverless_tracker = None

        # Residual generator (disabled by default)
        self.residual_generator = None

        # camera-ready: fine-grained phase profiler (None unless config["profile"])
        self.profiler = ProfileLog() if (args.get("profile") and ProfileLog) else None


    def _prof_now(self):
        """CUDA-synchronized perf_counter timestamp (only meaningful when profiling)."""
        if self.profiler is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _prof_add(self, name, t0, round_idx=None):
        if self.profiler is not None:
            self.profiler.add(name, self._prof_now() - t0, round_idx)

    def _build_hierarchy(self):
        # ... (original code - unchanged) ...
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
        # ... (original code - unchanged) ...
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

    # --- Aggregation functions - Phase 4 ---
    def edge_server_aggregation(self):
        # ... (original code - unchanged) ...
        print("Edge Aggregation Started")
        client_layer = self.structure[-1]
        edge_layer = self.structure[0]
        client_to_edge = self.connectivity[-1]

        edge_to_clients = {}
        for cid in self.client_cache:
            eid = client_to_edge[cid]
            edge_to_clients.setdefault(eid, []).append(cid)

        self.edge_cache = {}  # Reset edge cache

        for eid, cids in edge_to_clients.items():
            client_models_states = [client_layer[cid].state_dict() for cid in cids]
            if not client_models_states: continue
            
            avg_client_model = average_state_dicts(client_models_states)
            self.edge_cache[eid] = {"edge_model": edge_layer[eid].state_dict()}

            size_MB = get_model_size_MB(avg_client_model)
            self.comm_tracker["client_model_upload_MB"] += len(cids) * size_MB

            for cid in cids:
                client_layer[cid].load_state_dict(avg_client_model)
            self.comm_tracker["client_model_download_MB"] += len(cids) * size_MB

    def cloud_aggregation(self):
        # ... (original code - unchanged) ...
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

            if not edge_models_states: continue
                
            avg_edge_model = average_state_dicts(edge_models_states)
            size_edge_MB = get_model_size_MB(avg_edge_model)
            self.comm_tracker["edge_model_upload_MB"] += len(edge_ids) * size_edge_MB
            self.comm_tracker["edge_model_download_MB"] += len(edge_ids) * size_edge_MB

            for eid in edge_ids:
                edge_layer[eid].load_state_dict(avg_edge_model)

    def print_comm_report(self):
        """Print the communication-cost report and total."""
        total = 0.0
        for k, v in self.comm_tracker.items():
            if k != "total_comm_MB":
                total += v
        self.comm_tracker["total_comm_MB"] = total
        
        print("\n=== Communication Report ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

    def _build_lr_scheduler(self, optimizer):
        """Linear warmup -> cosine annealing schedule (peak LR unchanged, so the
        comparison vs baselines stays fair). Speeds convergence in few epochs.
        Controlled by config: use_lr_schedule (bool), warmup_epochs (int).
        Returns None when disabled."""
        if not self.args.get("use_lr_schedule", False):
            return None
        total = max(int(self._sched_total_epochs), 1)
        warmup = max(int(self.args.get("warmup_epochs", 0)), 0)

        def lr_lambda(epoch):  # epoch is 0-indexed scheduler step
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            # cosine from 1.0 -> ~0 over the remaining epochs
            progress = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def initialize_optimizers(self):
        """Initialize optimizers, GradScalers and (optional) LR schedulers.

        SSL tiers (client/edge) may use a separate, contrastive-appropriate
        learning rate `ssl_lr` (defaults to `lr` -> no change); the supervised
        cloud classifier keeps `lr` so it stays directly comparable to baselines.
        """
        self.optimizers = {}
        cloud_layer = len(self.args["mid_server"])
        ssl_lr = self.args.get("ssl_lr", self.args["lr"])
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {}
            lr_use = self.args["lr"] if layer == cloud_layer else ssl_lr
            for nid, model in nodes.items():
                opt = torch.optim.Adam(
                    model.parameters(),
                    lr=lr_use,
                    weight_decay=self.args["weight_decay"],
                )
                self.optimizers[layer][nid] = {
                    'optimizer': opt,
                    'scaler': torch.amp.GradScaler(enabled=(self.device.type == 'cuda')),
                    'scheduler': self._build_lr_scheduler(opt),
                }

    def _step_lr_schedulers(self):
        """Advance all per-node LR schedulers by one epoch (if enabled)."""
        for layer in self.optimizers.values():
            for od in layer.values():
                if od.get('scheduler') is not None:
                    od['scheduler'].step()

    # --- Helper methods for the training loop (optimized) ---

    def _client_ssl_extraction_phase(self, cid, loader, ssl_epochs):
        model = self.structure[-1][cid].to(self.device)
        opt_dict = self.optimizers[-1][cid]
        optimizer, scaler = opt_dict['optimizer'], opt_dict['scaler']
        
        # 1. Supervised Contrastive Training (Paper Eq. 1, Alg. 1)
        model.train()
        for _ in range(ssl_epochs):
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type=self.device.type):
                    v1, v2 = self.ssl_transforms(data), self.ssl_transforms(data)
                    z1 = model(v1)
                    z2 = model(v2)
                    # Pool 4D feature maps to 2D vectors for SupCon
                    z1_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), start_dim=1)
                    z2_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), start_dim=1)
                    # SupCon: positives = same class (Paper Eq. 1)
                    all_features = torch.cat([z1_flat, z2_flat], dim=0)
                    all_labels = torch.cat([target, target], dim=0)
                    loss = supervised_contrastive_loss(all_features, all_labels, temperature=self.supcon_temp)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # 2. Extraction - avoid storing all features in a list when unnecessary
        model.eval()
        all_protos, all_stds = {}, {}
        
        # Optimization: accumulate per batch to avoid huge tensors
        # For simplicity and correct std, use class-wise grouping
        feats_by_cls = {}
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                out = model(data)
                
                # Move target to CPU once
                target_cpu = target.numpy()
                for i, cls_id in enumerate(target_cpu):
                    if cls_id not in feats_by_cls: feats_by_cls[cls_id] = []
                    feats_by_cls[cls_id].append(out[i])

        # Compute mean/std per class (and the per-class support count, which
        # feeds the E-HSFP reliability network — previously left at 0).
        support_counts = {}
        for cls_id, tensors in feats_by_cls.items():
            stacked = torch.stack(tensors)
            all_protos[cls_id] = stacked.mean(0).cpu() # Move to CPU to save VRAM
            all_stds[cls_id] = stacked.std(0, unbiased=False).cpu()
            support_counts[int(cls_id)] = len(tensors)

        # Manual cleanup
        del feats_by_cls
        gc.collect()
        return (all_protos, all_stds), support_counts

    def _edge_ssl_extraction_phase(self, eid, cids, client_outputs, ssl_epochs, syn_samples_per_class):
        """(Phase 2) Run SSL and extract (proto, dist) for one edge."""
        model = self.structure[0][eid]
        opt_dict = self.optimizers[0][eid]
        optimizer = opt_dict['optimizer']
        scaler = opt_dict['scaler']

        # --- 2a. Collect and synthesize L1 data ---
        edge_specific_client_outputs = {cid: client_outputs[cid] for cid in cids if cid in client_outputs}

        # E-HSFP: Apply prototype dropout at client level
        if self.proto_dropout is not None and self.ecfg["dropout_mode"] in ("client_prototype", "both"):
            edge_specific_client_outputs = self.proto_dropout.apply_to_source_outputs(edge_specific_client_outputs)

        # E-HSFP: Use reliability-weighted aggregation when enabled
        syn_features_L1, syn_labels_L1 = reliability_weighted_aggregate(
            input_outputs=edge_specific_client_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device,
            reliability_net=self.reliability_net,
            memory=self.client_memory,
            generator=self.residual_generator,
            support_map={cid: getattr(self, "_client_support", {}).get(cid, {}) for cid in cids},
        )

        if syn_features_L1.shape[0] == 0:
             print(f"Edge {eid}: no prototypes from clients, skipping.")
             return None, {}

        print(f"Edge {eid}: generated {syn_features_L1.shape[0]} L1 samples (original labels). Starting SSL...")

        # --- 2b. SSL training ---
        model.to(self.device).train()
        
        # Move data to CPU
        syn_dataset_L1 = torch.utils.data.TensorDataset(
            syn_features_L1.cpu(), 
            syn_labels_L1.cpu() # include labels
        )
        del syn_features_L1, syn_labels_L1
        
        syn_loader_L1 = DataLoader(
            syn_dataset_L1, 
            batch_size=self.args['local_bs'], 
            shuffle=True,
            # --- fix AttributeError ---
            num_workers=0, # data already in RAM
            pin_memory=False,
            # --- end ---
            drop_last=True 
        )
        
        for epoch in range(ssl_epochs):
            total_loss = 0
            # Supervised Contrastive Training (Paper Alg. 2, Lines 9-11)
            for features, labels in syn_loader_L1:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Build edge SSL transforms lazily from actual feature map spatial dims
                if self.ssl_transforms_edge is None:
                    spatial_size = features.shape[-1]  # H dimension of feature map
                    self.ssl_transforms_edge = build_edge_ssl_transforms(spatial_size).to(self.device)

                with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
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
                    loss = supervised_contrastive_loss(all_features, all_labels, temperature=self.supcon_temp)

                    # E-HSFP: PRC loss at edge level
                    if self.ecfg["use_prc_loss"] and self.edge_memory is not None and len(self.edge_memory) > 0:
                        mem_p, mem_d = self.edge_memory.to_proto_dist_dicts()
                        cur_p = {cid: client_outputs[cid][0] for cid in cids if cid in client_outputs and client_outputs[cid] is not None}
                        cur_d = {cid: client_outputs[cid][1] for cid in cids if cid in client_outputs and client_outputs[cid] is not None}
                        if cur_p and mem_p:
                            def edge_repr_fn(x):
                                return torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(model(x)), start_dim=1)
                            prc = prototype_replay_consistency_loss(
                                cur_p, cur_d, mem_p, mem_d,
                                edge_repr_fn, self.ecfg["prc_num_samples"], self.device,
                            )
                            loss = loss + self.ecfg["lambda_prc"] * prc

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
        print(f"Edge {eid}: SSL done, final loss: {total_loss/len(syn_loader_L1):.4f}")

        # --- 2c. Extract proto/dist ---
        model.to(self.device).eval()
        edge_feats_L2 = []
        
        # Avoid FutureWarning
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
            syn_loader_L1_eval = DataLoader(
                syn_dataset_L1, # reuse dataset (features + labels)
                batch_size=self.args['local_bs'], 
                shuffle=False,
                # --- fix AttributeError ---
                num_workers=0,
                pin_memory=False
                # (no drop_last=True)
                # --- end ---
            )
            # DataLoader returns (features, labels)
            for features, _ in syn_loader_L1_eval: # take features only
                features = features.to(self.device)
                out_4d = model(features)
                
                # Flatten 4D -> 2D for the cloud
                out_2d = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(out_4d), start_dim=1)
                
                if self.input_shape_cloud is None:
                    self.input_shape_cloud = out_2d.shape[1:]
                
                edge_feats_L2.append(out_2d)
        
        edge_features_L2 = torch.cat(edge_feats_L2, dim=0) # (on GPU, 2D)
        
        # Get original labels (from CPU dataset)
        syn_labels_L1_cpu = syn_dataset_L1.tensors[1] 
        
        # Return dict {original_label: tensor_2D}
        edge_protos_dists = calculate_prototypes_and_distribution(edge_features_L2, syn_labels_L1_cpu)

        # Per-class support counts for the reliability network.
        uniq, cnts = torch.unique(syn_labels_L1_cpu, return_counts=True)
        support_counts = {int(c): int(n) for c, n in zip(uniq.tolist(), cnts.tolist())}

        del syn_loader_L1, syn_loader_L1_eval, edge_features_L2, syn_labels_L1_cpu, syn_dataset_L1
        return edge_protos_dists, support_counts

    def _cloud_supervised_phase(self, cloud_id, edge_outputs, syn_epochs, syn_samples_per_class):
        """(Phase 3) Run supervised training for the cloud."""
        model = self.structure[len(self.args["mid_server"])][cloud_id]
        opt_dict = self.optimizers[len(self.args["mid_server"])][cloud_id]
        optimizer = opt_dict['optimizer']
        scaler = opt_dict['scaler']

        # --- 3a. Collect and synthesize L2 data ---
        # E-HSFP: Apply prototype dropout at edge level
        if self.proto_dropout is not None and self.ecfg["dropout_mode"] in ("edge_prototype", "both"):
            edge_outputs = self.proto_dropout.apply_to_source_outputs(edge_outputs)

        # E-HSFP: reliability-weighted aggregation
        syn_features_L2, syn_labels_L2 = reliability_weighted_aggregate(
            input_outputs=edge_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device,
            reliability_net=self.reliability_net,
            memory=self.edge_memory,
            generator=self.residual_generator,
            support_map=getattr(self, "_edge_support", {}),
        )
        
        if syn_features_L2.shape[0] == 0:
            print(f"Cloud {cloud_id}: no prototypes from edges, skipping.")
            return 0.0

        print(f"Cloud {cloud_id}: generated {syn_features_L2.shape[0]} L2 samples (original labels). Starting supervised...")

        # --- 3b. Supervised training (CrossEntropy) ---
        model.to(self.device).train()
        
        syn_dataset_L2 = torch.utils.data.TensorDataset(
            syn_features_L2.cpu(), 
            syn_labels_L2.cpu() # original labels (e.g. 0-99)
        )
        del syn_features_L2, syn_labels_L2
        
        syn_loader_L2 = DataLoader(
            syn_dataset_L2, 
            batch_size=self.args['local_bs'], 
            shuffle=True,
            # --- fix AttributeError ---
            num_workers=0,
            pin_memory=False,
            # --- end ---
            drop_last=True
        )
        
        total_loss = 0
        for epoch in range(syn_epochs):
            epoch_loss = 0
            for features, labels in syn_loader_L2:
                features, labels = features.to(self.device), labels.to(self.device, non_blocking=True)
                
                # Avoid FutureWarning
                with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                    logits = model(features)
                    loss = self.criterion(logits, labels)

                    # E-HSFP: PRC loss at cloud level
                    if self.ecfg["use_prc_loss"] and self.edge_memory is not None and len(self.edge_memory) > 0:
                        mem_p, mem_d = self.edge_memory.to_proto_dist_dicts()
                        cur_p = {eid: edge_outputs[eid][0] for eid in edge_outputs if edge_outputs[eid] is not None}
                        cur_d = {eid: edge_outputs[eid][1] for eid in edge_outputs if edge_outputs[eid] is not None}
                        if cur_p and mem_p:
                            prc = prototype_replay_consistency_loss(
                                cur_p, cur_d, mem_p, mem_d,
                                model, self.ecfg["prc_num_samples"], self.device,
                            )
                            loss = loss + self.ecfg["lambda_prc"] * prc

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

            total_loss = epoch_loss / len(syn_loader_L2)

        print(f"Cloud {cloud_id}: supervised done, final loss: {total_loss:.4f}")
        del syn_dataset_L2, syn_loader_L2
        return total_loss
        
    def _run_validation(self, valid_dataset, best_f1, epoch):
        """(Phase 5) Run evaluation (optimized)."""
        client_model = self.structure[-1][0]
        eid = self.connectivity[-1][0]
        edge_model = self.structure[0][eid]
        cloud_model = self.structure[len(self.args["mid_server"])][0]

        client_model.to(self.device).eval()
        edge_model.to(self.device).eval()
        cloud_model.to(self.device).eval()
        
        loader = DataLoader(
            valid_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        all_preds_tensors, all_targets_tensors = [], []
        
        nan_detected = False
        inf_detected = False

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                out_cl = FullPipelineModel(client_model, edge_model, cloud_model)(data)

                # --- DEBUG: Check for NaN/Inf in model output ---
                if not nan_detected and torch.isnan(out_cl).any():
                    print("!!! WARNING: NaN detected in model output during validation!")
                    nan_detected = True
                if not inf_detected and torch.isinf(out_cl).any():
                    print("!!! WARNING: Inf detected in model output during validation!")
                    inf_detected = True
                # --- END DEBUG ---

                pred = out_cl.argmax(dim=1)
                all_preds_tensors.append(pred.cpu())
                all_targets_tensors.append(target.cpu()) # <-- Should append TARGET LABELS
        
        # Concatenate tensors first
        all_preds = torch.cat(all_preds_tensors).numpy()
        all_targets = torch.cat(all_targets_tensors).numpy()
        
        # --- DEBUG: Check labels before f1_score ---
        print("\n--- Validation Debug Info ---")
        print(f"Targets - Type: {all_targets.dtype}, Shape: {all_targets.shape}")
        print(f"Targets - Unique values: {np.unique(all_targets)}")
        print(f"Targets - Min: {np.min(all_targets)}, Max: {np.max(all_targets)}")
        
        print(f"Predictions - Type: {all_preds.dtype}, Shape: {all_preds.shape}")
        # Show unique predicted values, limit if too many
        unique_preds = np.unique(all_preds)
        print(f"Predictions - Unique values (first 20): {unique_preds[:20]}")
        if len(unique_preds) > 20:
            print(f"  ... ({len(unique_preds)} total unique predictions)")
        print(f"Predictions - Min: {np.min(all_preds)}, Max: {np.max(all_preds)}")
        print("--- End Validation Debug Info ---")
        # --- END DEBUG ---

        f1 = 0.0 # Default value (holds accuracy; name kept for output-key compatibility)
        pipeline_model = None
        try:
            # Primary metric: top-1 accuracy
            f1 = accuracy_score(all_targets, all_preds)

            if best_f1 < f1:
                print(f"Saved best model at epoch {epoch} with Acc: {f1 * 100:.2f} %")
                best_f1 = f1
                pipeline_model = FullPipelineModel(
                    client_model=copy.deepcopy(client_model),
                    edge_model=copy.deepcopy(edge_model),
                    cloud_model=copy.deepcopy(cloud_model),
                )
        except ValueError as e:
            print(f"!!! ERROR calculating F1 score: {e}")
            print("!!! Check debug info above for potential issues (NaNs, label ranges, types).")
            # Keep best_f1 as it was, don't update pipeline_model
            
        return f1, best_f1, pipeline_model

    # --- Main training method (reorganized) --

    def train_end_to_end(
        self,
        train_dataset,
        valid_dataset,
        test_dataset,
        user_groups,
        config,
        epochs,
        checkpoint_path="checkpoint_hfl.pt"
    ):
        # Total epochs for the (optional) warmup+cosine LR schedule
        self._sched_total_epochs = epochs
        self.initialize_optimizers()

        # Read config
        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]
        t1, t2 = int(config["t1"]), int(config["t2"])
        
        # Initialize metric storage
        validation_f1_list, cloud_loss_list = [], []
        best_f1 = 0
        best_pipeline_model = None
        start_epoch = 1

        # --- DataLoader optimization ---
        # persistent_workers=True avoids re-creating the DataLoader each epoch
        # pin_memory=True speeds up host->GPU transfer
        dl_kwargs = {
            "batch_size": local_bs,
            "num_workers": self.num_workers,
            "pin_memory": True if torch.cuda.is_available() else False,
            "persistent_workers": True if self.num_workers > 0 else False,
            "drop_last": True
        }

        for epoch in range(start_epoch, epochs + 1):
            print(f"\n{'='*20} EPOCH {epoch}/{epochs} {'='*20}")
            epoch_start_time = time.time()
            self.ehsfp_logger.reset_epoch()

            # E-HSFP: Begin serverless episode
            if self.serverless_tracker is not None:
                self.serverless_tracker.begin_episode(epoch)

            # 1. SELECT CLIENTS
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)
            client_outputs = {}
            # Per-round support counts feeding the reliability network
            self._client_support = {}
            self._edge_support = {}

            # --- PHASE 1: CLIENT PROCESSING ---
            print(f"-> Phase 1: Clients Processing ({len(idxs_users)} nodes)...")
            _t_pack = self._prof_now()
            for cid in idxs_users:
                self.client_cache.append(cid)

                # E-HSFP: Simulate serverless invocation
                if self.serverless_tracker is not None:
                    inv = self.serverless_tracker.simulate_invocation("client", cid, epoch)
                    if inv.timed_out:
                        continue  # simulate function timeout

                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(local_data, shuffle=True, **dl_kwargs)

                # SSL & Extraction
                client_outputs[cid], support_counts = self._client_ssl_extraction_phase(
                    cid, loader, ssl_epochs=config.get("ssl_epochs_client", 10)
                )

                # Track data transmission
                cost = get_proto_dist_size_MB(client_outputs[cid])
                self.comm_tracker["client_to_edge_data_MB"] += cost
                self._client_support[cid] = support_counts

                # E-HSFP: Store client prototypes in memory
                if self.client_memory is not None and client_outputs[cid] is not None:
                    proto_dict, dist_dict = client_outputs[cid]
                    self.client_memory.add_from_proto_dicts(
                        proto_dict, dist_dict,
                        source_id=f"client_{cid}",
                        round_idx=epoch,
                        support_counts=support_counts,
                    )
                    if self.serverless_tracker is not None:
                        self.serverless_tracker.record_prototype_processing(len(proto_dict))

            # E-HSFP: Mix client outputs with memory prototypes
            if self.client_memory is not None and len(self.client_memory) > 0:
                for cid in list(client_outputs.keys()):
                    if client_outputs[cid] is not None:
                        proto_dict, dist_dict = client_outputs[cid]
                        mixed_p, mixed_d = mix_current_and_memory(
                            proto_dict, dist_dict, self.client_memory,
                            alpha=self.ecfg["memory_replay_ratio"],
                            top_k=self.ecfg["memory_top_k"],
                        )
                        client_outputs[cid] = (mixed_p, mixed_d)
                if self.serverless_tracker is not None:
                    self.serverless_tracker.record_memory_replay(len(client_outputs))

            # Free cache after the client phase
            torch.cuda.empty_cache()
            gc.collect()

            self._prof_add("client_pack", _t_pack, epoch)

            # --- PHASE 2: EDGE PROCESSING ---
            print(f"-> Phase 2: Edge Processing...")
            _t_edge = self._prof_now()
            edge_to_clients = {}
            for cid in idxs_users:
                eid = self.connectivity[-1][cid]
                edge_to_clients.setdefault(eid, []).append(cid)

            edge_outputs = {}
            for eid, cids in edge_to_clients.items():
                # E-HSFP: Simulate serverless invocation for edge
                if self.serverless_tracker is not None:
                    inv = self.serverless_tracker.simulate_invocation("edge", eid, epoch)
                    if inv.timed_out:
                        edge_outputs[eid] = None
                        continue

                edge_outputs[eid], edge_support = self._edge_ssl_extraction_phase(
                    eid, cids, client_outputs,
                    ssl_epochs=config.get("ssl_epochs_edge", 10),
                    syn_samples_per_class=config.get("syn_samples_per_class", 50)
                )

                cost = get_proto_dist_size_MB(edge_outputs[eid])
                self.comm_tracker["edge_to_cloud_data_MB"] += cost
                self._edge_support[eid] = edge_support

                # E-HSFP: Store edge prototypes in memory
                if self.edge_memory is not None and edge_outputs[eid] is not None:
                    proto_dict, dist_dict = edge_outputs[eid]
                    self.edge_memory.add_from_proto_dicts(
                        proto_dict, dist_dict,
                        source_id=f"edge_{eid}",
                        round_idx=epoch,
                        support_counts=edge_support,
                    )

            # E-HSFP: Mix edge outputs with memory
            if self.edge_memory is not None and len(self.edge_memory) > 0:
                for eid in list(edge_outputs.keys()):
                    if edge_outputs[eid] is not None:
                        proto_dict, dist_dict = edge_outputs[eid]
                        mixed_p, mixed_d = mix_current_and_memory(
                            proto_dict, dist_dict, self.edge_memory,
                            alpha=self.ecfg["memory_replay_ratio"],
                            top_k=self.ecfg["memory_top_k"],
                        )
                        edge_outputs[eid] = (mixed_p, mixed_d)

            # Important: delete client_outputs once edges finish to free RAM
            del client_outputs
            torch.cuda.empty_cache()
            gc.collect()

            self._prof_add("edge_process", _t_edge, epoch)

            # --- PHASE 3: CLOUD PROCESSING ---
            print(f"-> Phase 3: Cloud Supervised Training...")
            _t_cloud = self._prof_now()

            # E-HSFP: Simulate serverless invocation for cloud
            if self.serverless_tracker is not None:
                self.serverless_tracker.simulate_invocation("cloud", 0, epoch)

            cloud_loss = self._cloud_supervised_phase(
                0, edge_outputs,
                syn_epochs=config.get("syn_epochs_cloud", 10),
                syn_samples_per_class=config.get("syn_samples_per_class", 50)
            )
            cloud_loss_list.append(cloud_loss)
            self._prof_add("cloud_process", _t_cloud, epoch)

            del edge_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # --- E-HSFP: Age memories and train reliability ---
            if self.client_memory is not None:
                self.client_memory.age_all()
            if self.edge_memory is not None:
                self.edge_memory.age_all()
            if self.reliability_net is not None and self.client_memory is not None:
                rel_loss = train_reliability_bootstrap(
                    self.reliability_net, self.reliability_optimizer,
                    self.client_memory, self.device,
                )
                self.ehsfp_logger.log("ehsfp/reliability_train_loss", rel_loss)

            # --- PHASE 4: AGGREGATION & VALIDATION ---
            if epoch % t1 == 0:
                _t_agg = self._prof_now()
                self.edge_server_aggregation()
                self._prof_add("edge_aggregate", _t_agg, epoch)

            if epoch % t2 == 0:
                _t_cagg = self._prof_now()
                self.cloud_aggregation()
                self._prof_add("cloud_aggregate", _t_cagg, epoch)

                # Evaluate model
                f1, current_best_f1, model_snapshot = self._run_validation(valid_dataset, best_f1, epoch)
                validation_f1_list.append(f1)

                if model_snapshot is not None:
                    best_f1 = current_best_f1
                    best_pipeline_model = model_snapshot
                    # Save the best-model checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_snapshot.state_dict(),
                        'best_f1': best_f1,
                        'config': config
                    }, checkpoint_path)
                    print(f"*** Checkpoint saved: {checkpoint_path} (F1: {best_f1*100:.2f}%)")

            # E-HSFP: End serverless episode
            if self.serverless_tracker is not None:
                self.serverless_tracker.end_episode()

            # --- E-HSFP: Log metrics ---
            if self.client_memory is not None:
                mem_records = self.client_memory.get_recent()
                avg_age = sum(r.age for r in mem_records) / max(len(mem_records), 1)
                avg_rel = sum(r.reliability for r in mem_records) / max(len(mem_records), 1)
                self.ehsfp_logger.log_memory_stats(
                    num_current=0, num_memory=len(self.client_memory),
                    replay_ratio=self.ecfg["memory_replay_ratio"],
                    avg_age=avg_age, avg_reliability=avg_rel,
                )
            if self.proto_dropout is not None:
                stats = self.proto_dropout.stats
                self.ehsfp_logger.log_dropout_stats(
                    self.ecfg["prototype_dropout_rate"], stats["dropped_prototypes"],
                )
            self.ehsfp_logger.log("ehsfp/cloud_loss", cloud_loss)
            self.ehsfp_logger.log_communication(
                self.comm_tracker["client_to_edge_data_MB"] + self.comm_tracker["edge_to_cloud_data_MB"],
                sum(v for k, v in self.comm_tracker.items() if k != "total_comm_MB"),
            )

            epoch_time = time.time() - epoch_start_time
            if self.profiler is not None:
                self.profiler.add("total_round", epoch_time, epoch)
            log_data = {
                "epoch": epoch,
                "cloud_loss": cloud_loss,
                "epoch_time_s": epoch_time,
                "client_to_edge_MB": self.comm_tracker["client_to_edge_data_MB"],
                "edge_to_cloud_MB": self.comm_tracker["edge_to_cloud_data_MB"],
            }
            if validation_f1_list:
                log_data["validation_f1"] = validation_f1_list[-1]
                log_data["best_f1"] = best_f1
            # Merge E-HSFP metrics
            log_data.update(self.ehsfp_logger.get_current())
            if self.serverless_tracker is not None:
                log_data.update(self.serverless_tracker.get_summary())

            if wandb is not None and wandb.run is not None:
                wandb.log(log_data)

            # Advance the warmup+cosine LR schedule (no-op if disabled)
            self._step_lr_schedulers()

            print(f"Epoch {epoch} finished in {epoch_time:.2f}s")

        # --- end: final test ---
        print("\n" + "="*50)
        print("TRAINING FINISHED. Loading best model for testing...")

        # Reload the best model from file for testing
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            print(f"Loaded best model from epoch {checkpoint['epoch']}")

        self.print_comm_report()

        output = {
            "validation_f1": validation_f1_list,
            "cloud_loss": cloud_loss_list,
            "best_f1": best_f1,
            "best_weight": best_pipeline_model,
            "comm_report": self.comm_tracker,
            "ehsfp_metrics": self.ehsfp_logger.finalize(),
        }
        # camera-ready: persist fine-grained profiling, if enabled
        if self.profiler is not None:
            prof_summary = self.profiler.summary()
            output["profile_summary"] = prof_summary
            tag = config.get("profile_tag", "hsfp")
            try:
                from camera_ready.io_utils import cr_dir
                import os as _os
                out_dir = cr_dir("profiling")
                self.profiler.to_csv(_os.path.join(out_dir, f"profile_{tag}_records.csv"))
                self.profiler.to_json(_os.path.join(out_dir, f"profile_{tag}_summary.json"))
                print(f"[profile] wrote {out_dir}/profile_{tag}_*.csv/json")
            except Exception as _e:
                print(f"[profile] could not write profile files: {_e}")
        if self.serverless_tracker is not None:
            output["serverless_metrics"] = self.serverless_tracker.get_summary()
        return output