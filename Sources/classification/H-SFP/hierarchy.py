import copy
import json
import logging
import math
import os
import time
from collections import deque
import gc
import numpy as np
import psutil
import torch
from ehsfp.communication import add_communication, mb_of, new_communication_tracker
from ehsfp.history_logger import RoundHistoryLogger, communication_snapshot
from ehsfp.research_metrics import rounds_to_convergence, trailing_window_stability
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import kornia.augmentation as K
from sklearn.metrics import accuracy_score, f1_score
from classification.training_metrics import save_prediction_artifact

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
    RuntimeCounters,
    aggregate_proto_dicts,
    center_features,
    center_source_outputs,
    derive_global_feature_mean,
    derive_whitening_transform,
    recenter_memory,
    validate_prototype_space,
    whiten_features,
    whiten_source_outputs,
    COSINE_SPACES,
    compose_client_objective_loss,
    validate_client_objective,
    index_batch_hash,
    write_run_fingerprints,
)

# camera-ready: optional fine-grained phase profiling (gated by config["profile"])
try:
    from camera_ready.profiling import ProfileLog
except Exception:  # pragma: no cover - keeps legacy contexts working
    ProfileLog = None
from ehsfp.synthesis import (
    SharedCovarianceGenerator, within_class_scatter, pooled_covariance, scatter_payload_MB,
)
from ehsfp.episodic_stress import (
    StalePacketQueue, PartialEdgeExecution, total_variance_aggregate,
    gaussian_w2_drift, CsvAppender, DIAGNOSTIC_COLUMNS, RELIABILITY_COLUMNS,
)

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
            # Variance squares its inputs.  Features produced by a CUDA-autocast
            # model may be fp16, where moderately large activations overflow the
            # ~65504 range and silently create a non-finite prototype sigma.
            # Reduce in fp32; these statistics feed synthetic data and reliability.
            class_features = class_features.float()
            prototypes[cls_label] = torch.mean(class_features, dim=0)
            distributions_std[cls_label] = torch.std(
                class_features, dim=0, unbiased=False,
            )
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

class _FeatureNoiseAug(nn.Module):
    """SSL augmentation for pooled (1x1-spatial) feature maps.

    Spatial crops/flips/blur are meaningless on a [C,1,1] vector, so we build
    the two SupCon views by perturbing in feature space: additive Gaussian
    noise + feature dropout. Used at the edge when the client already pools to a
    semantic vector (the new resnet18 split)."""

    def __init__(self, noise_std=0.1, dropout_p=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x + torch.randn_like(x) * self.noise_std
        return self.dropout(x)


def build_edge_ssl_transforms(spatial_size):
    """Build edge SSL augmentations adaptive to feature map spatial dims.

    For pooled vector features (spatial_size <= 1) spatial ops are undefined, so
    fall back to feature-space noise/dropout to create contrastive views."""
    if spatial_size <= 1:
        return _FeatureNoiseAug(noise_std=0.1, dropout_p=0.1)
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
    # Keep the inexpensive SupCon reduction in fp32.  In particular, its
    # matmul/exp/log operations are not numerically safe under fp16 autocast.
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = nn.functional.normalize(features.float(), dim=1)
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
    return mb_of(state_dict)

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
    def __init__(
        self, client_model, edge_model, cloud_model,
        prototype_space="raw", client_to_edge_mean=None, edge_to_cloud_mean=None,
        client_to_edge_whitener=None, edge_to_cloud_whitener=None,
    ):
        super().__init__()
        self.client = client_model
        self.edge = edge_model
        self.cloud = cloud_model
        self.prototype_space = validate_prototype_space(prototype_space)
        self.register_buffer("client_to_edge_mean", client_to_edge_mean)
        self.register_buffer("edge_to_cloud_mean", edge_to_cloud_mean)
        self.register_buffer("client_to_edge_whitener", client_to_edge_whitener)
        self.register_buffer("edge_to_cloud_whitener", edge_to_cloud_whitener)

    def _apply_space(self, x, mean, whitener):
        if self.prototype_space == "centered_cosine":
            return center_features(x, mean)
        if self.prototype_space == "whitened_cosine":
            return whiten_features(x, mean, whitener)
        return x

    def forward(self, x):
        with torch.amp.autocast(device_type='cuda', enabled=(x.device.type == 'cuda')):
            x = self.client(x)
            x = self._apply_space(x, self.client_to_edge_mean, self.client_to_edge_whitener)
            x = self.edge(x)
            # Apply GAP + flatten to match the prototype extraction pipeline
            # (edge extraction does AdaptiveAvgPool2d + flatten before cloud)
            x = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(x), start_dim=1)
            x = self._apply_space(x, self.edge_to_cloud_mean, self.edge_to_cloud_whitener)
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
        self.prototype_space = validate_prototype_space(args.get("prototype_space"))
        self.client_objective = validate_client_objective(args.get("client_objective"))
        self.client_objective_weight = float(args.get("client_objective_weight", 0.1))
        if self.client_objective_weight < 0:
            raise ValueError("client_objective_weight must be non-negative")
        self.client_supervised_heads = {}
        self.client_supervised_optimizers = {}
        self.global_feature_means = {
            "client_to_edge": None,
            "edge_to_cloud": None,
        }
        # ZCA whitening transforms per boundary (whitened_cosine only). Derived
        # from the between-class scatter of aggregated prototype means; never
        # transmitted (computed at the reconstructing tier).
        self.global_feature_whiteners = {
            "client_to_edge": None,
            "edge_to_cloud": None,
        }

        # client_sync (default off = historical behaviour): standard federated
        # client handling. A sampled client starts its local episode from its
        # edge's current client model, and its update joins a per-edge running
        # average that is aggregated every t1 rounds. Historically every client
        # kept a private persistent model (200 GPU copies) and only the last
        # round's clients were ever averaged, so prototypes reaching an edge came
        # from encoders in different feature spaces.
        self.client_sync = bool(args.get("client_sync", False))
        self.structure, self.connectivity = self._build_hierarchy()
        self.total_layers = len(args["mid_server"]) + 1
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if getattr(self, "client_sync", False):
            init = {k: v.detach().clone().cpu() for k, v in self.client_model.state_dict().items()}
            num_edges = self.args["mid_server"][0]
            self.edge_client_state = {e: {k: v.clone() for k, v in init.items()} for e in range(num_edges)}
            self._client_sum = {}
            self._client_cnt = {}
            self._work_client = self.client_model.to(self.device)
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
        self.comm_tracker = new_communication_tracker()
        # --- end ---
        
        self.client_cache = deque(maxlen=20)
        self.optimizers = {}
        
        self.input_shape_client = None
        self.input_shape_edge = None
        self.input_shape_cloud = None

        # Configurable; default preserves prior behavior. persistent_workers=True
        # across many per-client DataLoaders (20 clients x N rounds) can exhaust/
        # deadlock worker processes at proxy scale — set num_workers=0 to disable.
        _default_workers = 2 if os.name != 'nt' else 0
        self.num_workers = int(self.args.get("num_workers", _default_workers))
        print(f"Using {self.num_workers} workers for the DataLoader.")

        # --- E-HSFP components ---
        self.ecfg = get_ehsfp_config(args)
        self.ehsfp_logger = EHSFPMetricsLogger()
        self.runtime_counters = RuntimeCounters()

        # Episodic memory (client-level and edge-level)
        if self.ecfg["use_episodic_memory"]:
            _per_class = args.get("memory_per_class")
            _per_class = None if _per_class in (None, "", "none") else int(_per_class)
            self.client_memory = EpisodicPrototypeMemory(
                max_size=self.ecfg["memory_size"],
                max_age=self.ecfg["max_prototype_age"],
                per_class_capacity=_per_class,
            )
            self.edge_memory = EpisodicPrototypeMemory(
                max_size=self.ecfg["memory_size"],
                max_age=self.ecfg["max_prototype_age"],
                per_class_capacity=_per_class,
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
            # Uniform and sample-count weighting do not use a learned network.
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
            self.serverless_tracker = ServerlessMetricsTracker({
                **self.ecfg,
                **{k: args[k] for k in ("client_timeout_probability", "edge_timeout_probability")
                   if args.get(k) is not None},
            })
        else:
            self.serverless_tracker = None

        # Residual generator (disabled by default)
        self.residual_generator = None

        # Prototype synthesis model: "diag" (eq. 2, historical default) or
        # "shared_cov" (class means + one pooled within-class covariance).
        self.synthesis = args.get("synthesis", "diag")
        if self.synthesis not in ("diag", "shared_cov"):
            raise ValueError(f"unknown synthesis: {self.synthesis}")
        self._gen_c2e = SharedCovarianceGenerator() if self.synthesis == "shared_cov" else None
        self._gen_e2c = SharedCovarianceGenerator() if self.synthesis == "shared_cov" else None
        self._client_scatter, self._edge_scatter = {}, {}
        # Cross-round sufficient-statistic accumulation at each edge (gamma=0:
        # off, historical). Each stateless episode sees ~2 samples/class/client,
        # far too few for stable class statistics; decayed sums over rounds fix
        # that without transmitting anything extra.
        self.stat_decay = float(args.get("stat_accumulation_decay", 0.0) or 0.0)
        self._edge_acc = {}

        # --- Episodic stress protocol (journal completion campaign) ---------
        # Neutral defaults reproduce every earlier run exactly: no queue
        # (tau=0), no partial execution (q=0), diagnostics off. Each mechanism
        # has its own seeded generator, so enabling it never shifts training RNG.
        _seed = int(args.get("seed", 0) or 0)
        self.stale_queue = StalePacketQueue(int(args.get("staleness_tau", 0) or 0), _seed)
        self.cold_start_defers = bool(args.get("cold_start_defers_packet", False))
        self._stale_mode = self.stale_queue.active or self.cold_start_defers
        if self._stale_mode and self.prototype_space != "raw":
            raise ValueError("staleness / cold-start deferral is implemented for prototype_space=raw only")
        self.partial_edge = PartialEdgeExecution(
            float(args.get("partial_edge_probability", 0.0) or 0.0),
            float(args.get("partial_edge_fraction", 0.5) or 0.5),
            _seed,
        )
        _diag_dir = args.get("diagnostics_dir")
        self.diag_writer = (
            CsvAppender(os.path.join(_diag_dir, "episodic_diagnostics.csv"), DIAGNOSTIC_COLUMNS)
            if _diag_dir else None
        )
        self.rel_writer = (
            CsvAppender(os.path.join(_diag_dir, "reliability_weights.csv"), RELIABILITY_COLUMNS)
            if (_diag_dir and self.reliability_net is not None) else None
        )
        self._current_round = 0
        self._packet_age = {}
        self._fresh_client_outputs = {}
        self._fresh_client_support = {}

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

    def _flush_runtime_counters(self, config, epoch, phase):
        """Durably persist cumulative counters at round phase boundaries."""
        path = config.get("runtime_counters_path")
        if not path:
            return
        record = {"epoch": epoch, "phase": phase, "device": str(self.device)}
        record.update(self.runtime_counters.snapshot())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

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
                        layer_dict[cid] = None if getattr(self, "client_sync", False) else copy.deepcopy(self.client_model)
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
                            layer_dict[cid] = None if getattr(self, "client_sync", False) else copy.deepcopy(self.client_model)
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
        if getattr(self, "client_sync", False):
            for eid, cnt in list(self._client_cnt.items()):
                if cnt == 0:
                    continue
                summed = self._client_sum[eid]
                new_state = {}
                for k, v in summed.items():
                    new_state[k] = (v / cnt).to(self.edge_client_state[eid][k].dtype).cpu() \
                        if v.is_floating_point() else v.cpu()
                self.edge_client_state[eid] = new_state
                add_communication(self.comm_tracker, "client_to_edge_MB",
                                  mb=get_model_size_MB(new_state), copies=cnt)
            for eid, (hs, n) in getattr(self, "_head_sum", {}).items():
                self._edge_head_state[eid] = {k: v / n for k, v in hs.items()}
            if hasattr(self, "_head_sum"):
                self._head_sum = {}
            self._client_sum, self._client_cnt = {}, {}
            self.edge_cache = {eid: {"edge_model": self.structure[0][eid].state_dict()}
                               for eid in self.structure[0]}
            return
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
            add_communication(self.comm_tracker, "client_to_edge_MB", mb=size_MB, copies=len(cids))

            for cid in cids:
                client_layer[cid].load_state_dict(avg_client_model)
            add_communication(self.comm_tracker, "edge_to_client_MB", mb=size_MB, copies=len(cids))

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
            add_communication(self.comm_tracker, "edge_to_cloud_MB", mb=size_edge_MB, copies=len(edge_ids))
            add_communication(self.comm_tracker, "cloud_to_edge_MB", mb=size_edge_MB, copies=len(edge_ids))

            for eid in edge_ids:
                edge_layer[eid].load_state_dict(avg_edge_model)

    def print_comm_report(self):
        """Print the communication-cost report and total."""
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
                if model is None:  # client_sync: clients get a fresh optimizer per episode
                    continue
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
        if getattr(self, "client_sync", False):
            eid = self.connectivity[-1][cid]
            model = self._work_client
            model.load_state_dict(self.edge_client_state[eid])
            add_communication(self.comm_tracker, "edge_to_client_MB",
                              mb=get_model_size_MB(self.edge_client_state[eid]))
            # Fresh local optimizer per episode (standard FL); LR follows the
            # same warmup+cosine schedule the persistent optimizers used.
            lr = self.args.get("ssl_lr", self.args["lr"]) * self._sched_factor()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=self.args["weight_decay"])
            scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
            # The client-local supervised head is synchronised per edge like the
            # encoder (otherwise every newly sampled client starts a random head).
            head_state = getattr(self, "_edge_head_state", {}).get(eid)
            if self.client_objective == "supervised" and head_state is not None:
                w = head_state["weight"]
                head = nn.Linear(w.shape[1], w.shape[0]).to(self.device)
                head.load_state_dict(head_state)
                self.client_supervised_heads[cid] = head
                self.client_supervised_optimizers[cid] = torch.optim.Adam(
                    head.parameters(), lr=lr, weight_decay=self.args["weight_decay"])
        else:
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
                head_optimizer = self.client_supervised_optimizers.get(cid)
                if head_optimizer is not None:
                    head_optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
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

                    # PLAN-6 auxiliaries see only this client's activations and
                    # labels. The default branch leaves the legacy tensor alone.
                    if self.client_objective == "ssl_supcon":
                        loss = compose_client_objective_loss(
                            loss, self.client_objective, z1_flat, target,
                            weight=self.client_objective_weight,
                            temperature=self.supcon_temp,
                        )
                    elif self.client_objective == "supervised":
                        head = self.client_supervised_heads.get(cid)
                        if head is None:
                            head = nn.Linear(z1_flat.shape[1], self.args["num_classes"]).to(
                                device=z1_flat.device
                            )
                            self.client_supervised_heads[cid] = head
                            head_optimizer = torch.optim.Adam(
                                head.parameters(),
                                lr=self.args.get("ssl_lr", self.args["lr"]),
                                weight_decay=self.args["weight_decay"],
                            )
                            self.client_supervised_optimizers[cid] = head_optimizer
                            head_optimizer.zero_grad(set_to_none=True)
                        loss = compose_client_objective_loss(
                            loss, self.client_objective,
                            torch.cat([z1_flat, z2_flat], dim=0), all_labels,
                            linear_head=head, weight=self.client_objective_weight,
                        )
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                if head_optimizer is not None:
                    scaler.step(head_optimizer)
                scaler.update()

        # 2. Extraction - avoid storing all features in a list when unnecessary
        model.eval()
        all_protos, all_stds = {}, {}
        
        # Optimization: accumulate per batch to avoid huge tensors
        # For simplicity and correct std, use class-wise grouping
        feats_by_cls = {}
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                # Cast immediately after the autocast forward, before retaining
                # activations for mean/std.  fp16 variance can overflow and poison
                # the sigma feature consumed by the reliability network.
                out = model(data).float()
                
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

        if getattr(self, "synthesis", "diag") == "shared_cov":
            self._client_scatter[cid] = within_class_scatter(feats_by_cls)
            add_communication(self.comm_tracker, "client_to_edge_MB",
                              mb=scatter_payload_MB(self._client_scatter[cid][0].shape[0]))
        # Manual cleanup
        del feats_by_cls
        gc.collect()
        if getattr(self, "client_sync", False):
            eid = self.connectivity[-1][cid]
            sd = model.state_dict()
            if eid not in self._client_sum:
                self._client_sum[eid] = {k: (v.detach().float().clone() if v.is_floating_point()
                                             else v.detach().clone()) for k, v in sd.items()}
                self._client_cnt[eid] = 1
            else:
                for k, v in sd.items():
                    if v.is_floating_point():
                        self._client_sum[eid][k] += v.detach().float()
                    else:
                        self._client_sum[eid][k] = v.detach().clone()
                self._client_cnt[eid] += 1
            head = self.client_supervised_heads.pop(cid, None)
            self.client_supervised_optimizers.pop(cid, None)
            if head is not None:
                hs = {k: v.detach().float().clone() for k, v in head.state_dict().items()}
                if not hasattr(self, "_head_sum"):
                    self._head_sum, self._edge_head_state = {}, {}
                if eid in self._head_sum:
                    for k in hs:
                        self._head_sum[eid][0][k] += hs[k]
                    self._head_sum[eid][1] += 1
                else:
                    self._head_sum[eid] = [hs, 1]
        return (all_protos, all_stds), support_counts

    def _sched_factor(self):
        """Current multiplier of the warmup+cosine schedule (1.0 if disabled)."""
        if not self.args.get("use_lr_schedule", False):
            return 1.0
        total = max(int(self._sched_total_epochs), 1)
        warmup = max(int(self.args.get("warmup_epochs", 0)), 0)
        step = max(int(self._current_round) - 1, 0)
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

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
        _support = {cid: getattr(self, "_client_support", {}).get(cid, {}) for cid in cids}
        _gen = self.residual_generator
        _acc_cov = None
        if self.stat_decay > 0:
            edge_specific_client_outputs, _support, _acc_cov = self._accumulate_edge_stats(
                eid, edge_specific_client_outputs, _support)
        if self.synthesis == "shared_cov":
            self._gen_c2e.set_covariance(_acc_cov if _acc_cov is not None else pooled_covariance(
                self._client_scatter[c] for c in edge_specific_client_outputs
                if edge_specific_client_outputs[c] is not None and c in self._client_scatter))
            _gen = self._gen_c2e
        # Kept for the optional larger extraction draw below (statistics only).
        self._edge_agg_inputs = (edge_specific_client_outputs, _gen, _support)
        _agg = reliability_weighted_aggregate(
            input_outputs=edge_specific_client_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device,
            reliability_net=self.reliability_net,
            memory=self.client_memory,
            generator=_gen,
            support_map=_support,
            runtime_counters=self.runtime_counters,
            aggregation_mode=self.ecfg["aggregation_mode"],
            age_map=({cid: self._packet_age.get(cid, 0) for cid in cids} if self._stale_mode else None),
            weight_log=(self._reliability_logger("edge", eid) if self.rel_writer is not None else None),
            return_stats=self.diag_writer is not None,
        )
        if self.diag_writer is not None:
            syn_features_L1, syn_labels_L1, _used_stats = _agg
            self._log_drift("edge", eid, _used_stats, cids, edge_specific_client_outputs)
        else:
            syn_features_L1, syn_labels_L1 = _agg

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
        
        # PRC inputs are loop-invariant within this phase (memory is only written
        # between phases); cached on first use, bit-identical to per-batch recompute.
        prc_inputs = None
        # Serverless partial execution: the edge function may be terminated
        # after a fraction of its SSL minibatches (q=0 -> full budget, no RNG).
        _batch_budget = self.partial_edge.batch_budget(ssl_epochs * len(syn_loader_L1))
        self._edge_partial = _batch_budget < ssl_epochs * len(syn_loader_L1)
        _batches_done = 0
        for epoch in range(ssl_epochs):
            if _batches_done >= _batch_budget:
                break
            total_loss = 0
            # Supervised Contrastive Training (Paper Alg. 2, Lines 9-11)
            for features, labels in syn_loader_L1:
                if _batches_done >= _batch_budget:
                    break
                _batches_done += 1
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
                    if self.ecfg["use_prc_loss"] and self.client_memory is not None and len(self.client_memory) > 0:
                        if prc_inputs is None:
                            prc_inputs = (
                                *self.client_memory.to_proto_dist_dicts(),
                                *aggregate_proto_dicts({
                                    cid: client_outputs[cid] for cid in cids
                                    if cid in client_outputs and client_outputs[cid] is not None
                                }, device=self.device),
                            )
                        mem_p, mem_d, cur_p, cur_d = prc_inputs
                        if cur_p and mem_p:
                            def edge_repr_fn(x):
                                return torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(model(x)), start_dim=1)
                            prc = prototype_replay_consistency_loss(
                                cur_p, cur_d, mem_p, mem_d,
                                edge_repr_fn, self.ecfg["prc_num_samples"], self.device,
                                runtime_counters=self.runtime_counters,
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
        # Optional: estimate the edge->cloud statistics from a LARGER synthetic
        # draw than the 50/class used for SSL training (forward passes only).
        # With 50/class the per-class edge mean carries fresh sampling noise
        # that undoes the cross-round accumulation (diagnosed on CIFAR-100).
        _n_extract = int(self.args.get("edge_extract_samples_per_class", 0) or 0)
        _eval_bs = self.args['local_bs']
        if _n_extract > syn_samples_per_class:
            _outs, _g, _sup = self._edge_agg_inputs
            with torch.no_grad():
                _xf, _yf = reliability_weighted_aggregate(
                    input_outputs=_outs, num_samples_per_class=_n_extract, device=self.device,
                    reliability_net=self.reliability_net, memory=self.client_memory,
                    generator=_g, support_map=_sup, aggregation_mode=self.ecfg["aggregation_mode"],
                )
            syn_dataset_L1 = torch.utils.data.TensorDataset(_xf.cpu(), _yf.cpu())
            del _xf, _yf
            _eval_bs = 512
        
        # Avoid FutureWarning
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
            syn_loader_L1_eval = DataLoader(
                syn_dataset_L1, # reuse dataset (features + labels)
                batch_size=_eval_bs, 
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
        if self.synthesis == "shared_cov":
            lab = syn_labels_L1_cpu.to(edge_features_L2.device)
            self._edge_scatter[eid] = within_class_scatter(
                {int(c): list(edge_features_L2[lab == c].float()) for c in torch.unique(lab).tolist()})
            add_communication(self.comm_tracker, "edge_to_cloud_MB",
                              mb=scatter_payload_MB(edge_features_L2.shape[1]))

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
        _gen = self.residual_generator
        if self.synthesis == "shared_cov":
            self._gen_e2c.set_covariance(pooled_covariance(
                self._edge_scatter[e] for e in edge_outputs
                if edge_outputs[e] is not None and e in self._edge_scatter))
            _gen = self._gen_e2c
        _agg = reliability_weighted_aggregate(
            input_outputs=edge_outputs,
            num_samples_per_class=int(self.args.get("cloud_samples_per_class", 0) or syn_samples_per_class),
            device=self.device,
            reliability_net=self.reliability_net,
            memory=self.edge_memory,
            generator=_gen,
            support_map=getattr(self, "_edge_support", {}),
            runtime_counters=self.runtime_counters,
            aggregation_mode=self.ecfg["aggregation_mode"],
            weight_log=(self._reliability_logger("cloud", cloud_id) if self.rel_writer is not None else None),
            return_stats=self.diag_writer is not None,
        )
        if self.diag_writer is not None:
            syn_features_L2, syn_labels_L2, _used_stats = _agg
            self._log_drift("cloud", cloud_id, _used_stats, list(edge_outputs.keys()), edge_outputs,
                            fresh=self._fresh_edge_outputs, fresh_support=getattr(self, "_edge_support", {}))
        else:
            syn_features_L2, syn_labels_L2 = _agg
        
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
        # PRC inputs are loop-invariant within this phase (memory is only written
        # between phases); cached on first use, bit-identical to per-batch recompute.
        prc_inputs = None
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
                        if prc_inputs is None:
                            prc_inputs = (
                                *self.edge_memory.to_proto_dist_dicts(),
                                *aggregate_proto_dicts(edge_outputs, device=self.device),
                            )
                        mem_p, mem_d, cur_p, cur_d = prc_inputs
                        if cur_p and mem_p:
                            prc = prototype_replay_consistency_loss(
                                cur_p, cur_d, mem_p, mem_d,
                                model, self.ecfg["prc_num_samples"], self.device,
                                runtime_counters=self.runtime_counters,
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
        
    def _accumulate_edge_stats(self, eid, outputs, support):
        """Fold this round's client packets into decayed per-class sums at edge eid.

        Returns a single synthetic source {"edge_acc": (mu, sd)}, its support and
        (for shared_cov) the accumulated pooled covariance.
        """
        g = self.stat_decay
        acc = self._edge_acc.setdefault(eid, {"N": {}, "S1": {}, "S2": {}, "shape": {},
                                              "scatter": None, "dof": 0.0})
        for c in acc["N"]:
            acc["N"][c] *= g; acc["S1"][c] *= g; acc["S2"][c] *= g
        if acc["scatter"] is not None:
            acc["scatter"] *= g; acc["dof"] *= g
        for k, out in outputs.items():
            if out is None:
                continue
            protos, stds = out
            for c, p in protos.items():
                n = float(support.get(k, {}).get(c, 0) or 1)
                x = p.detach().float().flatten().cpu()
                sd = stds[c].detach().float().flatten().cpu()
                acc["shape"][c] = p.shape
                acc["N"][c] = acc["N"].get(c, 0.0) + n
                acc["S1"][c] = acc["S1"].get(c, 0) + n * x
                acc["S2"][c] = acc["S2"].get(c, 0) + n * (sd ** 2 + x ** 2)
            sc = self._client_scatter.get(k)
            if sc is not None:
                S, n_s, k_s = sc
                acc["scatter"] = S.detach().float().clone() if acc["scatter"] is None else acc["scatter"] + S.to(acc["scatter"].device)
                acc["dof"] += float(n_s - k_s)
        mu, sdd, sup = {}, {}, {}
        for c, N in acc["N"].items():
            if N <= 0:
                continue
            m = acc["S1"][c] / N
            var = (acc["S2"][c] / N - m ** 2).clamp(min=1e-8)
            mu[c] = m.reshape(acc["shape"][c]).to(self.device)
            sdd[c] = var.sqrt().reshape(acc["shape"][c]).to(self.device)
            sup[c] = int(round(N))
        cov = acc["scatter"] / acc["dof"] if (acc["scatter"] is not None and acc["dof"] > 1) else None
        return {"edge_acc": (mu, sdd)}, {"edge_acc": sup}, cov

    def _reliability_logger(self, tier, node):
        """Callback recording reliability weights and their input factors (Fig. 8)."""
        def _log(class_id, sids, feats, weights):
            w = weights.float().cpu()
            wn = w / w.sum().clamp(min=1e-8)
            f = feats.float().cpu()
            for k, sid in enumerate(sids):
                self.rel_writer.append({
                    "round": self._current_round, "tier": tier, "node": node,
                    "class_id": int(class_id), "source": str(sid),
                    "support_feat": float(f[k, 0]), "sigma_feat": float(f[k, 1]),
                    "age_feat": float(f[k, 2]), "distance_feat": float(f[k, 3]),
                    "age": int(self._packet_age.get(sid, 0)) if tier == "edge" else 0,
                    "weight": float(w[k]), "normalized_weight": float(wn[k]),
                })
        return _log

    def _log_drift(self, tier, node, used_stats, source_ids, used_outputs,
                   fresh=None, fresh_support=None):
        """Definition 6 oracle drift: synthesized vs complete-fresh statistics.

        Reference = support-weighted total-variance aggregate of every packet
        GENERATED this round by this node's participating sources, before
        prototype dropout, staleness delay and memory replay. Diagnostic only.
        """
        if used_stats is None:
            return
        labels, mus, sigmas = used_stats
        used = ({int(l): mus[k] for k, l in enumerate(labels)},
                {int(l): sigmas[k] for k, l in enumerate(labels)})
        if tier == "edge":
            fresh = {cid: out for cid, out in self._fresh_client_outputs.items()
                     if self.connectivity[-1][cid] == node}
            fresh_support = self._fresh_client_support
        ref = total_variance_aggregate(fresh or {}, fresh_support)
        if not ref[0]:
            return
        class_w = {}
        for sid, sup in (fresh_support or {}).items():
            if fresh is not None and sid in fresh:
                for c, n in sup.items():
                    class_w[c] = class_w.get(c, 0.0) + float(n)
        d = gaussian_w2_drift(used, ref, class_w or None)
        if d is None:
            return
        ages = [self._packet_age.get(k, 0) for k in source_ids
                if used_outputs.get(k) is not None] if tier == "edge" else []
        self.diag_writer.append({
            "round": self._current_round, "tier": tier, "node": node, **d,
            "fresh_sources": len([v for v in (fresh or {}).values() if v is not None]),
            "used_sources": len([v for v in used_outputs.values() if v is not None]),
            "stale_packets": sum(1 for a in ages if a > 0),
            "mean_packet_age": (sum(ages) / len(ages)) if ages else 0.0,
            "max_packet_age": max(ages) if ages else 0,
            "partial_execution": int(getattr(self, "_edge_partial", False)) if tier == "edge" else 0,
        })

    def _run_validation(self, valid_dataset, best_val_top1, epoch):
        """(Phase 5) Run evaluation (optimized).

        Evaluate on the FedAvg GLOBAL client model (average of the clients
        actually trained this round), not an arbitrary ``structure[-1][0]``.
        Under a fixed seed, client 0 may never be selected, leaving its local
        model at the frozen pretrained init — which made every checkpoint grade
        a never-trained encoder. Averaging the cached (trained) clients gives a
        correct, seed-robust, baseline-fair evaluation of the trained encoder.
        """
        client_layer = self.structure[-1]
        trained_cids = [c for c in dict.fromkeys(self.client_cache) if c in client_layer]
        if getattr(self, "client_sync", False):
            # Global client = average of the edge-level client models.
            global_client_state = average_state_dicts(list(self.edge_client_state.values()))
            client_model = copy.deepcopy(self.client_model)
            client_model.load_state_dict(global_client_state)
        elif trained_cids:
            global_client_state = average_state_dicts(
                [client_layer[c].state_dict() for c in trained_cids]
            )
            client_model = copy.deepcopy(self.client_model)
            client_model.load_state_dict(global_client_state)
        else:
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
                out_cl = self._full_pipeline(client_model, edge_model, cloud_model)(data)

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

        accuracy = 0.0
        f1 = 0.0
        pipeline_model = None
        try:
            accuracy = accuracy_score(all_targets, all_preds)
            f1 = f1_score(
                all_targets, all_preds, average="macro", zero_division=0
            )

            # Preserve the historical accuracy-based checkpoint selection.
            if best_val_top1 < accuracy:
                print(f"Saved best model at epoch {epoch} with Acc: {accuracy * 100:.2f} %")
                best_val_top1 = accuracy
                pipeline_model = self._full_pipeline(
                    client_model=copy.deepcopy(client_model),
                    edge_model=copy.deepcopy(edge_model),
                    cloud_model=copy.deepcopy(cloud_model),
                )
        except ValueError as e:
            print(f"!!! ERROR calculating F1 score: {e}")
            print("!!! Check debug info above for potential issues (NaNs, label ranges, types).")
            # Keep best_val_top1 as it was, don't update pipeline_model
            
        return f1, accuracy, best_val_top1, pipeline_model, all_preds, all_targets

    def _full_pipeline(self, client_model, edge_model, cloud_model):
        """Build an eval/checkpoint pipeline in the same spaces used to train."""
        return FullPipelineModel(
            client_model,
            edge_model,
            cloud_model,
            prototype_space=self.prototype_space,
            client_to_edge_mean=self.global_feature_means["client_to_edge"],
            edge_to_cloud_mean=self.global_feature_means["edge_to_cloud"],
            client_to_edge_whitener=self.global_feature_whiteners["client_to_edge"],
            edge_to_cloud_whitener=self.global_feature_whiteners["edge_to_cloud"],
        )

    def _center_round_outputs(self, boundary, outputs, support_map, memory):
        """Express transmitted prototypes in the centered/whitened space.

        The global mean (and, for whitened_cosine, the ZCA whitening transform)
        is derived from the already-received class means; no feature tensor is
        added to the communication payload.
        """
        if self.prototype_space == "raw" or not any(v is not None for v in outputs.values()):
            return outputs
        old_mean = self.global_feature_means[boundary]
        new_mean = derive_global_feature_mean(outputs, support_map=support_map).detach().cpu()
        recenter_memory(memory, old_mean, new_mean)
        payload_before = sum(get_proto_dist_size_MB(value) for value in outputs.values())
        if self.prototype_space == "whitened_cosine":
            whitener = derive_whitening_transform(
                outputs, new_mean, support_map=support_map
            ).detach().cpu()
            outputs = whiten_source_outputs(outputs, new_mean, whitener)
            self.global_feature_whiteners[boundary] = whitener
        else:
            outputs = center_source_outputs(outputs, new_mean)
        payload_after = sum(get_proto_dist_size_MB(value) for value in outputs.values())
        assert payload_after == payload_before, "prototype-space transform must not add a transmitted tensor"
        self.global_feature_means[boundary] = new_mean
        return outputs

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
        validation_f1_list, validation_accuracy_list, cloud_loss_list = [], [], []
        best_f1 = 0
        best_val_top1 = 0
        best_pipeline_model = None
        best_validation_result = None
        last_round_result = None
        artifact_dir = config.get(
            "prediction_artifact_dir", os.path.dirname(os.path.abspath(checkpoint_path))
        )
        history_logger = RoundHistoryLogger(
            config.get(
                "history_path",
                os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "history.csv"),
            )
        )
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
            round_communication_start = communication_snapshot(self.comm_tracker)
            round_validation_accuracy = None
            self.ehsfp_logger.reset_epoch()
            self._current_round = epoch
            _tier_peaks = {}
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # E-HSFP: Begin serverless episode
            if self.serverless_tracker is not None:
                self.serverless_tracker.begin_episode(epoch)

            # 1. SELECT CLIENTS
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)
            if epoch == start_epoch and config.get("run_fingerprints_path"):
                fingerprint_path = config["run_fingerprints_path"]
                with open(fingerprint_path, "r", encoding="utf-8") as handle:
                    fingerprints = json.load(handle)
                fingerprints["first_sampled_clients"] = [int(cid) for cid in idxs_users]
                fingerprints["first_sampled_clients_hash"] = index_batch_hash(idxs_users)
                write_run_fingerprints(fingerprint_path, fingerprints)
            client_outputs = {}
            _cold_cids = set()
            # Per-round support counts feeding the reliability network
            self._client_support = {}
            self._edge_support = {}
            self._client_scatter, self._edge_scatter = {}, {}

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
                    if inv.is_cold_start:
                        _cold_cids.add(cid)

                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(local_data, shuffle=True, **dl_kwargs)

                # SSL & Extraction
                client_outputs[cid], support_counts = self._client_ssl_extraction_phase(
                    cid, loader, ssl_epochs=config.get("ssl_epochs_client", 10)
                )

                # Track data transmission
                cost = get_proto_dist_size_MB(client_outputs[cid])
                add_communication(self.comm_tracker, "client_to_edge_MB", mb=cost)
                self._client_support[cid] = support_counts

                # E-HSFP: Store client prototypes in memory
                if (self.prototype_space == "raw" and self.client_memory is not None
                        and client_outputs[cid] is not None and not self._stale_mode):
                    proto_dict, dist_dict = client_outputs[cid]
                    self.client_memory.add_from_proto_dicts(
                        proto_dict, dist_dict,
                        source_id=f"client_{cid}",
                        round_idx=epoch,
                        support_counts=support_counts,
                    )
                    self.runtime_counters.increment(
                        "memory.writes", len(set(proto_dict) & set(dist_dict))
                    )
                    if self.serverless_tracker is not None:
                        self.serverless_tracker.record_prototype_processing(len(proto_dict))

            client_outputs = self._center_round_outputs(
                "client_to_edge", client_outputs, self._client_support, self.client_memory
            )

            # Complete fresh packets of this round (oracle reference for drift).
            self._fresh_client_outputs = dict(client_outputs)
            self._fresh_client_support = dict(self._client_support)
            self._packet_age = {}

            # Definition 3: bounded-staleness delivery. Packets generated now are
            # delivered at round + d, d ~ U{0..tau} (+1 for a deferred cold start);
            # the edge -- and its memory -- see only what has arrived.
            if self._stale_mode:
                for cid in list(client_outputs):
                    if client_outputs[cid] is None:
                        continue
                    self.stale_queue.enqueue(
                        cid, epoch, client_outputs[cid], self._client_support.get(cid, {}),
                        extra_delay=1 if (self.cold_start_defers and cid in _cold_cids) else 0,
                    )
                delivered = self.stale_queue.deliver(epoch)
                client_outputs = {p.key: p.outputs for p in delivered}
                self._client_support = {p.key: p.support for p in delivered}
                self._packet_cid = {p.key: p.cid for p in delivered}
                self._packet_age = {p.key: p.age(epoch) for p in delivered}
                if self.client_memory is not None:
                    for p in delivered:
                        self.client_memory.add_from_proto_dicts(
                            p.outputs[0], p.outputs[1], source_id=f"client_{p.cid}",
                            round_idx=p.generated_round, support_counts=p.support,
                            age=p.age(epoch),
                        )
                        self.runtime_counters.increment(
                            "memory.writes", len(set(p.outputs[0]) & set(p.outputs[1]))
                        )

            # Centered-space memory is written only after the round-global mean
            # is derivable from all existing class prototype statistics.
            if self.prototype_space == "centered_cosine" and self.client_memory is not None:
                for cid, outputs in client_outputs.items():
                    if outputs is None:
                        continue
                    proto_dict, dist_dict = outputs
                    self.client_memory.add_from_proto_dicts(
                        proto_dict, dist_dict, source_id=f"client_{cid}", round_idx=epoch,
                        support_counts=self._client_support.get(cid, {}),
                    )
                    self.runtime_counters.increment(
                        "memory.writes", len(set(proto_dict) & set(dist_dict))
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
                            runtime_counters=self.runtime_counters,
                            device=self.device,
                        )
                        client_outputs[cid] = (mixed_p, mixed_d)
                if self.serverless_tracker is not None:
                    self.serverless_tracker.record_memory_replay(len(client_outputs))

            # Free cache after the client phase
            torch.cuda.empty_cache()
            gc.collect()

            self._prof_add("client_pack", _t_pack, epoch)
            self._flush_runtime_counters(config, epoch, "clients_complete")

            # --- PHASE 2: EDGE PROCESSING ---
            print(f"-> Phase 2: Edge Processing...")
            _t_edge = self._prof_now()
            edge_to_clients = {}
            if self._stale_mode:
                for key in client_outputs:
                    eid = self.connectivity[-1][self._packet_cid[key]]
                    edge_to_clients.setdefault(eid, []).append(key)
            else:
                for cid in idxs_users:
                    eid = self.connectivity[-1][cid]
                    edge_to_clients.setdefault(eid, []).append(cid)
            if torch.cuda.is_available():
                _tier_peaks["client"] = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()

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
                add_communication(self.comm_tracker, "edge_to_cloud_MB", mb=cost)
                self._edge_support[eid] = edge_support

                # E-HSFP: Store edge prototypes in memory
                if (self.prototype_space == "raw" and self.edge_memory is not None
                        and edge_outputs[eid] is not None):
                    proto_dict, dist_dict = edge_outputs[eid]
                    self.edge_memory.add_from_proto_dicts(
                        proto_dict, dist_dict,
                        source_id=f"edge_{eid}",
                        round_idx=epoch,
                        support_counts=edge_support,
                    )
                    self.runtime_counters.increment(
                        "memory.writes", len(set(proto_dict) & set(dist_dict))
                    )

            edge_outputs = self._center_round_outputs(
                "edge_to_cloud", edge_outputs, self._edge_support, self.edge_memory
            )
            self._fresh_edge_outputs = dict(edge_outputs)

            if self.prototype_space == "centered_cosine" and self.edge_memory is not None:
                for eid, outputs in edge_outputs.items():
                    if outputs is None:
                        continue
                    proto_dict, dist_dict = outputs
                    self.edge_memory.add_from_proto_dicts(
                        proto_dict, dist_dict, source_id=f"edge_{eid}", round_idx=epoch,
                        support_counts=self._edge_support.get(eid, {}),
                    )
                    self.runtime_counters.increment(
                        "memory.writes", len(set(proto_dict) & set(dist_dict))
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
                            runtime_counters=self.runtime_counters,
                            device=self.device,
                        )
                        edge_outputs[eid] = (mixed_p, mixed_d)

            # Important: delete client_outputs once edges finish to free RAM
            del client_outputs
            torch.cuda.empty_cache()
            gc.collect()

            self._prof_add("edge_process", _t_edge, epoch)
            self._flush_runtime_counters(config, epoch, "edges_complete")

            # --- PHASE 3: CLOUD PROCESSING ---
            print(f"-> Phase 3: Cloud Supervised Training...")
            _t_cloud = self._prof_now()
            if torch.cuda.is_available():
                _tier_peaks["edge"] = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
                torch.cuda.reset_peak_memory_stats()

            # E-HSFP: Simulate serverless invocation for cloud
            if self.serverless_tracker is not None:
                self.serverless_tracker.simulate_invocation("cloud", 0, epoch)

            cloud_loss = self._cloud_supervised_phase(
                0, edge_outputs,
                syn_epochs=config.get("syn_epochs_cloud", 10),
                syn_samples_per_class=config.get("syn_samples_per_class", 50)
            )
            cloud_loss_list.append(cloud_loss)
            if torch.cuda.is_available():
                _tier_peaks["cloud"] = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            self._prof_add("cloud_process", _t_cloud, epoch)
            self._flush_runtime_counters(config, epoch, "cloud_complete")

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

            # --- VALIDATION (decoupled from the t1/t2 aggregation cadence) ---
            # t1/t2 are a COMMUNICATION choice; validation is only measurement.
            # Coupling them meant large-interval settings (e.g. t2=50) were
            # validated just 4x in 200 epochs and missed their accuracy peak.
            # Evaluate on a fixed fine cadence so every interval setting's best
            # checkpoint is captured fairly. eval_every defaults to 1 (every
            # epoch); always evaluate on the final epoch too.
            eval_every = max(int(self.args.get("eval_every", 1)), 1)
            if epoch % eval_every == 0 or epoch == epochs:
                (
                    f1, accuracy, current_best_val_top1, model_snapshot,
                    all_preds, all_targets,
                ) = self._run_validation(valid_dataset, best_val_top1, epoch)
                current_result = {
                    "round": epoch, "accuracy": accuracy, "macro_f1": f1
                }
                if epoch == epochs:
                    last_round_result = current_result
                    save_prediction_artifact(
                        os.path.join(artifact_dir, "last_round_predictions.npz"),
                        all_preds, all_targets,
                    )
                validation_f1_list.append(f1)
                validation_accuracy_list.append(accuracy)
                round_validation_accuracy = accuracy
                best_f1 = max(best_f1, f1)

                if model_snapshot is not None:
                    best_val_top1 = current_best_val_top1
                    best_pipeline_model = model_snapshot
                    best_validation_result = current_result
                    save_prediction_artifact(
                        os.path.join(artifact_dir, "best_val_predictions.npz"),
                        all_preds, all_targets,
                    )
                    # Save the best-model checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_snapshot.state_dict(),
                        'best_f1': f1,
                        'best_val_top1': best_val_top1,
                        'config': config
                        ,'resolved_config_hash': config["resolved_config_hash"]
                    }, checkpoint_path)
                    print(f"*** Checkpoint saved: {checkpoint_path} (Acc: {best_val_top1*100:.2f}%)")

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
                self.ehsfp_logger.log_dict({
                    "ehsfp/runtime/dropout.apply_calls": stats["apply_calls"],
                    "ehsfp/runtime/dropout.changed_calls": stats["changed_calls"],
                    "ehsfp/runtime/dropout.total_seen": stats["total_prototypes_seen"],
                    "ehsfp/runtime/dropout.active_set_total": stats["active_set_total"],
                    "ehsfp/runtime/dropout.active_set_observations": stats["active_set_observations"],
                    "ehsfp/runtime/dropout.active_set_last": stats["active_set_last"],
                })
                for key in (
                    "apply_calls", "changed_calls", "total_prototypes_seen",
                    "active_set_total", "active_set_observations", "active_set_last",
                ):
                    counter_key = "total_seen" if key == "total_prototypes_seen" else key
                    self.runtime_counters.set(f"dropout.{counter_key}", stats[key])
            self.ehsfp_logger.log("ehsfp/cloud_loss", cloud_loss)
            self.ehsfp_logger.log_dict({f"ehsfp/runtime/{k}": v for k, v in self.runtime_counters.snapshot().items()})
            self.ehsfp_logger.log_communication(
                self.comm_tracker["client_to_edge_MB"] + self.comm_tracker["edge_to_cloud_MB"],
                self.comm_tracker["total_comm_MB"],
            )

            epoch_time = time.time() - epoch_start_time
            if self.profiler is not None:
                self.profiler.add("total_round", epoch_time, epoch)
            round_event = ""
            if self.serverless_tracker is not None and self.serverless_tracker.episodes:
                timeout_tiers = sorted({
                    invocation.tier
                    for invocation in self.serverless_tracker.episodes[-1].invocations
                    if invocation.timed_out
                })
                round_event = ";".join(f"{tier}_timeout" for tier in timeout_tiers)
            history_logger.append_round(
                round_number=epoch,
                elapsed_time_s=epoch_time,
                train_loss=cloud_loss,
                validation_metric_name="accuracy",
                validation_metric_value=round_validation_accuracy,
                communication_start=round_communication_start,
                communication_current=self.comm_tracker,
                client_memory_records=(
                    len(self.client_memory) if self.client_memory is not None else 0
                ),
                edge_memory_records=(
                    len(self.edge_memory) if self.edge_memory is not None else 0
                ),
                event=round_event,
                # Process-wide CUDA peak while each tier's phase ran (all tiers
                # share one simulated GPU, so this is an upper bound per tier).
                resources={
                    f"{tier}_gpu_peak_{kind}_bytes": vals[i]
                    for tier, vals in _tier_peaks.items()
                    for i, kind in enumerate(("allocated", "reserved"))
                },
            )
            log_data = {
                "epoch": epoch,
                "cloud_loss": cloud_loss,
                "epoch_time_s": epoch_time,
                "client_to_edge_MB": self.comm_tracker["client_to_edge_MB"],
                "edge_to_cloud_MB": self.comm_tracker["edge_to_cloud_MB"],
            }
            if validation_f1_list:
                log_data["validation_f1"] = validation_f1_list[-1]
                log_data["best_f1"] = best_f1
                log_data["validation_accuracy"] = validation_accuracy_list[-1]
                log_data["best_val_top1"] = best_val_top1
            # Merge E-HSFP metrics
            log_data.update(self.ehsfp_logger.get_current())
            if self.serverless_tracker is not None:
                log_data.update(self.serverless_tracker.get_summary())

            if wandb is not None and wandb.run is not None:
                wandb.log(log_data)
            self.ehsfp_logger.append_jsonl(config["runtime_metrics_path"], log_data)
            self._flush_runtime_counters(config, epoch, "round_complete")

            # Advance the warmup+cosine LR schedule (no-op if disabled)
            self._step_lr_schedulers()

            print(f"Epoch {epoch} finished in {epoch_time:.2f}s")

        # --- end: final test ---
        print("\n" + "="*50)
        print("TRAINING FINISHED. Loading best model for testing...")

        # Reload the best model from file for testing
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            from ehsfp import validate_checkpoint_config_hash
            validate_checkpoint_config_hash(checkpoint, config["resolved_config_hash"])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")

        self.print_comm_report()

        stability_window = 5
        convergence_definition = (
            "trailing moving average window=3; threshold=0.95*best validation "
            "accuracy; patience=3 consecutive smoothed rounds; 1-based round"
        )

        output = {
            "validation_f1": validation_f1_list,
            "validation_accuracy": validation_accuracy_list,
            "cloud_loss": cloud_loss_list,
            "best_f1": best_f1,
            "best_val_top1": best_val_top1,
            "last_round_validation": last_round_result,
            "best_validation": best_validation_result,
            "selected_checkpoint": (
                {"source": "best_validation", **best_validation_result}
                if best_validation_result is not None else None
            ),
            "stability_per_round": trailing_window_stability(
                validation_accuracy_list, window=stability_window
            ),
            "stability_window": stability_window,
            "rounds_to_convergence": rounds_to_convergence(validation_accuracy_list),
            "rounds_to_convergence_definition": convergence_definition,
            "best_weight": best_pipeline_model,
            "total_comm_MB": self.comm_tracker["total_comm_MB"],
            "comm_report": self.comm_tracker,
            "ehsfp_metrics": self.ehsfp_logger.finalize(),
            "ehsfp_runtime_counters": self.runtime_counters.snapshot(),
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
