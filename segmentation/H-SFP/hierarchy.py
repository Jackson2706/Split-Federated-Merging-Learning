import copy
import gc
import logging
import os
import time
from collections import deque

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import kornia.augmentation as K

try:
    import wandb
except ImportError:
    wandb = None


# =============================================================================
# SECTION 1: UTILS
# =============================================================================

@torch.no_grad()
def generate_synthetic_data(prototypes, distributions_std, num_samples_per_class):
    num_classes = prototypes.shape[0]
    shape = prototypes.shape[1:]
    device = prototypes.device
    means = prototypes.unsqueeze(1)
    stds = distributions_std.unsqueeze(1)
    epsilon = torch.randn(
        num_classes, num_samples_per_class, *shape,
        device=device, dtype=prototypes.dtype,
    )
    epsilon.mul_(stds).add_(means)
    return epsilon.flatten(0, 1)


def _aggregate_prototypes_and_generate_data(input_outputs, num_samples_per_class, device):
    if not input_outputs:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    merged = {}
    for sid, outputs in input_outputs.items():
        if outputs is None:
            continue
        proto_dict, dist_dict = outputs
        for label, proto in proto_dict.items():
            if label not in merged:
                merged[label] = {"p": [], "d": []}
            merged[label]["p"].append(proto)
            merged[label]["d"].append(dist_dict[label])

    if not merged:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    final_labels = sorted(merged.keys())
    final_protos = torch.stack(
        [torch.stack(merged[l]["p"]).mean(0) for l in final_labels]
    ).to(device)
    final_dists = torch.stack([
        torch.sqrt(torch.stack([d**2 for d in merged[l]["d"]]).mean(0))
        for l in final_labels
    ]).to(device)
    del merged

    features = generate_synthetic_data(final_protos, final_dists, num_samples_per_class)
    labels = torch.tensor(final_labels, device=device).repeat_interleave(num_samples_per_class)
    return features, labels


def calculate_prototypes_and_distribution(fx, fy):
    """Calculate per-class prototypes (mean) and distributions (std).

    For segmentation, fy contains class labels derived from masks:
    0 = background, 1 = foreground.
    """
    unique_classes = torch.unique(fy)
    prototypes = {}
    distributions_std = {}

    for cls in unique_classes:
        cls_label = cls.item()
        mask = (fy == cls).to(fx.device)
        class_features = fx[mask]

        if class_features.shape[0] > 0:
            with torch.amp.autocast(device_type="cuda", enabled=(fx.device.type == "cuda")):
                prototypes[cls_label] = torch.mean(class_features, dim=0)
                distributions_std[cls_label] = torch.std(class_features, dim=0, unbiased=False)

    return prototypes, distributions_std


def _mask_to_label(mask, threshold=0.5):
    """Convert a segmentation mask [B, 1, H, W] to a per-image binary label.

    Returns 1 (foreground) if foreground ratio > threshold fraction of pixels,
    else 0 (background). This is used for prototype grouping.
    """
    # Use fraction of foreground pixels > a small threshold to determine class
    fg_ratio = mask.view(mask.size(0), -1).mean(dim=1)
    return (fg_ratio > 0.1).long()


# --- SSL ---

def build_client_ssl_transforms(input_size):
    return nn.Sequential(
        K.RandomResizedCrop(size=(input_size, input_size), scale=(0.5, 1.0)),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
    )


def build_edge_ssl_transforms(spatial_size):
    return nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomResizedCrop(size=(spatial_size, spatial_size), scale=(0.8, 1.0)),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
    )


def supervised_contrastive_loss(features, labels, temperature=0.5):
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


def average_state_dicts(state_dicts):
    if not state_dicts:
        return {}
    avg_dict = {}
    device = next(iter(state_dicts[0].values())).device
    for key in state_dicts[0].keys():
        tensors = [d[key].to(device) for d in state_dicts]
        avg_dict[key] = sum(tensors) / len(tensors)
    return avg_dict


def get_model_size_MB(state_dict):
    return sum(param.numel() for param in state_dict.values()) * 4 / 1e6


def get_proto_dist_size_MB(proto_dist_tuple):
    if proto_dist_tuple is None:
        return 0.0
    proto_dict, dist_dict = proto_dist_tuple
    total_bytes = 0
    for tensor in proto_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    for tensor in dist_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


# =============================================================================
# SECTION 2: DATA & MODEL WRAPPERS
# =============================================================================

class FullPipelineModel(nn.Module):
    """For inference: client -> edge -> decoder (spatial prediction)."""

    def __init__(self, client_model, edge_model, cloud_decoder):
        super().__init__()
        self.client = client_model
        self.edge = edge_model
        self.decoder = cloud_decoder

    def forward(self, x):
        x = self.client(x)
        x = self.edge(x)
        x = self.decoder(x)
        return x


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, mask = self.dataset[self.idxs[index]]
        return image.clone(), mask.clone()


# =============================================================================
# SECTION 3: HIERARCHICAL FL
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
        cloud_decoder,
        test_dataset=None,
    ):
        self.args = args
        self.client_model = client_model
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.cloud_decoder = cloud_decoder
        self.client_weight = client_weights
        self.edge_weight = edge_weights
        self.cloud_weight = cloud_weight

        self.structure, self.connectivity = self._build_hierarchy()
        self.total_layers = len(args["mid_server"]) + 1
        self.test_dataset = test_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_size = args.get("input_size", 224)
        self.ssl_transforms = build_client_ssl_transforms(input_size).to(self.device)
        self.ssl_transforms_edge = None
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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
        self.num_workers = 2 if os.name != "nt" else 0

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
                        else list(np.random.choice(all_clients, clients_per_edge, replace=False))
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
                        else list(np.random.choice(all_prev, servers_per_layer, replace=False))
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
                "Client Layer" if layer_idx == -1
                else ("Cloud Layer" if layer_idx == len(self.args["mid_server"]) else f"Edge Layer {layer_idx}")
            )
            logging.info(f"\n=== {layer_type} (Layer {layer_idx}) ===")
            for node_id in layer_nodes:
                logging.info(f"  Node ID {node_id}")

    # --- Aggregation ---

    def edge_server_aggregation(self):
        client_layer = self.structure[-1]
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
            self.edge_cache[eid] = {"edge_model": self.structure[0][eid].state_dict()}

            size_MB = get_model_size_MB(avg_client_model)
            self.comm_tracker["client_model_upload_MB"] += len(cids) * size_MB
            for cid in cids:
                client_layer[cid].load_state_dict(avg_client_model)
            self.comm_tracker["client_model_download_MB"] += len(cids) * size_MB

    def cloud_aggregation(self):
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
        total = sum(v for k, v in self.comm_tracker.items() if k != "total_comm_MB")
        self.comm_tracker["total_comm_MB"] = total
        print("\n=== Communication Report ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

    def initialize_optimizers(self):
        self.optimizers = {}
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {}
            for nid, model in nodes.items():
                self.optimizers[layer][nid] = {
                    "optimizer": torch.optim.Adam(
                        model.parameters(),
                        lr=self.args["lr"],
                        weight_decay=self.args["weight_decay"],
                    ),
                    "scaler": torch.amp.GradScaler(enabled=(self.device.type == "cuda")),
                }
        # Decoder optimizer
        self.decoder_optimizer = torch.optim.Adam(
            self.cloud_decoder.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )
        self.decoder_scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

    # --- Phase 1: Client SSL + Extraction ---

    def _client_ssl_extraction_phase(self, cid, loader, ssl_epochs):
        model = self.structure[-1][cid].to(self.device)
        opt_dict = self.optimizers[-1][cid]
        optimizer, scaler = opt_dict["optimizer"], opt_dict["scaler"]

        # SSL with SupCon on raw images
        model.train()
        for _ in range(ssl_epochs):
            for data, mask in loader:
                data = data.to(self.device, non_blocking=True)
                # Derive binary label from mask for SupCon
                target = _mask_to_label(mask).to(self.device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=self.device.type):
                    v1, v2 = self.ssl_transforms(data), self.ssl_transforms(data)
                    z1, z2 = model(v1), model(v2)
                    z1_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), start_dim=1)
                    z2_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), start_dim=1)
                    all_features = torch.cat([z1_flat, z2_flat], dim=0)
                    all_labels = torch.cat([target, target], dim=0)
                    loss = supervised_contrastive_loss(all_features, all_labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Extract prototypes per class (fg/bg)
        model.eval()
        feats_by_cls = {}
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            for data, mask in loader:
                data = data.to(self.device, non_blocking=True)
                out = model(data)
                target = _mask_to_label(mask)
                target_np = target.numpy()
                for i, cls_id in enumerate(target_np):
                    if cls_id not in feats_by_cls:
                        feats_by_cls[cls_id] = []
                    feats_by_cls[cls_id].append(out[i])

        all_protos, all_stds = {}, {}
        for cls_id, tensors in feats_by_cls.items():
            stacked = torch.stack(tensors)
            all_protos[cls_id] = stacked.mean(0).cpu()
            all_stds[cls_id] = stacked.std(0, unbiased=False).cpu()

        del feats_by_cls
        gc.collect()
        return all_protos, all_stds

    # --- Phase 2: Edge SSL + Extraction ---

    def _edge_ssl_extraction_phase(self, eid, cids, client_outputs, ssl_epochs, syn_samples_per_class):
        model = self.structure[0][eid]
        opt_dict = self.optimizers[0][eid]
        optimizer, scaler = opt_dict["optimizer"], opt_dict["scaler"]

        edge_specific_client_outputs = {cid: client_outputs[cid] for cid in cids if cid in client_outputs}
        syn_features_L1, syn_labels_L1 = _aggregate_prototypes_and_generate_data(
            input_outputs=edge_specific_client_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device,
        )

        if syn_features_L1.shape[0] == 0:
            return None

        model.to(self.device).train()
        syn_dataset_L1 = torch.utils.data.TensorDataset(syn_features_L1.cpu(), syn_labels_L1.cpu())
        del syn_features_L1, syn_labels_L1

        syn_loader_L1 = DataLoader(
            syn_dataset_L1, batch_size=self.args["local_bs"],
            shuffle=True, num_workers=0, pin_memory=False, drop_last=True,
        )

        for _ in range(ssl_epochs):
            for features, labels in syn_loader_L1:
                features = features.to(self.device)
                labels = labels.to(self.device)

                if self.ssl_transforms_edge is None:
                    spatial_size = features.shape[-1]
                    self.ssl_transforms_edge = build_edge_ssl_transforms(spatial_size).to(self.device)

                with torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                    view_1 = self.ssl_transforms_edge(features)
                    view_2 = self.ssl_transforms_edge(features)
                    z1, z2 = model(view_1), model(view_2)
                    z1_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), start_dim=1)
                    z2_flat = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), start_dim=1)
                    all_features = torch.cat([z1_flat, z2_flat], dim=0)
                    all_labels = torch.cat([labels, labels], dim=0)
                    loss = supervised_contrastive_loss(all_features, all_labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Extract edge-level prototypes
        model.to(self.device).eval()
        edge_feats_L2 = []
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
            syn_loader_eval = DataLoader(
                syn_dataset_L1, batch_size=self.args["local_bs"],
                shuffle=False, num_workers=0, pin_memory=False,
            )
            for features, _ in syn_loader_eval:
                features = features.to(self.device)
                out_4d = model(features)
                out_2d = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(out_4d), start_dim=1)
                edge_feats_L2.append(out_2d)

        edge_features_L2 = torch.cat(edge_feats_L2, dim=0)
        syn_labels_L1_cpu = syn_dataset_L1.tensors[1]
        edge_protos_dists = calculate_prototypes_and_distribution(edge_features_L2, syn_labels_L1_cpu)

        del syn_loader_L1, syn_loader_eval, edge_features_L2, syn_dataset_L1
        return edge_protos_dists

    # --- Phase 3: Cloud Supervised Training ---

    def _cloud_supervised_phase(self, cloud_id, edge_outputs, syn_epochs, syn_samples_per_class):
        model = self.structure[len(self.args["mid_server"])][cloud_id]
        opt_dict = self.optimizers[len(self.args["mid_server"])][cloud_id]
        optimizer, scaler = opt_dict["optimizer"], opt_dict["scaler"]

        syn_features_L2, syn_labels_L2 = _aggregate_prototypes_and_generate_data(
            input_outputs=edge_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device,
        )

        if syn_features_L2.shape[0] == 0:
            return 0.0

        model.to(self.device).train()
        syn_dataset_L2 = torch.utils.data.TensorDataset(syn_features_L2.cpu(), syn_labels_L2.cpu())
        del syn_features_L2, syn_labels_L2

        syn_loader_L2 = DataLoader(
            syn_dataset_L2, batch_size=self.args["local_bs"],
            shuffle=True, num_workers=0, pin_memory=False, drop_last=True,
        )

        total_loss = 0
        for _ in range(syn_epochs):
            epoch_loss = 0
            for features, labels in syn_loader_L2:
                features, labels = features.to(self.device), labels.to(self.device)
                with torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                    logits = model(features)
                    loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            total_loss = epoch_loss / max(len(syn_loader_L2), 1)

        del syn_dataset_L2, syn_loader_L2
        return total_loss

    # --- Phase 5: Validation ---

    def _run_validation(self, test_dataset, best_iou, epoch):
        """Run validation using the full pipeline (client -> edge -> decoder)."""
        from clients import compute_iou_and_dice, DiceFocalLoss

        client_model = self.structure[-1][0]
        eid = self.connectivity[-1][0]
        edge_model = self.structure[0][eid]

        client_model.to(self.device).eval()
        edge_model.to(self.device).eval()
        self.cloud_decoder.to(self.device).eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_iou, test_dice, total_samples = 0.0, 0.0, 0

        with torch.no_grad():
            for data, mask in loader:
                data, mask = data.to(self.device), mask.to(self.device)
                out = client_model(data)
                out = edge_model(out)
                out = self.cloud_decoder(out)
                out = (out > 0.5).float()
                iou, dice = compute_iou_and_dice(out, mask)
                test_iou += iou
                test_dice += dice
                total_samples += 1

        test_iou /= max(total_samples, 1)
        test_dice /= max(total_samples, 1)

        pipeline_model = None
        if test_iou > best_iou:
            best_iou = test_iou
            print(f"New best IoU at epoch {epoch}: {best_iou * 100:.2f}%")
            pipeline_model = FullPipelineModel(
                client_model=copy.deepcopy(client_model),
                edge_model=copy.deepcopy(edge_model),
                cloud_decoder=copy.deepcopy(self.cloud_decoder),
            )

        return test_iou, test_dice, best_iou, pipeline_model

    # --- Phase 3b: Train decoder on full pipeline ---

    def _train_decoder_phase(self, train_dataset, user_groups, idxs_users, decoder_epochs):
        """Train the cloud decoder using full forward pass through frozen client+edge."""
        from clients import DiceFocalLoss

        client_model = self.structure[-1][0]
        eid = self.connectivity[-1][0]
        edge_model = self.structure[0][eid]

        client_model.to(self.device).eval()
        edge_model.to(self.device).eval()
        self.cloud_decoder.to(self.device).train()

        seg_criterion = DiceFocalLoss().to(self.device)

        # Use a subset of data for decoder training
        all_idxs = []
        for cid in idxs_users:
            all_idxs.extend(list(user_groups[cid]))
        subset = DatasetSplit(train_dataset, all_idxs)
        loader = DataLoader(subset, batch_size=self.args["local_bs"], shuffle=True, drop_last=True)

        for _ in range(decoder_epochs):
            for data, mask in loader:
                data, mask = data.to(self.device), mask.to(self.device)
                self.decoder_optimizer.zero_grad()

                with torch.no_grad():
                    feat = client_model(data)
                    feat = edge_model(feat)

                with torch.amp.autocast(device_type="cuda", enabled=(self.device.type == "cuda")):
                    pred = self.cloud_decoder(feat)
                    loss = seg_criterion(pred, mask)

                self.decoder_scaler.scale(loss).backward()
                self.decoder_scaler.step(self.decoder_optimizer)
                self.decoder_scaler.update()

        self.cloud_decoder.eval()
        return loss.item() if isinstance(loss, torch.Tensor) else 0.0

    # --- Main Training Loop ---

    def train_end_to_end(
        self,
        train_dataset,
        valid_dataset,
        test_dataset,
        user_groups,
        config,
        epochs,
        checkpoint_path="checkpoint_hsfp_seg.pt",
    ):
        self.initialize_optimizers()

        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]
        t1, t2 = int(config["t1"]), int(config["t2"])

        validation_iou_list, validation_dice_list, cloud_loss_list = [], [], []
        best_iou = 0
        best_pipeline_model = None

        dl_kwargs = {
            "batch_size": local_bs,
            "num_workers": self.num_workers,
            "pin_memory": True if torch.cuda.is_available() else False,
            "persistent_workers": True if self.num_workers > 0 else False,
            "drop_last": True,
        }

        for epoch in range(1, epochs + 1):
            print(f"\n{'='*20} EPOCH {epoch}/{epochs} {'='*20}")
            epoch_start_time = time.time()

            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)
            client_outputs = {}

            # --- PHASE 1: CLIENT PROCESSING ---
            print(f"-> Phase 1: Clients Processing ({len(idxs_users)} nodes)...")
            for cid in idxs_users:
                self.client_cache.append(cid)
                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(local_data, shuffle=True, **dl_kwargs)
                client_outputs[cid] = self._client_ssl_extraction_phase(
                    cid, loader, ssl_epochs=config.get("ssl_epochs_client", 10),
                )
                cost = get_proto_dist_size_MB(client_outputs[cid])
                self.comm_tracker["client_to_edge_data_MB"] += cost

            torch.cuda.empty_cache()
            gc.collect()

            # --- PHASE 2: EDGE PROCESSING ---
            print("-> Phase 2: Edge Processing...")
            edge_to_clients = {}
            for cid in idxs_users:
                eid = self.connectivity[-1][cid]
                edge_to_clients.setdefault(eid, []).append(cid)

            edge_outputs = {}
            for eid, cids in edge_to_clients.items():
                edge_outputs[eid] = self._edge_ssl_extraction_phase(
                    eid, cids, client_outputs,
                    ssl_epochs=config.get("ssl_epochs_edge", 10),
                    syn_samples_per_class=config.get("syn_samples_per_class", 50),
                )
                cost = get_proto_dist_size_MB(edge_outputs[eid])
                self.comm_tracker["edge_to_cloud_data_MB"] += cost

            del client_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # --- PHASE 3: CLOUD PROCESSING ---
            print("-> Phase 3: Cloud Supervised Training...")
            cloud_loss = self._cloud_supervised_phase(
                0, edge_outputs,
                syn_epochs=config.get("syn_epochs_cloud", 10),
                syn_samples_per_class=config.get("syn_samples_per_class", 50),
            )
            cloud_loss_list.append(cloud_loss)

            del edge_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # --- PHASE 3b: DECODER TRAINING ---
            print("-> Phase 3b: Decoder Training...")
            decoder_loss = self._train_decoder_phase(
                train_dataset, user_groups, idxs_users,
                decoder_epochs=config.get("decoder_epochs", 5),
            )

            # --- PHASE 4: AGGREGATION & VALIDATION ---
            if epoch % t1 == 0:
                self.edge_server_aggregation()

            if epoch % t2 == 0:
                self.cloud_aggregation()

                iou, dice, current_best_iou, model_snapshot = self._run_validation(
                    test_dataset, best_iou, epoch,
                )
                validation_iou_list.append(iou)
                validation_dice_list.append(dice)

                if model_snapshot is not None:
                    best_iou = current_best_iou
                    best_pipeline_model = model_snapshot
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model_snapshot.state_dict(),
                        "best_iou": best_iou,
                        "config": config,
                    }, checkpoint_path)
                    print(f"*** Checkpoint saved: {checkpoint_path} (IoU: {best_iou*100:.2f}%)")

            epoch_time = time.time() - epoch_start_time

            if wandb is not None and wandb.run is not None:
                log_data = {
                    "epoch": epoch,
                    "cloud_loss": cloud_loss,
                    "decoder_loss": decoder_loss,
                    "epoch_time_s": epoch_time,
                    "client_to_edge_MB": self.comm_tracker["client_to_edge_data_MB"],
                    "edge_to_cloud_MB": self.comm_tracker["edge_to_cloud_data_MB"],
                }
                if validation_iou_list:
                    log_data["validation_iou"] = validation_iou_list[-1]
                    log_data["validation_dice"] = validation_dice_list[-1]
                    log_data["best_iou"] = best_iou
                wandb.log(log_data)

            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

        print("\n" + "=" * 50)
        print("TRAINING FINISHED.")
        self.print_comm_report()

        return {
            "validation_iou": validation_iou_list,
            "validation_dice": validation_dice_list,
            "cloud_loss": cloud_loss_list,
            "best_iou": best_iou,
            "best_weight": best_pipeline_model,
            "comm_report": self.comm_tracker,
        }
