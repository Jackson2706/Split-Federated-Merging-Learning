import copy
import logging
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
from sklearn.metrics import f1_score
# =============================================================================
# SECTION 1: CÁC HÀM TIỆN ÍCH (UTILS)
# =============================================================================

@torch.no_grad()
def generate_synthetic_data(prototypes, distributions_std, num_samples_per_class):
    """
    Tối ưu bộ nhớ bằng cách tạo dữ liệu trực tiếp trên thiết bị của prototypes.
    """
    num_classes = prototypes.shape[0]
    shape = prototypes.shape[1:]
    device = prototypes.device
    
    # Sử dụng torch.addmm hoặc broadcasting thông minh để tránh tạo tensor epsilon quá lớn rồi mới cộng
    # Shape: [C, N, ...]
    means = prototypes.unsqueeze(1) 
    stds = distributions_std.unsqueeze(1)
    
    # Tạo trực tiếp trên GPU, sử dụng dtype phù hợp để tiết kiệm VRAM
    epsilon = torch.randn(num_classes, num_samples_per_class, *shape, 
                          device=device, dtype=prototypes.dtype)
    
    # In-place operation để tiết kiệm bộ nhớ
    epsilon.mul_(stds).add_(means)
    
    return epsilon.flatten(0, 1) # [C*N, ...]

def _aggregate_prototypes_and_generate_data(input_outputs, num_samples_per_class, device):
    """
    Gom nhóm và tạo dữ liệu nhanh hơn bằng cách giảm bớt các vòng lặp Python.
    """
    if not input_outputs:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    # Gom nhóm theo label nhanh hơn bằng dictionary
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

    # Giải phóng dictionary ngay lập tức
    del merged
    
    features = generate_synthetic_data(final_protos, final_dists, num_samples_per_class)
    labels = torch.tensor(final_labels, device=device).repeat_interleave(num_samples_per_class)
    
    return features, labels

    
def calculate_prototypes_and_distribution(fx: torch.Tensor, fy: torch.Tensor):
    """
    Tính toán prototype và distribution.
    (fx (features) trên GPU, fy (labels) trên CPU).
    """
    unique_classes = torch.unique(fy) 
    prototypes = {}
    distributions_std = {}

    for cls in unique_classes:
        cls_label = cls.item() 
        mask = (fy == cls).to(fx.device)
        class_features = fx[mask]
        
        if class_features.shape[0] > 0:
            # Sửa cảnh báo FutureWarning
            with torch.amp.autocast(device_type='cuda', enabled=(fx.device.type == 'cuda')):
                prototypes[cls_label] = torch.mean(class_features, dim=0)
                distributions_std[cls_label] = torch.std(class_features, dim=0, unbiased=False)
        else:
            print(f"Cảnh báo: Lớp {cls_label} không có mẫu nào trong dữ liệu đã xử lý.")

    return prototypes, distributions_std

# --- Các hàm SSL (Self-Supervised Learning) ---

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
    """Tính InfoNCE loss cho đầu ra 4D (feature map)."""
    z1 = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(z1), start_dim=1)
    z2 = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(z2), start_dim=1)
    
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    sim_matrix = torch.matmul(z1, z2.mT) / temperature # Sửa .T
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.mT, labels) # Sửa .T
    
    return (loss_a + loss_b) / 2

def info_nce_loss_2d(z1, z2, temperature=0.5):
    """Tính InfoNCE loss cho đầu vào 2D (vector feature)."""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    sim_matrix = torch.matmul(z1, z2.mT) / temperature # Sửa .T
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.mT, labels) # Sửa .T
    
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

# --- Các hàm tiện ích của class (HFL Utils) ---

def average_state_dicts(state_dicts):
    if not state_dicts: return {}
    avg_dict = {}
    device = next(iter(state_dicts[0].values())).device
    for key in state_dicts[0].keys():
        tensors = [d[key].to(device) for d in state_dicts]
        avg_dict[key] = sum(tensors) / len(tensors)
    return avg_dict

def get_model_size_MB(state_dict):
    """Tính kích thước model (MB) từ state_dict."""
    return (sum(param.numel() for param in state_dict.values()) * 4 / 1e6)

def get_proto_dist_size_MB(proto_dist_tuple: tuple) -> float:
    """(MỚI) Tính toán kích thước (MB) của tuple (proto_dict, dist_dict)."""
    if proto_dist_tuple is None:
        return 0.0
        
    proto_dict, dist_dict = proto_dist_tuple
    total_bytes = 0
    
    # Tính kích thước của tất cả tensor trong dict prototypes
    for tensor in proto_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
        
    # Tính kích thước của tất cả tensor trong dict distributions
    for tensor in dist_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
        
    return total_bytes / (1024**2)

# =============================================================================
# SECTION 2: ĐỊNH NGHĨA DỮ LIỆU VÀ MÔ HÌNH
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
# SECTION 3: LỚP HIERARCHICAL FL CHÍNH
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
        
        # --- (MỚI) Cập nhật Comm Tracker ---
        self.comm_tracker = {
            "client_to_edge_data_MB": 0.0,  # (MỚI) Chi phí gửi (proto, dist)
            "edge_to_cloud_data_MB": 0.0,   # (MỚI) Chi phí gửi (proto, dist)
            "client_model_upload_MB": 0.0,  # (CŨ) Chi phí FedAvg model
            "client_model_download_MB": 0.0,
            "edge_model_upload_MB": 0.0,
            "edge_model_download_MB": 0.0,
            "total_comm_MB": 0.0           # (MỚI) Tổng chi phí
        }
        # --- KẾT THÚC SỬA ---
        
        self.client_cache = deque(maxlen=20)
        self.optimizers = {}
        
        self.input_shape_client = None
        self.input_shape_edge = None
        self.input_shape_cloud = None

        self.num_workers = 2 if os.name != 'nt' else 0
        print(f"Sử dụng {self.num_workers} workers cho DataLoader.")


    def _build_hierarchy(self):
        # ... (Mã gốc của bạn - không đổi) ...
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
        # ... (Mã gốc của bạn - không đổi) ...
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

    # --- Các hàm tổng hợp (Aggregation) - Giai đoạn 4 ---
    def edge_server_aggregation(self):
        # ... (Mã gốc của bạn - không đổi) ...
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
        # ... (Mã gốc của bạn - không đổi) ...
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
        """(MỚI) In báo cáo chi phí giao tiếp và tính tổng."""
        total = 0.0
        for k, v in self.comm_tracker.items():
            if k != "total_comm_MB":
                total += v
        self.comm_tracker["total_comm_MB"] = total
        
        print("\n=== Communication Report ===")
        for k, v in self.comm_tracker.items():
            print(f"{k}: {v:.2f} MB")

    def initialize_optimizers(self):
        """Khởi tạo optimizers VÀ GradScalers cho tất cả model.
        Paper Section 4: 'All models use the Adam optimizer (lr = 1e-4)'
        """
        self.optimizers = {}
        for layer, nodes in self.structure.items():
            self.optimizers[layer] = {}
            for nid, model in nodes.items():
                self.optimizers[layer][nid] = {
                    'optimizer': torch.optim.Adam(
                        model.parameters(),
                        lr=self.args["lr"],
                        weight_decay=self.args["weight_decay"],
                    ),
                    'scaler': torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
                }

    # --- Các hàm con cho vòng lặp Huấn luyện (ĐÃ TỐI ƯU) ---

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
                    loss = supervised_contrastive_loss(all_features, all_labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # 2. Extraction - Không lưu toàn bộ features vào list nếu không cần
        model.eval()
        all_protos, all_stds = {}, {}
        
        # Tối ưu: Tính toán tích lũy theo batch để tránh lưu tensor khổng lồ
        # Tuy nhiên để đơn giản và chính xác với std, ta dùng class-wise grouping
        feats_by_cls = {}
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                out = model(data)
                
                # Chuyển target sang CPU một lần
                target_cpu = target.numpy()
                for i, cls_id in enumerate(target_cpu):
                    if cls_id not in feats_by_cls: feats_by_cls[cls_id] = []
                    feats_by_cls[cls_id].append(out[i])

        # Tính mean/std cho từng class
        for cls_id, tensors in feats_by_cls.items():
            stacked = torch.stack(tensors)
            all_protos[cls_id] = stacked.mean(0).cpu() # Đẩy về CPU để tiết kiệm VRAM
            all_stds[cls_id] = stacked.std(0, unbiased=False).cpu()
        
        # Dọn dẹp thủ công
        del feats_by_cls
        gc.collect() 
        return all_protos, all_stds

    def _edge_ssl_extraction_phase(self, eid, cids, client_outputs, ssl_epochs, syn_samples_per_class):
        """(Giai đoạn 2) Chạy SSL và trích xuất (proto, dist) cho 1 Edge."""
        model = self.structure[0][eid]
        opt_dict = self.optimizers[0][eid]
        optimizer = opt_dict['optimizer']
        scaler = opt_dict['scaler']
        
        # --- 2a. Thu thập và Tạo Data L1 (SỬA LỖI LOGIC NHÃN) ---
        edge_specific_client_outputs = {cid: client_outputs[cid] for cid in cids if cid in client_outputs}

        # Dùng helper để gom nhóm nhãn gốc
        syn_features_L1, syn_labels_L1 = _aggregate_prototypes_and_generate_data(
            input_outputs=edge_specific_client_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device
        )
        
        if syn_features_L1.shape[0] == 0:
             print(f"Edge {eid}: Không có prototype nào từ client, bỏ qua.")
             return None
        
        print(f"Edge {eid}: Đã tạo {syn_features_L1.shape[0]} mẫu L1 (với nhãn gốc). Bắt đầu SSL...")

        # --- 2b. Huấn luyện SSL ---
        model.to(self.device).train()
        
        # Chuyển data về CPU
        syn_dataset_L1 = torch.utils.data.TensorDataset(
            syn_features_L1.cpu(), 
            syn_labels_L1.cpu() # <-- SỬA LỖI LOGIC (Thêm labels)
        )
        del syn_features_L1, syn_labels_L1
        
        syn_loader_L1 = DataLoader(
            syn_dataset_L1, 
            batch_size=self.args['local_bs'], 
            shuffle=True,
            # --- SỬA LỖI AttributeError ---
            num_workers=0, # Dữ liệu đã ở trong RAM
            pin_memory=False,
            # --- KẾT THÚC SỬA ---
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
                    loss = supervised_contrastive_loss(all_features, all_labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
        print(f"Edge {eid}: Hoàn tất SSL, Loss cuối: {total_loss/len(syn_loader_L1):.4f}")

        # --- 2c. Trích xuất Proto/Dist (SỬA LỖI LOGIC SHAPE) ---
        model.to(self.device).eval()
        edge_feats_L2 = []
        
        # Sửa cảnh báo FutureWarning
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
            syn_loader_L1_eval = DataLoader(
                syn_dataset_L1, # Tái sử dụng dataset (có cả feats, labels)
                batch_size=self.args['local_bs'], 
                shuffle=False,
                # --- SỬA LỖI AttributeError ---
                num_workers=0,
                pin_memory=False
                # (KHÔNG có drop_last=True)
                # --- KẾT THÚC SỬA ---
            )
            # Sửa lỗi: DataLoader trả về (features, labels)
            for features, _ in syn_loader_L1_eval: # Chỉ lấy features
                features = features.to(self.device)
                out_4d = model(features)
                
                # Làm phẳng 4D -> 2D cho Cloud
                out_2d = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(out_4d), start_dim=1)
                
                if self.input_shape_cloud is None:
                    self.input_shape_cloud = out_2d.shape[1:]
                
                edge_feats_L2.append(out_2d)
        
        edge_features_L2 = torch.cat(edge_feats_L2, dim=0) # (trên GPU, 2D)
        
        # Lấy nhãn GỐC (từ dataset CPU)
        syn_labels_L1_cpu = syn_dataset_L1.tensors[1] 
        
        # Trả về dict {nhãn_gốc: tensor_2D}
        edge_protos_dists = calculate_prototypes_and_distribution(edge_features_L2, syn_labels_L1_cpu)
        
        del syn_loader_L1, syn_loader_L1_eval, edge_features_L2, syn_labels_L1_cpu, syn_dataset_L1
        return edge_protos_dists

    def _cloud_supervised_phase(self, cloud_id, edge_outputs, syn_epochs, syn_samples_per_class):
        """(Giai đoạn 3) Chạy Supervised training cho Cloud."""
        model = self.structure[len(self.args["mid_server"])][cloud_id]
        opt_dict = self.optimizers[len(self.args["mid_server"])][cloud_id]
        optimizer = opt_dict['optimizer']
        scaler = opt_dict['scaler']
        
        # --- 3a. Thu thập và Tạo Data L2 (SỬA LỖI LOGIC NHÃN) ---
        # Dùng helper để gom nhóm nhãn gốc
        syn_features_L2, syn_labels_L2 = _aggregate_prototypes_and_generate_data(
            input_outputs=edge_outputs,
            num_samples_per_class=syn_samples_per_class,
            device=self.device
        )
        
        if syn_features_L2.shape[0] == 0:
            print(f"Cloud {cloud_id}: Không có prototype nào từ Edge, bỏ qua.")
            return 0.0

        print(f"Cloud {cloud_id}: Đã tạo {syn_features_L2.shape[0]} mẫu L2 (với nhãn gốc). Bắt đầu Supervised...")

        # --- 3b. Huấn luyện Supervised (CrossEntropy) ---
        model.to(self.device).train()
        
        syn_dataset_L2 = torch.utils.data.TensorDataset(
            syn_features_L2.cpu(), 
            syn_labels_L2.cpu() # nhãn gốc (ví dụ: 0-99)
        )
        del syn_features_L2, syn_labels_L2
        
        syn_loader_L2 = DataLoader(
            syn_dataset_L2, 
            batch_size=self.args['local_bs'], 
            shuffle=True,
            # --- SỬA LỖI AttributeError ---
            num_workers=0,
            pin_memory=False,
            # --- KẾT THÚC SỬA ---
            drop_last=True
        )
        
        total_loss = 0
        for epoch in range(syn_epochs):
            epoch_loss = 0
            for features, labels in syn_loader_L2:
                features, labels = features.to(self.device), labels.to(self.device, non_blocking=True)
                
                # Sửa cảnh báo FutureWarning
                with torch.amp.autocast(device_type='cuda', enabled=(self.device.type == 'cuda')):
                    logits = model(features)
                    loss = self.criterion(logits, labels) # (Giờ đã chính xác)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                
            total_loss = epoch_loss / len(syn_loader_L2)
        
        print(f"Cloud {cloud_id}: Hoàn tất Supervised, Loss cuối: {total_loss:.4f}")
        del syn_dataset_L2, syn_loader_L2
        return total_loss
        
    def _run_validation(self, valid_dataset, best_f1, epoch):
        """(Giai đoạn 5) Chạy đánh giá (đã tối ưu)."""
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

        f1 = 0.0 # Default value
        pipeline_model = None
        try:
            # Calculate F1 score (where the error occurred)
            f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) # Added zero_division=0
            
            if best_f1 < f1:
                print(f"Lưu model tốt nhất tại epoch {epoch} với F1: {f1 * 100:.2f} %")
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

    # --- Phương thức Huấn luyện Chính (Đã tổ chức lại) --

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
        self.initialize_optimizers()
        
        # Lấy cấu hình
        num_users = config["num_users"]
        frac = config["frac"]
        local_bs = config["local_bs"]
        t1, t2 = int(config["t1"]), int(config["t2"])
        
        # Khởi tạo lưu trữ metrics
        validation_f1_list, cloud_loss_list = [], []
        best_f1 = 0
        best_pipeline_model = None
        start_epoch = 1

        # --- TỐI ƯU DATALOADER ---
        # persistent_workers=True giúp DataLoader không bị khởi tạo lại mỗi epoch
        # pin_memory=True giúp chuyển dữ liệu lên GPU nhanh hơn
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
            
            # 1. CHỌN CLIENTS
            m = max(int(frac * num_users), 1)
            idxs_users = np.random.choice(range(num_users), m, replace=False)
            client_outputs = {}

            # --- GIAI ĐOẠN 1: CLIENT PROCESSING ---
            print(f"-> Phase 1: Clients Processing ({len(idxs_users)} nodes)...")
            for cid in idxs_users:
                self.client_cache.append(cid)
                local_data = DatasetSplit(train_dataset, user_groups[cid])
                loader = DataLoader(local_data, shuffle=True, **dl_kwargs)

                # SSL & Extraction
                client_outputs[cid] = self._client_ssl_extraction_phase(
                    cid, loader, ssl_epochs=config.get("ssl_epochs_client", 10)
                )
                
                # Theo dõi truyền tải dữ liệu
                cost = get_proto_dist_size_MB(client_outputs[cid])
                self.comm_tracker["client_to_edge_data_MB"] += cost

            # Giải phóng bộ nhớ đệm sau phase Client
            torch.cuda.empty_cache()
            gc.collect()

            # --- GIAI ĐOẠN 2: EDGE PROCESSING ---
            print(f"-> Phase 2: Edge Processing...")
            edge_to_clients = {}
            for cid in idxs_users:
                eid = self.connectivity[-1][cid]
                edge_to_clients.setdefault(eid, []).append(cid)

            edge_outputs = {}
            for eid, cids in edge_to_clients.items():
                edge_outputs[eid] = self._edge_ssl_extraction_phase(
                    eid, cids, client_outputs, 
                    ssl_epochs=config.get("ssl_epochs_edge", 10), 
                    syn_samples_per_class=config.get("syn_samples_per_class", 50)
                )
                
                cost = get_proto_dist_size_MB(edge_outputs[eid])
                self.comm_tracker["edge_to_cloud_data_MB"] += cost

            # Quan trọng: Xóa client_outputs ngay khi Edge xong để giải phóng RAM
            del client_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # --- GIAI ĐOẠN 3: CLOUD PROCESSING ---
            print(f"-> Phase 3: Cloud Supervised Training...")
            cloud_loss = self._cloud_supervised_phase(
                0, edge_outputs, 
                syn_epochs=config.get("syn_epochs_cloud", 10), 
                syn_samples_per_class=config.get("syn_samples_per_class", 50)
            )
            cloud_loss_list.append(cloud_loss)

            del edge_outputs
            torch.cuda.empty_cache()
            gc.collect()

            # --- GIAI ĐOẠN 4: AGGREGATION & VALIDATION ---
            if epoch % t1 == 0:
                self.edge_server_aggregation()
                
            if epoch % t2 == 0:
                self.cloud_aggregation()
                
                # Đánh giá model
                f1, current_best_f1, model_snapshot = self._run_validation(valid_dataset, best_f1, epoch)
                validation_f1_list.append(f1)
                
                if model_snapshot is not None:
                    best_f1 = current_best_f1
                    best_pipeline_model = model_snapshot
                    # LƯU CHECKPOINT MODEL TỐT NHẤT
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_snapshot.state_dict(),
                        'best_f1': best_f1,
                        'config': config
                    }, checkpoint_path)
                    print(f"*** Checkpoint saved: {checkpoint_path} (F1: {best_f1*100:.2f}%)")

            print(f"Epoch {epoch} hoàn tất trong {time.time() - epoch_start_time:.2f}s")

        # --- KẾT THÚC: TEST CUỐI CÙNG ---
        print("\n" + "="*50)
        print("TRAINING FINISHED. Loading best model for testing...")
        
        # Load lại model tốt nhất từ file để test
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            # Giả sử bạn có hàm tạo model từ state_dict
            # best_pipeline_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")

        self.print_comm_report()
        
        return {
            "validation_f1": validation_f1_list,
            "cloud_loss": cloud_loss_list,
            "best_f1": best_f1,
            "best_weight": best_pipeline_model,
            "comm_report": self.comm_tracker
        }