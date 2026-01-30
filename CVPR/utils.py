import torch
from torch import nn
import kornia.augmentation as K
import kornia.geometry.transform as K_T
import numpy as np

# --- Các hàm tạo dữ liệu & Tính toán ---

def generate_synthetic_data(
    prototypes: torch.Tensor, 
    distributions_std: torch.Tensor, 
    num_samples_per_class: int
) -> (torch.Tensor, torch.Tensor):
    """
    Tạo ra cả features và labels tổng hợp từ prototypes (mean) và 
    distributions (std).
    """
    
    num_classes, feature_dim = prototypes.shape
    device = prototypes.device
    means = prototypes.unsqueeze(1)
    stds = distributions_std.unsqueeze(1)
    epsilon = torch.randn(
        num_classes, 
        num_samples_per_class, 
        feature_dim, 
        device=device
    )
    synthetic_features = (means + (stds * epsilon)).reshape(-1, feature_dim)
    labels_base = torch.arange(num_classes, device=device)
    synthetic_labels = torch.repeat_interleave(
        labels_base, 
        repeats=num_samples_per_class
    )
    return synthetic_features, synthetic_labels

def calculate_prototypes_and_distribution(fx: torch.Tensor, fy: torch.Tensor):
    """
    Tính toán prototype (mean) và distribution (std) cho các feature
    của từng lớp.
    """
    unique_classes = torch.unique(fy.cpu())
    prototypes = {}
    distributions_std = {}
    # print(f"Bắt đầu tính toán cho {len(unique_classes)} lớp duy nhất...") # (Giảm log)

    for cls in unique_classes:
        cls_label = cls.item() 
        class_features = fx[fy == cls]
        
        if class_features.shape[0] > 0:
            prototypes[cls_label] = torch.mean(class_features, dim=0)
            distributions_std[cls_label] = torch.std(class_features, dim=0, unbiased=False) 
        else:
            print(f"Cảnh báo: Lớp {cls_label} không có mẫu nào trong dữ liệu đã xử lý.")

    return prototypes, distributions_std

# --- Các hàm SSL (Self-Supervised Learning) ---

# Định nghĩa augmentations cho SSL
# (Kích thước 32, 32 đang được hard-code, có thể truyền vào làm tham số nếu cần)
ssl_transforms = nn.Sequential(
    K_T.Resize((32, 32)), 
    K.RandomResizedCrop(size=(32,32), scale=(0.5, 1.0)),
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    K.RandomGrayscale(p=0.2)
)

def info_nce_loss(z1, z2, temperature=0.5):
    """Tính InfoNCE loss cho đầu ra 4D (feature map) của model."""
    # Tự động thực hiện Global Average Pooling
    z1 = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(z1), start_dim=1)
    z2 = torch.flatten(nn.AdaptiveAvgPool2d((1,1))(z2), start_dim=1)
    
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.T, labels)
    
    return (loss_a + loss_b) / 2

# --- Các hàm tiện ích của class (HFL Utils) ---

def average_state_dicts(state_dicts):
    """Tính trung bình các state_dict của model."""
    if not state_dicts:
        return {}
    avg_dict = {}
    device = next(iter(state_dicts[0].values())).device

    for key in state_dicts[0].keys():
        tensors = [d[key].to(device) for d in state_dicts]
        avg_dict[key] = sum(tensors) / len(tensors)
    return avg_dict

def get_model_size_MB(state_dict):
    """Tính kích thước model (MB) từ state_dict."""
    return (
        sum(param.numel() for param in state_dict.values()) * 4 / 1e6
    )  # MB

def estimate_gradient_size_MB(model, input_shape, device="cpu"):
    """Ước lượng kích thước output của model (MB)."""
    model = model.to(device).eval()
    # Thêm batch dimension nếu input_shape là (C, H, W)
    if len(input_shape) == 3:
        input_shape = (1,) + input_shape 
        
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    numel = output.numel()
    element_size = output.element_size()
    size_MB = (numel * element_size) / (1024**2)
    return size_MB