# scripts/00_dataset_preparation.py

import yaml
import os
import torch
import numpy as np
import requests
import zipfile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


# =====================================================================================
# CELL 1: Bảng điều khiển cấu hình dữ liệu ⚙️
# Đây là nơi duy nhất bạn cần chỉnh sửa.
# Định nghĩa cấu hình dữ liệu bạn muốn sử dụng cho lần chạy mô phỏng tiếp theo.
# =====================================================================================
class DataConfig:
    # --- 1. Chọn bộ dữ liệu, cách phân phối, và số lượng client ---
    DATASET_NAME = "cifar10"  # 'cifar10', 'cifar100', 'tiny_imagenet'
    DISTRIBUTION_MODE = "iid"  # 'iid', 'non_iid_shards', 'non_iid_dirichlet'
    NUM_CLIENTS = 20  # Số lượng client để chia dữ liệu

    # --- 2. Các tham số chi tiết (chỉ được dùng khi cần) ---
    SHARDS_PER_CLIENT = 2  # Dành cho 'non_iid_shards'
    DIRICHLET_ALPHA = 0.5  # Dành cho 'non_iid_dirichlet'

    # --- 3. Đường dẫn ---
    CONFIG_PATH = "../configs/config.yaml"  # Nơi lưu file config sẽ được tạo ra
    DATA_PATH = "../data"  # Nơi lưu các bộ dữ liệu được tải về


# =====================================================================================
# CELL 2: Hàm tạo/ghi đè file config.yaml 📝
# =====================================================================================
def generate_single_config_file(config):
    """Tạo ra một file config duy nhất dựa trên các thiết lập hiện tại."""

    # Gom các tham số dữ liệu vào một dictionary để lưu
    params_to_save = {
        "data_config": {
            "dataset_name": config.DATASET_NAME,
            "distribution_mode": config.DISTRIBUTION_MODE,
            "num_clients": config.NUM_CLIENTS,
            "shards_per_client": config.SHARDS_PER_CLIENT,
            "dirichlet_alpha": config.DIRICHLET_ALPHA,
        }
    }

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(config.CONFIG_PATH), exist_ok=True)

    # Mở file ở chế độ 'w' (write) để ghi đè
    with open(config.CONFIG_PATH, "w") as f:
        yaml.dump(params_to_save, f, default_flow_style=False, sort_keys=False)

    print(
        f"✅ Đã ghi đè file cấu hình tại: {os.path.abspath(config.CONFIG_PATH)}"
    )
    print("\n📜 Nội dung file:")
    print(yaml.dump(params_to_save, default_flow_style=False, sort_keys=False))


# =====================================================================================
# CELL 3: Các hàm xử lý dữ liệu (để notebook khác import) 🛠️
# =====================================================================================
class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data, self.targets = data, torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.targets[idx]
        img = (
            Image.fromarray(sample)
            if isinstance(sample, np.ndarray)
            else Image.open(sample).convert("RGB")
        )
        if self.transform:
            img = self.transform(img)
        return img, label


def download_and_prepare_tiny_imagenet(data_path):
    dataset_path = os.path.join(data_path, "tiny-imagenet-200")
    if os.path.exists(dataset_path):
        return
    print("Đang tải Tiny ImageNet (~240MB)...")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_path, "tiny-imagenet-200.zip")
    os.makedirs(data_path, exist_ok=True)
    r = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Giải nén...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)
    print("Cấu trúc lại thư mục validation...")
    val_dir = os.path.join(dataset_path, "val")
    val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
    val_img_to_class = {
        line.strip().split("\t")[0]: line.strip().split("\t")[1]
        for line in open(val_annotations_path, "r")
    }
    for img, cls in val_img_to_class.items():
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        os.rename(
            os.path.join(val_dir, "images", img),
            os.path.join(val_dir, cls, img),
        )
    os.rmdir(os.path.join(val_dir, "images"))
    os.remove(zip_path)
    print("Tiny ImageNet đã sẵn sàng.")


def prepare_data(
    dataset_name,
    num_clients,
    data_path,
    batch_size=32,
    seed=42,
    distribution_mode="iid",
    **kwargs,
):
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_ds_raw = datasets.CIFAR10(data_path, train=True, download=True)
        test_ds = datasets.CIFAR10(
            data_path, train=False, download=True, transform=transform
        )
        num_classes = 10
        train_data, train_targets = train_ds_raw.data, np.array(
            train_ds_raw.targets
        )
    elif dataset_name == "cifar100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        train_ds_raw = datasets.CIFAR100(data_path, train=True, download=True)
        test_ds = datasets.CIFAR100(
            data_path, train=False, download=True, transform=transform
        )
        num_classes = 100
        train_data, train_targets = train_ds_raw.data, np.array(
            train_ds_raw.targets
        )
    elif dataset_name == "tiny_imagenet":
        download_and_prepare_tiny_imagenet(data_path)
        transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )
        train_ds_raw = datasets.ImageFolder(
            os.path.join(data_path, "tiny-imagenet-200/train")
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_path, "tiny-imagenet-200/val"),
            transform=transform,
        )
        num_classes = 200
        train_data, train_targets = [
            s[0] for s in train_ds_raw.samples
        ], np.array([s[1] for s in train_ds_raw.samples])
    else:
        raise ValueError(f"Dataset '{dataset_name}' không được hỗ trợ.")
    indices = np.arange(len(train_data))
    client_indices = []
    if distribution_mode == "iid":
        np.random.seed(seed)
        np.random.shuffle(indices)
        client_indices = np.array_split(indices, num_clients)
    elif distribution_mode == "non_iid_shards":
        shards_per_client = kwargs.get("shards_per_client", 2)
        num_shards = num_clients * shards_per_client
        sorted_indices = np.argsort(train_targets)
        shards = np.array_split(sorted_indices, num_shards)
        np.random.seed(seed)
        np.random.shuffle(shards)
        shards_split = np.array_split(shards, num_clients)
        client_indices = [np.concatenate(s, axis=0) for s in shards_split]
    elif distribution_mode == "non_iid_dirichlet":
        alpha = kwargs.get("dirichlet_alpha", 0.5)
        np.random.seed(seed)
        label_dist = np.random.dirichlet([alpha] * num_classes, num_clients)
        class_indices = [
            indices[train_targets == i] for i in range(num_classes)
        ]
        [np.random.shuffle(ci) for ci in class_indices]
        class_starts = [0] * num_classes
        client_indices = [[] for _ in range(num_clients)]
        for i in range(num_clients):
            client_i_indices = []
            for j in range(num_classes):
                take = int(label_dist[i, j] * len(class_indices[j]))
                client_i_indices.extend(
                    class_indices[j][class_starts[j] : class_starts[j] + take]
                )
                class_starts[j] += take
            client_indices[i] = client_i_indices
    client_loaders = []
    for idxs in client_indices:
        data_subset = [train_data[i] for i in idxs]
        targets_subset = train_targets[idxs]
        client_ds = CustomDataset(
            data_subset, targets_subset, transform=transform
        )
        client_loaders.append(
            DataLoader(client_ds, batch_size=batch_size, shuffle=True)
        )
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
    return (
        client_loaders,
        test_loader,
        (
            test_ds.classes
            if hasattr(test_ds, "classes")
            else train_ds_raw.classes
        ),
    )


# =====================================================================================
# Main execution block
# =====================================================================================
if __name__ == "__main__":
    # Khởi tạo đối tượng config từ lớp đã định nghĩa ở trên
    config = DataConfig()

    # Chạy hàm để tạo file config
    generate_single_config_file(config)
