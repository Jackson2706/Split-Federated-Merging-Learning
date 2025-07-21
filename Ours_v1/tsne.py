import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch import nn


# ========== Config ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/jackson/Desktop/Split-Federated-Merging-Learning/cifar10_iid:True_alexnet_20 users_t1:10_t2:20.pt"
batch_size = 128
n_components = 2
perplexity = 30
random_state = 42

# ========== Transforms ==========
valid_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

# ========== Dataset & Loader ==========
test_dataset = CIFAR10(
    root="/mnt/Data/cifar/",
    train=True,
    download=True,
    transform=valid_transform,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========== Load Model ==========

model = torch.load(model_path, weights_only=False)


# ========== Collect All Features ==========
features_raw = []
features_encoded = []
labels_all = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Extracting features"):
        images = images.to(device)

        raw = images.view(images.size(0), -1)  # Flatten input image
        encoded = model.client.eval()(images)
        encoded = model.edge.eval()(encoded)  # Get encoded features
        encoded = encoded.view(encoded.size(0), -1)

        features_raw.append(raw.cpu())
        features_encoded.append(encoded.cpu())
        labels_all.append(labels)

# Stack all results
X_raw = torch.cat(features_raw).numpy()
X_encoded = torch.cat(features_encoded).numpy()
y = torch.cat(labels_all).numpy()


# ========== Run t-SNE ==========
def run_tsne(data, name):
    print(f"Running t-SNE on {name}...")
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        metric="cosine",
        verbose=1,
    )
    return tsne.fit_transform(data)


X_raw_tsne = run_tsne(X_raw, "raw inputs")
X_encoded_tsne = run_tsne(X_encoded, "encoded features")


# ========== Plot ==========
import os

def plot_tsne(tsne_data, labels, title, save_path=None):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_data[:, 0], tsne_data[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(10), label="Class")
    plt.grid(True)
    plt.tight_layout()

    # Save if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

plot_tsne(X_raw_tsne, y, "t-SNE of Raw Input Images", save_path="figures/tsne_rawiid.png")
plot_tsne(X_encoded_tsne, y, "t-SNE of Encoded Features", save_path="figures/tsne_encodediid.png")
