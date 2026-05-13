"""
Self-Supervised Learning (SSL) utilities.

Provides augmentation pipelines and contrastive loss functions
used in the SSL phases of hierarchical split-federated learning.
"""

import torch
from torch import nn
import kornia.augmentation as K
import kornia.geometry.transform as K_T


def build_client_ssl_transforms(size=(32, 32)):
    """
    Build SSL augmentation pipeline for client-side (raw image space).

    Args:
        size: Target spatial size (H, W) for the transforms.

    Returns:
        nn.Sequential augmentation pipeline.
    """
    return nn.Sequential(
        K_T.Resize(size),
        K.RandomResizedCrop(size=size, scale=(0.5, 1.0)),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        K.RandomGrayscale(p=0.2)
    )


def build_edge_ssl_transforms(size=(8, 8)):
    """
    Build SSL augmentation pipeline for edge-side (feature map space).

    Args:
        size: Target spatial size (H, W) for the transforms.

    Returns:
        nn.Sequential augmentation pipeline.
    """
    return nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5)
    )


def info_nce_loss_4d(z1, z2, temperature=0.5):
    """
    Compute InfoNCE contrastive loss for 4D feature map outputs.

    Applies Global Average Pooling before computing cosine similarity.

    Args:
        z1, z2: Feature maps of shape [B, C, H, W].
        temperature: Softmax temperature for similarity scaling.

    Returns:
        Scalar loss tensor.
    """
    z1 = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), start_dim=1)
    z2 = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), start_dim=1)

    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    sim_matrix = torch.matmul(z1, z2.mT) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.mT, labels)

    return (loss_a + loss_b) / 2


def info_nce_loss_2d(z1, z2, temperature=0.5):
    """
    Compute InfoNCE contrastive loss for 2D feature vector inputs.

    Args:
        z1, z2: Feature vectors of shape [B, D].
        temperature: Softmax temperature for similarity scaling.

    Returns:
        Scalar loss tensor.
    """
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)

    sim_matrix = torch.matmul(z1, z2.mT) / temperature
    labels = torch.arange(z1.shape[0]).to(z1.device)
    loss_a = nn.CrossEntropyLoss()(sim_matrix, labels)
    loss_b = nn.CrossEntropyLoss()(sim_matrix.mT, labels)

    return (loss_a + loss_b) / 2
