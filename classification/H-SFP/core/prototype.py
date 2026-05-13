"""
Prototype computation and synthetic data generation.

Provides utilities for computing class prototypes (mean) and distributions (std)
from feature representations, and generating synthetic data from them.
"""

import torch
from torch import nn


@torch.no_grad()
def generate_synthetic_data(prototypes, distributions_std, num_samples_per_class):
    """
    Generate synthetic data directly on the device of prototypes.
    Memory-efficient via in-place operations.

    Args:
        prototypes: Tensor of shape [C, *feature_shape] with class means.
        distributions_std: Tensor of shape [C, *feature_shape] with class stds.
        num_samples_per_class: Number of synthetic samples per class.

    Returns:
        Tensor of shape [C * num_samples_per_class, *feature_shape].
    """
    num_classes = prototypes.shape[0]
    shape = prototypes.shape[1:]
    device = prototypes.device

    means = prototypes.unsqueeze(1)
    stds = distributions_std.unsqueeze(1)

    # Create directly on GPU, using matching dtype to save VRAM
    epsilon = torch.randn(
        num_classes, num_samples_per_class, *shape,
        device=device, dtype=prototypes.dtype
    )

    # In-place operations to save memory
    epsilon.mul_(stds).add_(means)

    return epsilon.flatten(0, 1)  # [C*N, ...]


def aggregate_prototypes_and_generate_data(input_outputs, num_samples_per_class, device):
    """
    Aggregate prototypes from multiple sources and generate synthetic data.

    Groups prototypes by label, averages them, then generates synthetic
    samples using the averaged prototypes and distributions.

    Args:
        input_outputs: Dict {source_id: (proto_dict, dist_dict)} or None values.
        num_samples_per_class: Number of synthetic samples per class.
        device: Target torch device.

    Returns:
        Tuple of (features, labels) tensors.
    """
    if not input_outputs:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    # Group by label using dictionary
    merged = {}
    for sid, outputs in input_outputs.items():
        if outputs is None:
            continue
        proto_dict, dist_dict = outputs
        for label, proto in proto_dict.items():
            if label not in merged:
                merged[label] = {'p': [], 'd': []}
            merged[label]['p'].append(proto)
            merged[label]['d'].append(dist_dict[label])

    if not merged:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    final_labels = sorted(merged.keys())
    final_protos = torch.stack(
        [torch.stack(merged[l]['p']).mean(0) for l in final_labels]
    ).to(device)
    # Paper Eq. 5: average variance (σ²), then sqrt to get σ
    final_dists = torch.stack([
        torch.sqrt(torch.stack([d**2 for d in merged[l]['d']]).mean(0))
        for l in final_labels
    ]).to(device)

    # Free dictionary immediately
    del merged

    features = generate_synthetic_data(final_protos, final_dists, num_samples_per_class)
    labels = torch.tensor(final_labels, device=device).repeat_interleave(num_samples_per_class)

    return features, labels


def calculate_prototypes_and_distribution(fx, fy):
    """
    Calculate prototype (mean) and distribution (std) for features of each class.

    Args:
        fx: Features tensor (typically on GPU).
        fy: Labels tensor (can be on CPU).

    Returns:
        Tuple of (prototypes_dict, distributions_std_dict) keyed by class label.
    """
    unique_classes = torch.unique(fy)
    prototypes = {}
    distributions_std = {}

    for cls in unique_classes:
        cls_label = cls.item()
        mask = (fy == cls).to(fx.device)
        class_features = fx[mask]

        if class_features.shape[0] > 0:
            with torch.amp.autocast(device_type='cuda', enabled=(fx.device.type == 'cuda')):
                prototypes[cls_label] = torch.mean(class_features, dim=0)
                distributions_std[cls_label] = torch.std(class_features, dim=0, unbiased=False)
        else:
            print(f"Warning: Class {cls_label} has no samples in the processed data.")

    return prototypes, distributions_std
