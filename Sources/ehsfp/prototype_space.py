"""Prototype-space ablations shared by H-SFP tiers."""

from typing import Dict, Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F


PROTOTYPE_SPACES = ("raw", "centered_cosine", "whitened_cosine")

# Prototype spaces that use the angular (cosine) cloud head + a centered frame.
COSINE_SPACES = ("centered_cosine", "whitened_cosine")


def validate_prototype_space(value: Optional[str]) -> str:
    """Resolve the opt-in prototype space without changing legacy configs."""
    value = "raw" if value is None else str(value)
    if value not in PROTOTYPE_SPACES:
        raise ValueError(
            f"prototype_space must be one of {PROTOTYPE_SPACES}, got {value!r}"
        )
    return value


@torch.no_grad()
def derive_whitening_transform(
    input_outputs: Mapping,
    mean: torch.Tensor,
    support_map: Optional[Mapping] = None,
    shrinkage: float = 0.1,
    eps: float = 1e-3,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Derive a symmetric ZCA whitening matrix from the BETWEEN-class scatter.

    The class prototype means are already transmitted; their covariance around
    the global ``mean`` (weighted by existing support counts) captures the
    dominant, near-collinear directions that collapse raw prototypes. ZCA
    whitening ``W = V diag((lambda+eps)^-1/2) V^T`` decorrelates and equalizes
    those directions. Shrinkage toward the identity keeps small/degenerate
    eigenvalues from exploding. No feature tensor is added to the payload: this
    is computed at the reconstructing tier from already-received class means.
    """
    target_device = torch.device(device) if device is not None else mean.device
    mu = mean.detach().to(target_device).reshape(-1)
    dim = mu.numel()
    scatter = torch.zeros(dim, dim, device=target_device, dtype=torch.float32)
    total = 0.0
    for source_id, outputs in input_outputs.items():
        if outputs is None:
            continue
        prototypes, _ = outputs
        for class_id, prototype in prototypes.items():
            count = 1
            if support_map is not None:
                count = int(support_map.get(source_id, {}).get(class_id, 0)) or 1
            centered = prototype.detach().to(target_device).reshape(-1).float() - mu.float()
            scatter = scatter + count * torch.outer(centered, centered)
            total += count
    if total <= 0:
        return torch.eye(dim, device=target_device, dtype=torch.float32)
    cov = scatter / total
    # Shrink toward a scaled identity for numerical stability.
    trace_mean = torch.trace(cov) / dim
    cov = (1.0 - shrinkage) * cov + shrinkage * trace_mean * torch.eye(
        dim, device=target_device, dtype=torch.float32
    )
    evals, evecs = torch.linalg.eigh(cov)
    inv_sqrt = torch.rsqrt(torch.clamp(evals, min=eps))
    whitener = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.t()
    return whitener.detach()


def whiten_features(
    features: torch.Tensor,
    mean: Optional[torch.Tensor],
    whitener: Optional[torch.Tensor],
) -> torch.Tensor:
    """Center then whiten batched real/synthetic features: ``(x - mu) W``.

    Preserves the original per-sample feature shape (e.g. ``[B, C, 1, 1]``) by
    flattening to ``[B, D]`` for the matmul and reshaping back.
    """
    out = center_features(features, mean)
    if whitener is None:
        return out
    orig_shape = out.shape
    flat = out.reshape(orig_shape[0], -1)
    w = whitener.to(device=flat.device, dtype=flat.dtype)
    return (flat @ w).reshape(orig_shape)


@torch.no_grad()
def whiten_source_outputs(
    input_outputs: Mapping,
    mean: torch.Tensor,
    whitener: torch.Tensor,
) -> Dict:
    """Express prototypes in the centered+whitened space.

    Class means are centered and whitened. Diagonal stds are transformed to the
    diagonal of ``W diag(std^2) W^T`` so the per-class Gaussian remains a valid
    (diagonal-approximated) sampler in the whitened frame.
    """
    whitened = {}
    for source_id, outputs in input_outputs.items():
        if outputs is None:
            whitened[source_id] = None
            continue
        prototypes, stds = outputs
        w = whitener.to(device=next(iter(prototypes.values())).device, dtype=torch.float32)
        mu = mean.to(w.device, torch.float32).reshape(-1)
        new_protos = {}
        for class_id, prototype in prototypes.items():
            orig_shape = prototype.shape
            centered = prototype.detach().to(w.device).float().reshape(-1) - mu
            new_protos[class_id] = (centered @ w).reshape(orig_shape).to(prototype.dtype)
        new_stds = stds
        if isinstance(stds, Mapping):
            new_stds = {}
            for class_id, std in stds.items():
                orig_shape = std.shape
                var = (std.detach().to(w.device).float().reshape(-1) ** 2)
                # diag(W diag(var) W^T) = sum_j W_ij^2 var_j  (per output dim i)
                new_var = (w.pow(2) @ var).clamp_min(0.0)
                new_stds[class_id] = new_var.sqrt().reshape(orig_shape).to(std.dtype)
        whitened[source_id] = (new_protos, new_stds)
    return whitened


@torch.no_grad()
def derive_global_feature_mean(
    input_outputs: Mapping,
    support_map: Optional[Mapping] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Derive the sample-weighted global mean from existing prototype stats.

    Each transmitted class prototype is already a local class mean. Its support
    count is existing scalar metadata used by reliability; no feature tensor is
    added to the communication payload. Missing/zero support falls back to one.
    """
    weighted_sum = None
    total = 0.0
    for source_id, outputs in input_outputs.items():
        if outputs is None:
            continue
        prototypes, _ = outputs
        for class_id, prototype in prototypes.items():
            count = 1
            if support_map is not None:
                count = int(support_map.get(source_id, {}).get(class_id, 0)) or 1
            target_device = torch.device(device) if device is not None else prototype.device
            term = prototype.detach().to(target_device) * count
            weighted_sum = term.clone() if weighted_sum is None else weighted_sum + term
            total += count
    if weighted_sum is None:
        raise ValueError("cannot derive a global feature mean from empty prototype stats")
    return weighted_sum / total


def center_features(features: torch.Tensor, mean: Optional[torch.Tensor]) -> torch.Tensor:
    """Subtract a boundary mean from batched real or synthetic features."""
    if mean is None:
        return features
    return features - mean.to(device=features.device, dtype=features.dtype).unsqueeze(0)


def center_source_outputs(input_outputs: Mapping, mean: torch.Tensor) -> Dict:
    """Return prototype dictionaries expressed in the centered space."""
    centered = {}
    for source_id, outputs in input_outputs.items():
        if outputs is None:
            centered[source_id] = None
            continue
        prototypes, stds = outputs
        centered[source_id] = (
            {
                class_id: prototype - mean.to(prototype.device, prototype.dtype)
                for class_id, prototype in prototypes.items()
            },
            stds,
        )
    return centered


@torch.no_grad()
def recenter_memory(memory, old_mean: Optional[torch.Tensor], new_mean: torch.Tensor) -> None:
    """Move historical centered records into a newly selected centered frame."""
    if memory is None or old_mean is None:
        return
    shift = old_mean.detach().cpu() - new_mean.detach().cpu()
    for record in memory.get_recent():
        record.mu = record.mu + shift.to(record.mu.dtype)


class CosineClassifier(nn.Module):
    """Angular classifier: ``logits = scale * cosine(weight, feature)``."""

    def __init__(self, in_features: int, num_classes: int, initial_scale: float = 16.0):
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(features.shape[0], -1)
        return self.scale * F.linear(
            F.normalize(features, dim=1), F.normalize(self.weight, dim=1)
        )
