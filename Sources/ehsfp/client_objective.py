"""Client-local objective ablations for classification H-SFP.

These helpers accept only client features and that client's labels. They have
no access to edge/cloud tensors, prototypes, memory, or communication state.
"""

from __future__ import annotations

import torch
from torch import nn


CLIENT_OBJECTIVES = ("ssl", "ssl_supcon", "supervised")


def validate_client_objective(value: str | None) -> str:
    objective = "ssl" if value is None else str(value)
    if objective not in CLIENT_OBJECTIVES:
        raise ValueError(
            f"client_objective must be one of {CLIENT_OBJECTIVES}, got {objective!r}"
        )
    return objective


def local_supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss over one client's feature batch only."""
    if features.ndim != 2:
        raise ValueError(f"client features must be 2D, got {tuple(features.shape)}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("client labels must be 1D and aligned with client features")

    features = nn.functional.normalize(features, dim=1)
    logits = features @ features.T / temperature
    self_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp().masked_fill(self_mask, 0)
    log_prob = logits - (exp_logits.sum(dim=1, keepdim=True) + 1e-8).log()
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.sum() * 0.0
    per_anchor = (log_prob * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
    return -per_anchor[valid].mean()


def compose_client_objective_loss(
    ssl_loss: torch.Tensor,
    objective: str,
    client_features: torch.Tensor | None = None,
    client_labels: torch.Tensor | None = None,
    linear_head: nn.Module | None = None,
    weight: float = 0.1,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Add a client-local auxiliary term while preserving ``ssl`` by identity."""
    objective = validate_client_objective(objective)
    if objective == "ssl":
        return ssl_loss
    if client_features is None or client_labels is None:
        raise ValueError(f"{objective} requires client-local features and labels")
    if objective == "ssl_supcon":
        auxiliary = local_supervised_contrastive_loss(
            client_features, client_labels, temperature=temperature
        )
    else:
        if linear_head is None:
            raise ValueError("supervised requires a client-local linear head")
        auxiliary = nn.functional.cross_entropy(linear_head(client_features), client_labels)
    return ssl_loss + float(weight) * auxiliary
