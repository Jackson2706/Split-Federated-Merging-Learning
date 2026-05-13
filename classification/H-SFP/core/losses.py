"""
Contrastive loss functions for federated learning.

Implements the Supervised Contrastive Loss (SupCon) as specified
in the H-SFP paper (Eq. 1), where positive pairs are samples
sharing the same class label.
"""

import torch
from torch import nn


def supervised_contrastive_loss(features, labels, temperature=0.5):
    """
    Supervised Contrastive Loss (SupCon) — Paper Eq. 1.

    L = Σ_i (-1/|P(i)|) Σ_{p∈P(i)} log[exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ)]

    where P(i) are positive samples sharing the same class label as i,
    and A(i) are all other samples in the batch.

    Args:
        features: [B, D] feature vectors (will be L2-normalized).
        labels: [B] integer class labels.
        temperature: Scalar temperature τ (default: 0.5).

    Returns:
        Scalar loss tensor.
    """
    features = nn.functional.normalize(features, dim=1)
    batch_size = features.shape[0]
    device = features.device

    # Similarity matrix [B, B]
    sim_matrix = torch.matmul(features, features.T) / temperature

    # Mask for positive pairs (same class, excluding self)
    labels_col = labels.view(-1, 1)
    positive_mask = torch.eq(labels_col, labels_col.T).float()
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = positive_mask - self_mask  # Remove self-similarity

    # Numerical stability: subtract max logit
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # Denominator: sum over all non-self entries
    exp_logits = torch.exp(logits) * (1 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Mean log-prob over positives
    num_positives = positive_mask.sum(dim=1)
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)

    # Only include anchors that have at least one positive
    valid = (num_positives > 0).float()
    loss = -(valid * mean_log_prob).sum() / (valid.sum() + 1e-8)

    return loss
