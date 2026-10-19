"""
Learnable Reliability-Aware Prototype Aggregation for E-HSFP.

A small neural network that learns to weight prototype contributions
based on metadata features (support count, sigma magnitude, age, etc.).
"""

import torch
import torch.nn as nn
from typing import List, Optional
from ehsfp.memory import PrototypeRecord


class PrototypeReliabilityNetwork(nn.Module):
    """Learns a scalar reliability weight for each prototype.

    Input features per prototype:
        [support_count_norm, sigma_magnitude, age_norm,
         dist_from_center, prev_reliability, source_contribution]

    Output: scalar weight in [0, 1] via sigmoid.
    """

    INPUT_DIM = 6

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, INPUT_DIM] per-prototype metadata features.

        Returns:
            [N] reliability weights in [0, 1].
        """
        return self.net(features).squeeze(-1)


def build_reliability_features(
    records: List[PrototypeRecord],
    class_center: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the input feature vector for the reliability network.

    Args:
        records: list of PrototypeRecord for one class.
        class_center: optional pre-computed class center (mu) for distance.
        device: compute device on which to build the features. Memory records
            remain CPU-backed and are copied only while being consumed here.

    Returns:
        [N, 6] tensor of normalized metadata features.
    """
    if device is None:
        device = class_center.device if class_center is not None else torch.device("cpu")
    device = torch.device(device)

    n = len(records)
    feats = torch.zeros(n, PrototypeReliabilityNetwork.INPUT_DIM, device=device)

    support_counts = torch.tensor(
        [r.support_count for r in records], dtype=torch.float32, device=device,
    )
    sigma_mags = torch.stack([
        r.sigma.to(device=device, dtype=torch.float32).norm() for r in records
    ])
    ages = torch.tensor([r.age for r in records], dtype=torch.float32, device=device)
    prev_rels = torch.tensor(
        [r.reliability for r in records], dtype=torch.float32, device=device,
    )

    # Normalize support count: log(1 + n) / log(1 + max_n)
    max_sc = support_counts.max().clamp(min=1.0)
    feats[:, 0] = torch.log1p(support_counts) / torch.log1p(max_sc)

    # Sigma magnitude: normalized by max
    max_sigma = sigma_mags.max().clamp(min=1e-8)
    feats[:, 1] = sigma_mags / max_sigma

    # Age: normalized by max
    max_age = ages.max().clamp(min=1.0)
    feats[:, 2] = ages / max_age

    # Distance from class center
    if class_center is not None:
        class_center = class_center.to(device=device, dtype=torch.float32)
        dists = torch.stack([
            (r.mu.to(device=device, dtype=torch.float32) - class_center).norm()
            for r in records
        ])
        max_dist = dists.max().clamp(min=1e-8)
        feats[:, 3] = dists / max_dist
    # else: stays 0

    # Previous reliability
    feats[:, 4] = prev_rels

    # Source contribution score: use support_count as proxy
    feats[:, 5] = feats[:, 0]  # same as normalized support count

    return feats


def compute_heuristic_reliability(
    records: List[PrototypeRecord],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute heuristic reliability targets for bootstrap training.

    Higher is better: high support count, low sigma, low age, low distance.

    Returns:
        [N] tensor of target reliability values in [0, 1].
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    n = len(records)
    if n == 0:
        return torch.zeros(0, device=device)

    scores = torch.zeros(n, device=device)
    for i, r in enumerate(records):
        # Higher support count -> higher reliability
        sc_score = min(r.support_count / 100.0, 1.0) if r.support_count > 0 else 0.5
        # Lower sigma magnitude -> higher reliability
        sigma_mag = r.sigma.to(device).norm().item()
        sigma_score = 1.0 / (1.0 + sigma_mag)
        # Lower age -> higher reliability
        age_score = 1.0 / (1.0 + r.age)
        # Combine (equal weights as bootstrap)
        scores[i] = (sc_score + sigma_score + age_score) / 3.0

    return scores
