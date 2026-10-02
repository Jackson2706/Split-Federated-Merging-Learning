"""Shared-covariance (LDA-style) prototype synthesis.

The diagonal model x = mu_y + sigma_y * eps (eq. 2) ignores feature correlations,
so the upper tier trains on samples that lie off the real feature manifold.
Here every source additionally reports its within-class scatter
S_k = sum_i (x_i - mu_{y_i})(x_i - mu_{y_i})^T (a d x d symmetric statistic, no
sample-level data); the aggregating tier pools them into one covariance Sigma
and synthesizes x = mu_y + L eps with L = chol(Sigma). Diagnostic evidence
(tools/diag/pipeline_gap.py, CIFAR-100 depth-4, held-out train split): a head
trained on diagonal samples reached 23.8% vs 35.1% with pooled-covariance
whitening, against a 36.7% real-feature probe.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch


def within_class_scatter(feats_by_cls: Dict[int, Iterable[torch.Tensor]]) -> Tuple[torch.Tensor, int, int]:
    """Return (scatter d x d fp32, n_samples, n_classes) from per-class feature lists."""
    S, n, k = None, 0, 0
    for tensors in feats_by_cls.values():
        X = torch.stack(list(tensors)).float().flatten(1)
        Xc = X - X.mean(0, keepdim=True)
        s = Xc.T @ Xc
        S = s if S is None else S + s
        n += X.shape[0]
        k += 1
    return S, n, k


def pooled_covariance(stats: Iterable[Tuple[torch.Tensor, int, int]]) -> Optional[torch.Tensor]:
    S, n, k = None, 0, 0
    for s, ni, ki in stats:
        if s is None:
            continue
        S = s.clone() if S is None else S + s.to(S.device)
        n += ni
        k += ki
    if S is None or n - k <= 1:
        return None
    return S / float(n - k)


def scatter_payload_MB(d: int) -> float:
    """Upper triangle of a d x d fp32 symmetric matrix."""
    return d * (d + 1) / 2 * 4 / 2 ** 20


class SharedCovarianceGenerator:
    """Drop-in for the ``generator=`` hook of reliability_weighted_aggregate."""

    def __init__(self, jitter: float = 1e-4):
        self.jitter = jitter
        self.L = None

    def set_covariance(self, cov: Optional[torch.Tensor]) -> None:
        if cov is None:
            self.L = None
            return
        d = cov.shape[0]
        eye = torch.eye(d, device=cov.device, dtype=torch.float32)
        cov = cov.float()
        scale = cov.diagonal().mean().clamp(min=1e-8)
        for j in (self.jitter, 1e-3, 1e-2, 1e-1):
            L, info = torch.linalg.cholesky_ex(cov + j * scale * eye)
            if int(info) == 0:
                self.L = L
                return
        self.L = None

    @torch.no_grad()
    def generate(self, prototypes, distributions_std, num_samples_per_class):
        C = prototypes.shape[0]
        shape = prototypes.shape[1:]
        mu = prototypes.float().reshape(C, -1)
        d = mu.shape[1]
        if self.L is None or self.L.shape[0] != d:
            eps = torch.randn(C, num_samples_per_class, d, device=mu.device)
            x = mu[:, None] + distributions_std.float().reshape(C, 1, d) * eps
        else:
            eps = torch.randn(C * num_samples_per_class, d, device=mu.device)
            x = mu.repeat_interleave(num_samples_per_class, 0) + eps @ self.L.to(mu.device).T
        return x.reshape(C * num_samples_per_class, *shape).to(prototypes.dtype)
