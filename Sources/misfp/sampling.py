"""Seeded synthesis from class-conditional diagonal mixtures.

For class c: draw component r ~ Categorical(pi_c), then

    z = mu_r + sqrt(max(v_r, eps)) * epsilon,   epsilon ~ N(0, I)

with label c for every component. Components are never averaged before
sampling. Class sampling priors (how many samples per class) are separate from
within-class component weights pi. No renormalization of samples is applied
(the H-SFP raw-space convention).
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Sequence

import torch

from .packets import ClassMixture


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def make_generator(*parts) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(derive_seed(*parts))
    return g


def sample_mixture(cm: ClassMixture, n: int, gen: torch.Generator, var_floor: float = 0.0):
    """Returns (samples [n, d] float64, component index [n])."""
    if n <= 0:
        return torch.empty(0, cm.d, dtype=torch.float64), torch.empty(0, dtype=torch.long)
    comp = torch.multinomial(cm.weights, n, replacement=True, generator=gen)
    std = cm.variances.clamp_min(var_floor).sqrt()
    eps = torch.randn(n, cm.d, generator=gen, dtype=torch.float64)
    return cm.means[comp] + std[comp] * eps, comp


def sample_component(cm: ClassMixture, r: int, n: int, gen: torch.Generator,
                     var_floor: float = 0.0) -> torch.Tensor:
    std = cm.variances[r].clamp_min(var_floor).sqrt()
    eps = torch.randn(n, cm.d, generator=gen, dtype=torch.float64)
    return cm.means[r] + std * eps


def synthesize(classes: Dict[int, ClassMixture], n_per_class: int, gen: torch.Generator,
               var_floor: float = 0.0, feature_shape: Optional[Sequence[int]] = None,
               device="cpu"):
    """Balanced class prior (n_per_class each, as in H-SFP); within-class pi.

    Returns float32 features [N, *feature_shape], labels [N], component idx [N]."""
    feats, labels, comps = [], [], []
    for c in sorted(classes):
        z, r = sample_mixture(classes[c], n_per_class, gen, var_floor)
        feats.append(z)
        comps.append(r)
        labels.append(torch.full((n_per_class,), int(c), dtype=torch.long))
    if not feats:
        return (torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long))
    X = torch.cat(feats).float()
    if feature_shape is not None:
        X = X.reshape(X.shape[0], *feature_shape)
    return X.to(device), torch.cat(labels).to(device), torch.cat(comps)
