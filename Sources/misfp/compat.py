"""Representation-compatibility diagnostics.

Separated client clusters can reflect genuine data diversity OR different
clients' encoders placing the same data at different coordinates (in legacy
H-SFP each client trains a private encoder copy between t1 aggregations).
These diagnostics are computed from packets only (no extra data access).
"""

from __future__ import annotations

from typing import Dict, List

import torch

from .packets import ClassMixture, MixturePacket


def between_source_dispersion(packets: List[MixturePacket]) -> Dict[str, float]:
    """For classes seen by >= 2 sources: ratio of the between-source variance
    of the (collapsed) class means to the mean within-source variance, summed
    over dimensions. Large ratios at equal data distributions (IID) point to
    coordinate mismatch rather than data diversity."""
    per_class: Dict[int, List[ClassMixture]] = {}
    for p in packets:
        for c, cm in p.classes.items():
            per_class.setdefault(c, []).append(cm.collapse())
    ratios = []
    for c, cms in per_class.items():
        if len(cms) < 2:
            continue
        n = torch.cat([m.counts for m in cms])
        mu = torch.cat([m.means for m in cms])
        v = torch.cat([m.variances for m in cms])
        w = n / n.sum()
        gmu = (w[:, None] * mu).sum(0)
        between = float((w[:, None] * (mu - gmu) ** 2).sum())
        within = float((w[:, None] * v).sum())
        if within > 0:
            ratios.append(between / within)
    if not ratios:
        return {"classes": 0, "mean_ratio": None, "median_ratio": None}
    t = torch.tensor(ratios)
    return {"classes": len(ratios), "mean_ratio": float(t.mean()),
            "median_ratio": float(t.median())}
