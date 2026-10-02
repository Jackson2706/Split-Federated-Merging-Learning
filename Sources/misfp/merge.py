"""Mixture-preserving merging and budget-adaptive compression.

Merge cost (heuristic, not a guarantee of minority-mode preservation):

    W2^2(a, b)      = ||mu_a - mu_b||^2 + ||sqrt(v_a) - sqrt(v_b)||^2
    merge_cost(a,b) = n_a n_b / (n_a + n_b) * W2^2(a, b)

Merging uses exact empirical moment pooling with population variances:

    n  = n_a + n_b
    mu = (n_a mu_a + n_b mu_b) / n
    v  = [n_a (v_a + (mu_a - mu)^2) + n_b (v_b + (mu_b - mu)^2)] / n

so total count, mean and per-dimension second moment of the class are unchanged
by any sequence of merges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .packets import ClassMixture, concat_mixtures


def pool_two(n_a, mu_a, v_a, n_b, mu_b, v_b):
    n = n_a + n_b
    mu = (n_a * mu_a + n_b * mu_b) / n
    v = (n_a * (v_a + (mu_a - mu) ** 2) + n_b * (v_b + (mu_b - mu) ** 2)) / n
    return n, mu, v


def w2_squared(mu_a, v_a, mu_b, v_b) -> float:
    return float(((mu_a - mu_b) ** 2).sum() + ((v_a.sqrt() - v_b.sqrt()) ** 2).sum())


def merge_cost(n_a, mu_a, v_a, n_b, mu_b, v_b) -> float:
    n_a, n_b = float(n_a), float(n_b)
    return n_a * n_b / (n_a + n_b) * w2_squared(mu_a, v_a, mu_b, v_b)


def _pairwise_costs(cm: ClassMixture) -> torch.Tensor:
    """[R, R] merge-cost matrix (upper triangle meaningful)."""
    n = cm.counts
    mu = cm.means
    sd = cm.variances.sqrt()
    dmu = torch.cdist(mu, mu) ** 2
    dsd = torch.cdist(sd, sd) ** 2
    harm = (n[:, None] * n[None, :]) / (n[:, None] + n[None, :])
    return harm * (dmu + dsd)


def best_pair(cm: ClassMixture):
    """Smallest-cost pair (i < j) of a CANONICAL mixture, deterministic ties."""
    if cm.R < 2:
        return None
    C = _pairwise_costs(cm)
    best = None
    for i in range(cm.R):
        for j in range(i + 1, cm.R):
            key = (float(C[i, j]), i, j)
            if best is None or key < best:
                best = key
    return best


def merge_pair(cm: ClassMixture, i: int, j: int) -> ClassMixture:
    n, mu, v = pool_two(cm.counts[i], cm.means[i], cm.variances[i],
                        cm.counts[j], cm.means[j], cm.variances[j])
    keep = [r for r in range(cm.R) if r not in (i, j)]
    k = torch.tensor(keep, dtype=torch.long)
    out = ClassMixture(cm.class_id,
                       torch.cat([cm.counts[k], n.reshape(1)]),
                       torch.cat([cm.means[k], mu[None]]),
                       torch.cat([cm.variances[k], v[None]]))
    return out.canonical()


@dataclass
class MergeEvent:
    class_id: int
    cost: float
    n_a: float
    n_b: float
    bytes_saved: int


@dataclass
class CompressionReport:
    status: str = "ok"  # ok | infeasible
    bytes_before: Optional[int] = None
    bytes_after: Optional[int] = None
    budget_bytes: Optional[int] = None
    per_class_cap: Optional[int] = None
    events: List[MergeEvent] = field(default_factory=list)
    components_before: int = 0
    components_after: int = 0

    def summary(self, classes: Dict[int, ClassMixture]) -> dict:
        multi = [cm for cm in classes.values() if cm.R >= 2]
        minority = [float(cm.weights.min()) for cm in multi]
        Rs = [cm.R for cm in classes.values()]
        return {
            "status": self.status,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "budget_bytes": self.budget_bytes,
            "per_class_cap": self.per_class_cap,
            "merges": len(self.events),
            "merge_cost_total": float(sum(e.cost for e in self.events)),
            "merge_cost_max": float(max((e.cost for e in self.events), default=0.0)),
            "components_before": self.components_before,
            "components_after": self.components_after,
            "classes": len(classes),
            "mean_components_per_class": (sum(Rs) / len(Rs)) if Rs else 0.0,
            "max_components_per_class": max(Rs) if Rs else 0,
            "frac_classes_multimodal": (len(multi) / len(classes)) if classes else 0.0,
            "mean_minority_mass": (sum(minority) / len(minority)) if minority else None,
        }


def greedy_merge_to_cap(cm: ClassMixture, cap: Optional[int], events=None,
                        component_bytes: int = 0) -> ClassMixture:
    cm = cm.canonical()
    if cap is None:
        return cm
    if cap < 1:
        raise ValueError("per-class cap must be >= 1")
    while cm.R > cap:
        cost, i, j = best_pair(cm)
        if events is not None:
            events.append(MergeEvent(cm.class_id, cost, float(cm.counts[i]),
                                     float(cm.counts[j]), component_bytes))
        cm = merge_pair(cm, i, j)
    return cm


def compress_classes(
    classes: Dict[int, ClassMixture],
    per_class_cap: Optional[int] = None,
    budget_bytes: Optional[int] = None,
    nbytes_fn: Optional[Callable[[Dict[int, ClassMixture]], int]] = None,
    component_bytes: int = 0,
):
    """Enforce a per-class component cap, then a total serialized-byte budget.

    Under a budget, repeatedly merge the feasible same-class pair with the
    smallest merge-cost increase per byte saved (ties: class id, then canonical
    pair index) until the packet fits.  At least one component per represented
    class is always kept; if even that does not fit, ``status='infeasible'`` is
    returned with the one-component-per-class packet (classes are never
    silently dropped).
    """
    report = CompressionReport(per_class_cap=per_class_cap, budget_bytes=budget_bytes)
    report.components_before = sum(cm.R for cm in classes.values())
    if nbytes_fn is not None:
        report.bytes_before = nbytes_fn(classes)
    out = {c: greedy_merge_to_cap(cm, per_class_cap, report.events, component_bytes)
           for c, cm in sorted(classes.items())}
    if budget_bytes is not None:
        if nbytes_fn is None or component_bytes <= 0:
            raise ValueError("a byte budget needs nbytes_fn and component_bytes")
        # Cache the best pair per class; only the merged class changes per step.
        cand = {c: best_pair(cm) for c, cm in out.items()}
        while nbytes_fn(out) > budget_bytes:
            choice = None
            for c, bp in cand.items():
                if bp is None:
                    continue
                key = (bp[0] / component_bytes, c, bp[1], bp[2])
                if choice is None or key < choice:
                    choice = key
            if choice is None:
                report.status = "infeasible"
                break
            _, c, i, j = choice
            cm = out[c]
            report.events.append(MergeEvent(c, cand[c][0], float(cm.counts[i]),
                                            float(cm.counts[j]), component_bytes))
            out[c] = merge_pair(cm, i, j)
            cand[c] = best_pair(out[c])
    report.components_after = sum(cm.R for cm in out.values())
    if nbytes_fn is not None:
        report.bytes_after = nbytes_fn(out)
    return out, report


def collapse_baseline_average(per_source: List[ClassMixture]) -> ClassMixture:
    """Reproduce the ORIGINAL H-SFP ``aggregation_mode=average`` rule.

    Unweighted mean of source means and unweighted mean of source variances.
    This ignores support counts and the between-source spread of the means, so
    it is NOT moment preserving. Kept only for the K=1 regression comparison.
    Each source must contribute exactly one component.
    """
    for m in per_source:
        if m.R != 1:
            raise ValueError("baseline_average pooling is defined for one component per source")
    stacked = concat_mixtures(per_source)
    return ClassMixture(stacked.class_id, stacked.counts.sum().reshape(1),
                        stacked.means.mean(0, keepdim=True),
                        stacked.variances.mean(0, keepdim=True))
