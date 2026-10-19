"""Episodic stress mechanisms and oracle diagnostics for the E-HSFP campaign.

Implements the parts of the manuscript's episodic update protocol that the
training loop previously only *labelled* but did not *simulate*:

* ``StalePacketQueue`` -- Definition 3 (stale prototype updates). A client
  packet generated at episode ``s`` is delivered to its edge at episode
  ``t = s + d`` with ``d ~ Uniform{0, ..., tau_max}``; the edge trains on the
  packets that actually arrive at ``t``. This delays the *statistics
  themselves*; it is not a change of age labels.
* ``PartialEdgeExecution`` -- serverless partial execution: with probability
  ``q`` an edge function is terminated after a fraction ``f`` of its local
  SSL minibatches.
* ``gaussian_w2_drift`` -- Definition 6 / eq. (5): squared 2-Wasserstein
  distance between the diagonal Gaussian an upper tier actually synthesizes
  from and the reference induced by complete, fresh class statistics.
  This is an ORACLE diagnostic: the reference uses packets the tier did not
  receive (dropped / delayed ones). It is logged, never fed back to training.

Every mechanism draws from its own seeded ``torch.Generator`` so enabling one
never perturbs the global RNG streams used by training, and every mechanism is
an exact no-op at its neutral setting (tau_max=0, q=0).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

import torch

ProtoDicts = Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]


@dataclass
class StalePacket:
    key: str                 # unique packet key "<cid>@<generation round>"
    cid: int
    generated_round: int
    deliver_round: int
    outputs: ProtoDicts
    support: Dict[int, int]

    def age(self, current_round: int) -> int:
        return current_round - self.generated_round


class StalePacketQueue:
    """Bounded-staleness delivery of client->edge prototype packets."""

    def __init__(self, tau_max: int, seed: int):
        if tau_max < 0:
            raise ValueError("tau_max must be >= 0")
        self.tau_max = int(tau_max)
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed) + 7919)
        self._pending: List[StalePacket] = []
        self.enqueued = 0
        self.delivered = 0

    @property
    def active(self) -> bool:
        return self.tau_max > 0

    def enqueue(self, cid: int, generated_round: int, outputs: ProtoDicts,
                support: Mapping[int, int], extra_delay: int = 0) -> int:
        delay = 0
        if self.tau_max > 0:
            delay = int(torch.randint(0, self.tau_max + 1, (1,), generator=self.rng).item())
        delay = min(self.tau_max, delay + int(extra_delay)) if self.tau_max > 0 else int(extra_delay)
        self._pending.append(StalePacket(
            key=f"{int(cid)}@{int(generated_round)}",
            cid=int(cid),
            generated_round=int(generated_round),
            deliver_round=int(generated_round) + delay,
            outputs=outputs,
            support=dict(support),
        ))
        self.enqueued += 1
        return delay

    def deliver(self, current_round: int) -> List[StalePacket]:
        ready = [p for p in self._pending if p.deliver_round <= current_round]
        self._pending = [p for p in self._pending if p.deliver_round > current_round]
        self.delivered += len(ready)
        return ready

    def __len__(self) -> int:
        return len(self._pending)


class PartialEdgeExecution:
    """Serverless partial edge execution (function killed mid-episode)."""

    def __init__(self, probability: float, fraction: float, seed: int):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.probability = float(probability)
        self.fraction = float(fraction)
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed) + 104729)
        self.events = 0

    def batch_budget(self, total_batches: int) -> int:
        """Number of SSL minibatches this edge invocation may execute."""
        if self.probability <= 0.0 or total_batches <= 0:
            return total_batches
        if torch.rand(1, generator=self.rng).item() < self.probability:
            self.events += 1
            return max(1, int(self.fraction * total_batches))
        return total_batches


def total_variance_aggregate(
    outputs: Mapping[Hashable, Optional[ProtoDicts]],
    support_map: Optional[Mapping[Hashable, Mapping[int, int]]] = None,
) -> ProtoDicts:
    """Support-weighted class statistics by the law of total variance (eq. 10-11)."""
    grouped: Dict[int, Dict[str, list]] = {}
    for sid, out in outputs.items():
        if out is None:
            continue
        protos, stds = out
        for cls, mu in protos.items():
            g = grouped.setdefault(cls, {"mu": [], "sd": [], "n": []})
            g["mu"].append(mu.detach().float())
            g["sd"].append(stds[cls].detach().float().to(mu.device))
            n = 0
            if support_map is not None:
                n = int(support_map.get(sid, {}).get(cls, 0))
            g["n"].append(n)
    ref_p, ref_d = {}, {}
    for cls, g in grouped.items():
        dev = g["mu"][0].device
        mus = torch.stack([m.to(dev) for m in g["mu"]])
        sds = torch.stack([s.to(dev) for s in g["sd"]])
        w = torch.tensor(g["n"], dtype=torch.float32, device=dev)
        if w.sum() <= 0:
            w = torch.ones_like(w)
        w = (w / w.sum()).view(-1, *([1] * (mus.dim() - 1)))
        mu = (w * mus).sum(0)
        var = (w * (sds ** 2 + (mus - mu.unsqueeze(0)) ** 2)).sum(0)
        ref_p[cls] = mu
        ref_d[cls] = torch.sqrt(var + 1e-8)
    return ref_p, ref_d


def gaussian_w2_drift(
    used: ProtoDicts,
    reference: ProtoDicts,
    class_weights: Optional[Mapping[int, float]] = None,
) -> Optional[Dict[str, float]]:
    """Eq. (5): per-class ||mu-mu*||^2 + ||sigma-sigma*||^2, aggregated with pi_t.

    Classes present in only one of the two distributions are counted
    (``classes_missing``) but cannot contribute a distance. Returns None when no
    class is shared.
    """
    used_p, used_d = used
    ref_p, ref_d = reference
    shared = sorted(set(used_p) & set(ref_p))
    if not shared:
        return None
    per_class = {}
    for cls in shared:
        dev = ref_p[cls].device
        dm = (used_p[cls].detach().float().to(dev) - ref_p[cls]).pow(2).sum()
        ds = (used_d[cls].detach().float().to(dev) - ref_d[cls]).pow(2).sum()
        per_class[cls] = float((dm + ds).item())
    if class_weights:
        w = torch.tensor([float(class_weights.get(c, 0.0)) for c in shared])
        if w.sum() <= 0:
            w = torch.ones(len(shared))
    else:
        w = torch.ones(len(shared))
    w = w / w.sum()
    vals = torch.tensor([per_class[c] for c in shared])
    return {
        "drift_w2": float((w * vals).sum().item()),
        "drift_w2_max": float(vals.max().item()),
        "classes_shared": len(shared),
        "classes_missing": len(set(ref_p) - set(used_p)),
    }


class CsvAppender:
    """Append-only CSV with a fixed header (flushed every row)."""

    def __init__(self, path: str, columns: List[str]):
        self.path = path
        self.columns = list(columns)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self.columns)

    def append(self, row: Mapping[str, object]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.columns, extrasaction="ignore").writerow(
                {c: row.get(c, "") for c in self.columns}
            )


DIAGNOSTIC_COLUMNS = [
    "round", "tier", "node", "drift_w2", "drift_w2_max", "classes_shared",
    "classes_missing", "fresh_sources", "used_sources", "stale_packets",
    "mean_packet_age", "max_packet_age", "partial_execution",
]

RELIABILITY_COLUMNS = [
    "round", "tier", "node", "class_id", "source", "support_feat", "sigma_feat",
    "age_feat", "distance_feat", "age", "weight", "normalized_weight",
]
