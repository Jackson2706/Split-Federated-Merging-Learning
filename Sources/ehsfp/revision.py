"""Revision diagnostics: drift/coverage oracle maths, packet loss, calibration.

Everything here is measurement or an opt-in mechanism with its own RNG; the
oracle helpers never touch the global RNG streams or any training state.
"""
from __future__ import annotations

import random
from typing import Dict, Mapping, Optional

import numpy as np
import torch


class PacketLoss:
    """Assumption 8: each expected packet is dropped independently with prob. p."""

    def __init__(self, p: float, seed: int):
        if not 0.0 <= p < 1.0:
            raise ValueError("packet loss must be in [0, 1)")
        self.p = float(p)
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed) + 15485863)

    def lost(self) -> bool:
        return self.p > 0 and torch.rand(1, generator=self.rng).item() < self.p


def prop2_class_loss(p: float, S: int, S_j: int) -> float:
    """Proposition 2: Pr(class j lost) = p^{S_j} - (S_j/S) p^S (forced retention)."""
    return p ** S_j - (S_j / S) * p ** S


def mask_class_rows(before: Mapping, after: Mapping):
    """Per class present before masking: (class, S, S_j, removed)."""
    S = sum(1 for v in before.values() if v is not None)
    held_before, held_after = {}, set()
    for v in before.values():
        if v is not None:
            for c in v[0]:
                held_before[c] = held_before.get(c, 0) + 1
    for v in after.values():
        if v is not None:
            held_after.update(v[0])
    return [(int(c), S, n, int(c not in held_after)) for c, n in sorted(held_before.items())]


def class_set(outputs: Mapping) -> set:
    out = set()
    for v in outputs.values():
        if v is not None:
            out.update(int(c) for c in v[0])
    return out


def between_source_dispersion(outputs: Mapping) -> Optional[float]:
    """Mean over classes of (1/S_j) sum_l ||mu_l^j - mean_l mu_l^j||^2 (classes with S_j >= 2)."""
    by_cls: Dict[int, list] = {}
    for v in outputs.values():
        if v is not None:
            for c, mu in v[0].items():
                by_cls.setdefault(int(c), []).append(mu.detach().float().flatten().cpu())
    vals = []
    for mus in by_cls.values():
        if len(mus) >= 2:
            M = torch.stack(mus)
            vals.append(float((M - M.mean(0)).pow(2).sum(1).mean()))
    return sum(vals) / len(vals) if vals else None


def _sqrtm_psd(A: torch.Tensor) -> torch.Tensor:
    w, V = torch.linalg.eigh(A)
    return (V * w.clamp(min=0).sqrt()) @ V.T


def bures_sq(A: torch.Tensor, B: torch.Tensor) -> float:
    """B^2(A, B) = tr A + tr B - 2 tr (A^{1/2} B A^{1/2})^{1/2} (float64, symmetric eigh)."""
    A = A.double()
    B = B.double().to(A.device)
    rA = _sqrtm_psd(A)
    cross = rA @ B @ rA
    cross = 0.5 * (cross + cross.T)
    w = torch.linalg.eigvalsh(cross).clamp(min=0).sqrt().sum()
    return float(max(torch.trace(A) + torch.trace(B) - 2 * w, torch.tensor(0.0, dtype=torch.float64)))


def w2_report(actual, reference, class_counts: Mapping[int, float],
              cov_actual: Optional[torch.Tensor] = None,
              cov_reference: Optional[torch.Tensor] = None) -> Optional[dict]:
    """Sum_j pi_j W2^2(Q^j, Q*^j) with its mean and covariance terms.

    ``actual``/``reference``: ({class: mean}, {class: std}). With both shared
    full covariances given, the covariance term is the Bures term computed once
    (pi sums to one); otherwise the diagonal formula ||sigma - sigma*||^2.
    pi_j are the reference class counts renormalized over shared classes.
    """
    ap, ad = actual
    rp, rd = reference
    shared = sorted(set(ap) & set(rp))
    if not shared:
        return None
    w = torch.tensor([float(class_counts.get(c, 0.0)) for c in shared], dtype=torch.float64)
    w = w / w.sum() if w.sum() > 0 else torch.full((len(shared),), 1.0 / len(shared), dtype=torch.float64)
    mean_t = cov_t = 0.0
    full = cov_actual is not None and cov_reference is not None
    for k, c in enumerate(shared):
        dev = rp[c].device
        mean_t += float(w[k]) * float((ap[c].detach().double().to(dev) - rp[c].double()).pow(2).sum())
        if not full:
            cov_t += float(w[k]) * float((ad[c].detach().double().to(dev) - rd[c].double()).pow(2).sum())
    if full:
        cov_t = bures_sq(cov_actual, cov_reference)
    return {"delta_proto": mean_t + cov_t, "mean_term": mean_t, "cov_term": cov_t,
            "cov_type": "full" if full else "diag", "shared_classes": len(shared)}


def ece_nll(logits: torch.Tensor, targets: torch.Tensor, bins: int = 15):
    """Expected calibration error (equal-width bins on max softmax prob) and NLL."""
    logits = logits.double()
    prob = logits.softmax(1)
    conf, pred = prob.max(1)
    acc = (pred == targets).double()
    ece = 0.0
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += float(m.double().mean() * (conf[m].mean() - acc[m].mean()).abs())
    nll = float(torch.nn.functional.cross_entropy(logits, targets))
    return ece, nll


class rng_guard:
    """Save and restore every global RNG stream (Python, NumPy, torch CPU/CUDA)."""

    def __enter__(self):
        self.state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                      torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        return self

    def __exit__(self, *exc):
        py, npy, cpu, cuda = self.state
        random.setstate(py)
        np.random.set_state(npy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        return False


def global_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_global_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class _GenState:
    """Picklable stand-in for a torch.Generator inside a checkpoint."""

    def __init__(self, g: torch.Generator):
        self.state = g.get_state()

    def restore(self) -> torch.Generator:
        g = torch.Generator()
        g.set_state(self.state)
        return g


def swap_generators(objs, to_state: bool):
    """Replace ``obj.rng`` torch.Generators by picklable states (or back)."""
    for o in objs:
        r = getattr(o, "rng", None)
        if to_state and isinstance(r, torch.Generator):
            o.rng = _GenState(r)
        elif not to_state and isinstance(r, _GenState):
            o.rng = r.restore()

