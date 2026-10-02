"""Local class-conditional mixture construction.

Minimal, transparent estimator: deterministic seeded k-means (k-means++ init,
Lloyd iterations, first-index tie breaking) followed by EMPIRICAL within-cluster
population moments.  Adaptive K is a held-out NLL heuristic, not an optimal
selector:

  1. split the class's local TRAINING features into fit / validation parts
     (seeded, disjoint; test data is never seen here);
  2. fit K = 1..cap on the fit part;
  3. score(K) = NLL_val(K)/d + penalty_per_component * (K - 1);
  4. take argmin, but keep K = 1 unless it improves score(1) by at least
     ``min_improvement`` nats/dim;
  5. refit the chosen K on all local training features.

Insufficient support or validation data -> K = 1 with a recorded fallback.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch

from .merge import pool_two
from .packets import ClassMixture

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class FitConfig:
    k_mode: str = "fixed"  # fixed | adaptive
    k: int = 1  # fixed K, or the cap for adaptive
    min_component_support: int = 3
    kmeans_iters: int = 50
    kmeans_n_init: int = 3
    reservoir_size: Optional[int] = 4096
    adaptive_val_fraction: float = 0.3
    adaptive_min_val: int = 4
    adaptive_min_improvement: float = 0.02
    adaptive_penalty_per_component: float = 0.0
    nll_rel_var_floor: float = 1e-2
    nll_abs_var_floor: float = 1e-6
    chunk_size: int = 4096

    def __post_init__(self):
        if self.k_mode not in ("fixed", "adaptive"):
            raise ValueError(f"k_mode must be fixed|adaptive, got {self.k_mode!r}")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.min_component_support < 1:
            raise ValueError("min_component_support must be >= 1")


def _sqdist(X, C):
    d = (X * X).sum(1, keepdim=True) - 2.0 * X @ C.T + (C * C).sum(1)[None, :]
    return d.clamp_min_(0.0)


def kmeans(X: torch.Tensor, k: int, gen: torch.Generator, iters: int = 50, n_init: int = 3):
    """Deterministic (given ``gen`` state) k-means on float64 CPU data.

    Returns (centers [k, d], labels [n], inertia).  ``k`` must not exceed the
    number of distinct rows. Empty clusters are re-seeded at the point farthest
    from its assigned center (first index on ties)."""
    n = X.shape[0]
    if k == 1:
        c = X.mean(0, keepdim=True)
        return c, torch.zeros(n, dtype=torch.long), float(_sqdist(X, c).sum())
    best = None
    for _ in range(max(1, n_init)):
        first = int(torch.randint(n, (1,), generator=gen))
        centers = [X[first]]
        d2 = _sqdist(X, X[first:first + 1]).squeeze(1)
        for _j in range(1, k):
            total = float(d2.sum())
            if total <= 0:
                break
            idx = int(torch.multinomial(d2 / total, 1, generator=gen))
            centers.append(X[idx])
            d2 = torch.minimum(d2, _sqdist(X, X[idx:idx + 1]).squeeze(1))
        C = torch.stack(centers)
        if C.shape[0] < k:
            return None  # degenerate (caller limits k to distinct rows)
        labels = None
        for _it in range(iters):
            D = _sqdist(X, C)
            new = D.argmin(1)
            for j in range(k):  # empty-cluster repair
                if not (new == j).any():
                    far = int(D.gather(1, new[:, None]).squeeze(1).argmax())
                    C[j] = X[far]
                    D = _sqdist(X, C)
                    new = D.argmin(1)
            if labels is not None and torch.equal(new, labels):
                break
            labels = new
            for j in range(k):
                C[j] = X[labels == j].mean(0)
        D = _sqdist(X, C)
        labels = D.argmin(1)
        inertia = float(D.gather(1, labels[:, None]).sum())
        if best is None or inertia < best[2]:
            best = (C.clone(), labels.clone(), inertia)
    return best


def stream_moments(X: torch.Tensor, centers: torch.Tensor, chunk: int = 4096):
    """Assign every row to its nearest center and accumulate exact population
    moments chunk-by-chunk (Chan/Welford pooling: no E[x^2]-mu^2 cancellation)."""
    k, d = centers.shape
    n = torch.zeros(k, dtype=torch.float64)
    mu = torch.zeros(k, d, dtype=torch.float64)
    v = torch.zeros(k, d, dtype=torch.float64)
    for s in range(0, X.shape[0], chunk):
        xb = X[s:s + chunk]
        lab = _sqdist(xb, centers).argmin(1)
        for j in range(k):
            m = lab == j
            nb = int(m.sum())
            if nb == 0:
                continue
            xj = xb[m]
            mb = xj.mean(0)
            vb = ((xj - mb) ** 2).mean(0)
            if n[j] == 0:
                n[j], mu[j], v[j] = float(nb), mb, vb
            else:
                n[j], mu[j], v[j] = pool_two(n[j], mu[j], v[j],
                                             torch.tensor(float(nb), dtype=torch.float64), mb, vb)
    return n, mu, v


def mixture_log_prob(X: torch.Tensor, cm: ClassMixture, var_floor) -> torch.Tensor:
    """Per-row log density under the diagonal mixture with floored variances."""
    v = torch.maximum(cm.variances, torch.as_tensor(var_floor, dtype=torch.float64))
    logw = torch.log(cm.weights)
    out = []
    for r in range(cm.R):
        diff2 = (X - cm.means[r]) ** 2
        lp = -0.5 * (LOG_2PI + torch.log(v[r]) + diff2 / v[r]).sum(1)
        out.append(lp + logw[r])
    return torch.logsumexp(torch.stack(out, 1), 1)


def nll_var_floor(X: torch.Tensor, cfg: FitConfig) -> float:
    """Scalar floor relative to the class's mean per-dim variance (fit data only)."""
    base = float(X.var(0, unbiased=False).mean()) if X.shape[0] > 1 else 0.0
    return max(cfg.nll_abs_var_floor, cfg.nll_rel_var_floor * base)


def _fit_fixed(X, k, cfg: FitConfig, gen):
    """Fit k components, reducing k until every component has min support.

    Returns (ClassMixture, k_used, mode) where mode is 'exact' or 'reservoir'."""
    n = X.shape[0]
    n_unique = int(torch.unique(X, dim=0).shape[0])
    k = max(1, min(k, n // cfg.min_component_support, n_unique))
    fitX = X
    mode = "exact"
    if cfg.reservoir_size is not None and n > cfg.reservoir_size:
        perm = torch.randperm(n, generator=gen)[: cfg.reservoir_size]
        fitX = X[perm.sort().values]
        mode = "reservoir"
    while True:
        res = kmeans(fitX, k, gen, cfg.kmeans_iters, cfg.kmeans_n_init) if k > 1 else None
        centers = res[0] if res is not None else X.mean(0, keepdim=True)
        if k > 1 and res is None:
            k -= 1
            continue
        # Exact moments: stream ALL eligible features through the chosen centers.
        cnt, mu, v = stream_moments(X, centers, cfg.chunk_size)
        keep = cnt > 0
        if k == 1 or (cnt[keep] >= cfg.min_component_support).all() and keep.sum() == k:
            cnt, mu, v = cnt[keep], mu[keep], v[keep]
            return ClassMixture(-1, cnt, mu, v), int(cnt.shape[0]), mode
        k -= 1


def fit_class_mixture(X: torch.Tensor, class_id: int, cfg: FitConfig, gen: torch.Generator):
    """Fit one class's mixture from its local training features.

    Returns (ClassMixture | None, info dict). None means the class is absent
    (no finite features)."""
    t0 = time.perf_counter()
    X = torch.as_tensor(X).detach().to("cpu", torch.float64).reshape(X.shape[0], -1)
    finite = torch.isfinite(X).all(1)
    dropped = int((~finite).sum())
    X = X[finite]
    n = X.shape[0]
    info = {"class_id": int(class_id), "n": int(n), "dropped_nonfinite": dropped,
            "k_mode": cfg.k_mode, "k_requested": cfg.k, "fallback": None,
            "nll_val": None, "fit_mode": "exact"}
    if n == 0:
        info.update(k_selected=0, fallback="absent", fit_time_s=time.perf_counter() - t0)
        return None, info

    if cfg.k_mode == "fixed":
        cm, k_used, mode = _fit_fixed(X, cfg.k, cfg, gen)
        if k_used < cfg.k:
            info["fallback"] = "insufficient_support"
    else:
        n_val = max(cfg.adaptive_min_val, int(round(cfg.adaptive_val_fraction * n)))
        n_fit = n - n_val
        if cfg.k < 2 or n_fit < 2 * cfg.min_component_support:
            cm, k_used, mode = _fit_fixed(X, 1, cfg, gen)
            info["fallback"] = "insufficient_validation_support" if cfg.k >= 2 else None
        else:
            perm = torch.randperm(n, generator=gen)
            Xv, Xf = X[perm[:n_val]], X[perm[n_val:]]
            floor = nll_var_floor(Xf, cfg)
            d = X.shape[1]
            scores = {}
            seen = set()
            for k in range(1, cfg.k + 1):
                cand, k_used, _ = _fit_fixed(Xf, k, cfg, gen)
                if k_used in seen:  # support limit reached; larger k repeats
                    break
                seen.add(k_used)
                nll = float(-mixture_log_prob(Xv, cand, floor).mean()) / d
                scores[k_used] = nll + cfg.adaptive_penalty_per_component * (k_used - 1)
            best_k = min(scores, key=lambda k: (scores[k], k))
            if scores[1] - scores[best_k] < cfg.adaptive_min_improvement:
                best_k = 1
            info["nll_val"] = {int(k): float(s) for k, s in scores.items()}
            cm, k_used, mode = _fit_fixed(X, best_k, cfg, gen)
            if k_used < best_k:
                info["fallback"] = "refit_support_reduced"
    cm = ClassMixture(int(class_id), cm.counts, cm.means, cm.variances).canonical()
    info.update(k_selected=cm.R, fit_mode=mode, fit_time_s=time.perf_counter() - t0)
    return cm, info
