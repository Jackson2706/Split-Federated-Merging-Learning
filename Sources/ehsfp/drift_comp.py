"""C1 drift-compensated replay and C2 bias-variance-optimal replay weight.

Both use only statistics a learner already receives: the class means it has
stored in its episodic memory. ``DriftState`` keeps, per source l and class j,
the last ``window`` received class means with their episode stamps.

C1 (``rate``): for each source/class with >= 2 reports use the last two,
r = (mu_s2 - mu_s1) / (s2 - s1); the drift of a stale record is the
kernel-weighted mean of these rate samples times its age.
C1 (``pair``): kernel-weighted mean of the displacements of sources/classes that
reported at both the record's stamp s and a later stamp (scaled to age t - s);
falls back to ``rate`` when there is none.
C2: alpha*_j = b_j^2 / (b_j^2 + v_j / n_j), clipped to [alpha_min, 1].
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch


class DriftState:
    """Last ``window`` (stamp, class mean) reports per (source, class)."""

    def __init__(self, window: int = 5):
        self.window = max(int(window), 2)
        self.hist: Dict[Tuple[str, int], List[Tuple[int, torch.Tensor]]] = {}
        self.version = 0
        self._cache: dict = {}

    def __getstate__(self):  # device caches are never checkpointed
        d = dict(self.__dict__)
        d["_cache"] = {}
        return d

    def observe(self, source_id: str, class_id: int, stamp: int, mu: torch.Tensor) -> None:
        key = (str(source_id), int(class_id))
        x = mu.detach().float().flatten().cpu()
        h = self.hist.setdefault(key, [])
        for i, (s, _) in enumerate(h):
            if s == stamp:
                h[i] = (stamp, x)
                break
        else:
            h.append((int(stamp), x))
            h.sort(key=lambda e: e[0])
            del h[:-self.window]
        self.version += 1

    def nbytes(self) -> int:
        return sum(m.numel() * 4 + 8 for h in self.hist.values() for _, m in h)

    # -- vectorised views (cached per version / device) -----------------------
    def _rate_view(self, device):
        k = ("rate", self.version, str(device))
        if k not in self._cache:
            self._cache = {k: self._build_rate(device)}
        return self._cache[k]

    def _build_rate(self, device):
        srcs, R, A = [], [], []
        for (src, _c), h in self.hist.items():
            if len(h) >= 2:
                (s1, m1), (s2, m2) = h[-2], h[-1]
                R.append((m2 - m1) / float(s2 - s1))
                A.append(m2)
                srcs.append(src)
        if not R:
            return None
        return torch.stack(R).to(device), torch.stack(A).to(device), srcs

    def _pair_view(self, stamp, now, device):
        k = ("pair", self.version, int(stamp), int(now), str(device))
        if k not in self._cache:
            D, A, srcs = [], [], []
            for (src, _c), h in self.hist.items():
                at = [m for s, m in h if s == stamp]
                later = [(s, m) for s, m in h if s > stamp]
                if at and later:
                    s_new, m_new = later[-1]
                    D.append((m_new - at[0]) * (float(now - stamp) / float(s_new - stamp)))
                    A.append(at[0])
                    srcs.append(src)
            view = (torch.stack(D).to(device), torch.stack(A).to(device), srcs) if D else None
            self._cache[k] = view
        return self._cache[k]


def _kernel_mean(mu_rec, src_rec, S, anchor, src_s, h, beta, min_w):
    """Kernel-weighted mean and trace-variance of samples S for each record.

    w = exp(-||mu_rec - anchor||^2 / (2 h^2)) * (beta if same source else 1).
    Returns (mean [R,d], var [R], ok [R] bool).
    """
    S, anchor, mu = S.double(), anchor.double(), mu_rec.double()
    w = torch.exp(-torch.cdist(mu, anchor).pow(2) / (2.0 * h * h))
    same = torch.tensor([[1.0 if a == b else 0.0 for b in src_s] for a in src_rec],
                        dtype=torch.float64, device=w.device)
    w = w * (1.0 + (beta - 1.0) * same)
    tot = w.sum(1)
    ok = tot >= min_w
    denom = tot.clamp(min=1e-300).unsqueeze(1)
    mean = (w @ S) / denom
    var = ((w @ S.pow(2).sum(1, keepdim=True)) / denom).squeeze(1) - mean.pow(2).sum(1)
    return mean.float(), var.clamp(min=0).float(), ok


class ReplayAdjuster:
    """Configuration and arithmetic of C1 / C2 (state lives in the memory)."""

    def __init__(self, drift_comp: str = "off", alpha_mode: str = "fixed", src_bonus: float = 2.0,
                 min_weight: float = 1e-6, alpha_min: float = 0.05):
        if drift_comp not in ("off", "rate", "pair"):
            raise ValueError(f"unknown drift_comp: {drift_comp}")
        if alpha_mode not in ("fixed", "optimal"):
            raise ValueError(f"unknown alpha_mode: {alpha_mode}")
        self.drift_comp, self.alpha_mode = drift_comp, alpha_mode
        self.src_bonus, self.min_weight, self.alpha_min = float(src_bonus), float(min_weight), float(alpha_min)

    @property
    def active(self) -> bool:
        return self.drift_comp != "off" or self.alpha_mode == "optimal"

    @staticmethod
    def bandwidth(cur_means: Sequence[torch.Tensor]) -> Optional[float]:
        """h = median pairwise distance among the current class means."""
        if len(cur_means) < 2:
            return None
        X = torch.stack([m.detach().float().flatten().cpu() for m in cur_means])
        d = torch.pdist(X)
        h = float(d.median()) if d.numel() else 0.0
        return h if h > 0 else None

    def estimate(self, state: DriftState, records, now: int, h: Optional[float], device):
        """Per record: (displacement to the current space [R,d], residual var [R], ok [R] bool)."""
        R = len(records)
        d = records[0].mu.numel()
        disp = torch.zeros(R, d)
        var = torch.zeros(R)
        ok = torch.zeros(R, dtype=torch.bool)
        if h is None or R == 0:
            return disp, var, ok
        mu = torch.stack([r.mu.detach().float().flatten() for r in records]).to(device)
        src = [r.source_id for r in records]
        age = torch.tensor([float(now - r.round_idx) for r in records], device=device)
        if self.drift_comp == "pair":
            for s in sorted({r.round_idx for r in records}):
                idx = [i for i, r in enumerate(records) if r.round_idx == s]
                view = state._pair_view(s, now, device)
                if view is None:
                    continue
                D, A, ss = view
                m, v, o = _kernel_mean(mu[idx], [src[i] for i in idx], D, A, ss, h, self.src_bonus, self.min_weight)
                ii = torch.tensor(idx)
                disp[ii], var[ii], ok[ii] = m.cpu(), v.cpu(), o.cpu()
        todo = ~ok
        view = state._rate_view(device)
        if view is not None and bool(todo.any()):
            Rt, A, ss = view
            idx = torch.nonzero(todo).flatten().tolist()
            m, v, o = _kernel_mean(mu[idx], [src[i] for i in idx], Rt, A, ss, h, self.src_bonus, self.min_weight)
            a = age[idx].unsqueeze(1)
            ii = torch.tensor(idx)
            disp[ii], var[ii], ok[ii] = (m * a).cpu(), (v * a.squeeze(1) ** 2).cpu(), o.cpu()
        return disp, var, ok

    def optimal_alpha(self, b2: float, v: float, n: float, fixed: float) -> Tuple[float, bool]:
        """alpha* = b2 / (b2 + v/n) clipped to [alpha_min, 1]; (fixed, True) if not estimable."""
        if b2 is None or n is None or n <= 0 or v is None:
            return fixed, True
        noise = v / n
        den = b2 + noise
        if den <= 0:
            return fixed, True
        a = b2 / den
        return float(min(max(a, self.alpha_min), 1.0)), False


def compensated_copy(record, mu_new: torch.Tensor):
    """A record with a replaced mean; the memory itself is never modified."""
    return replace(record, mu=mu_new.reshape(record.mu.shape).to(record.mu.dtype))
