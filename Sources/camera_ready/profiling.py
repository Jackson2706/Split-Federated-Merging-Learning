"""
CUDA-synchronized phase timing for the runtime-overhead experiment.

Usage:
    prof = ProfileLog()
    with prof.phase("pack", round_idx=epoch):
        ... client packing ...
    ...
    prof.to_csv(path); prof.summary()

Timers call torch.cuda.synchronize() around GPU work so measurements are accurate.
The hooks in hierarchy.py are gated on config["profile"]; when disabled, no timing
code runs and behavior is unchanged.
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


class ProfileLog:
    def __init__(self, use_cuda_sync=True):
        self.use_cuda_sync = use_cuda_sync
        self.records = []                 # list of {phase, round, seconds}
        self._totals = defaultdict(float) # phase -> cumulative seconds

    @contextmanager
    def phase(self, name, round_idx=None):
        if self.use_cuda_sync:
            _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.use_cuda_sync:
                _sync()
            dt = time.perf_counter() - t0
            self.records.append({"phase": name, "round": round_idx, "seconds": dt})
            self._totals[name] += dt

    def add(self, name, seconds, round_idx=None):
        self.records.append({"phase": name, "round": round_idx, "seconds": seconds})
        self._totals[name] += seconds

    def total(self, name):
        return self._totals.get(name, 0.0)

    def summary(self):
        """Per-phase {count, total_s, mean_s}."""
        counts = defaultdict(int)
        for r in self.records:
            counts[r["phase"]] += 1
        return {
            p: {"count": counts[p], "total_s": self._totals[p],
                "mean_s": self._totals[p] / max(counts[p], 1)}
            for p in self._totals
        }

    def to_csv(self, path):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "round", "seconds"])
            w.writeheader()
            for r in self.records:
                w.writerow(r)
        return path

    def to_json(self, path):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"records": self.records, "summary": self.summary()},
                      f, indent=2)
        return path
