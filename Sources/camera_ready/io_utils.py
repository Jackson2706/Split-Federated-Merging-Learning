"""
I/O helpers for camera-ready experiments: result paths, JSON/CSV writers, and
mean +/- std aggregation over seeds. Pure-Python (no torch dependency).
"""

import csv
import json
import math
import os
from collections import defaultdict

# Repo root = parent of this package directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR_ROOT = os.path.join(REPO_ROOT, "results", "camera_ready")


def cr_dir(*parts):
    """Return (and create) a subdirectory under results/camera_ready/."""
    d = os.path.join(CR_ROOT, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    # Make numpy / torch scalars and arrays JSON-serializable.
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def write_csv(rows, path, fieldnames=None):
    """Write a list of dicts to CSV. fieldnames inferred (ordered union) if None."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate_seeds(rows, group_keys, metric_keys):
    """Aggregate rows (list of dicts) to mean/std per group.

    Args:
        rows: list of dicts (e.g. one per seed).
        group_keys: keys identifying an experiment condition (everything but seed).
        metric_keys: numeric keys to aggregate.

    Returns:
        list of dicts, one per group, with <metric>_mean, <metric>_std, n_seeds.
    """
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in group_keys)
        buckets[key].append(r)

    out = []
    for key, group in buckets.items():
        agg = {k: v for k, v in zip(group_keys, key)}
        agg["n_seeds"] = len(group)
        for m in metric_keys:
            vals = []
            for r in group:
                try:
                    vals.append(float(r[m]))
                except (KeyError, TypeError, ValueError):
                    pass
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                agg[f"{m}_mean"] = mean
                agg[f"{m}_std"] = math.sqrt(var)
            else:
                agg[f"{m}_mean"] = ""
                agg[f"{m}_std"] = ""
        out.append(agg)
    return out


def fmt_mean_std(mean, std, pct=False, prec=2):
    """Format 'mean +/- std' for tables. Returns '' if mean is missing."""
    if mean == "" or mean is None:
        return ""
    scale = 100.0 if pct else 1.0
    try:
        m = float(mean) * scale
        s = float(std) * scale if std not in ("", None) else 0.0
    except (TypeError, ValueError):
        return ""
    return f"{m:.{prec}f} $\\pm$ {s:.{prec}f}"
