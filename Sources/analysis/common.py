"""Shared loading for the revision analysis scripts (reads results/revision only)."""
import csv
import glob
import json
import math
import os
import re

import yaml

EXPECTED_SEEDS = 3
# Reference configuration of each experiment (paired tests compare against it).
REF_E3 = "E3_ehsfp_cifar100_iid_alpha0p3_tau0_oracle1"  # = paper reference condition
REFERENCE = {
    "E3": REF_E3,
    "E4": "E4_ehsfp_cifar100_iid_p_train0p2_loss0p3_oracle1",
    "E6": REF_E3,  # lambda_prc = 0.1, standard PRC is the reference config (oracle changes no training)
}
E5_REFERENCE = "E5_ehsfp_{dataset}_dir0p3_ref"


def load_runs(root):
    """Finished runs (DONE present): one dict per run directory."""
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        if not os.path.exists(os.path.join(d, "DONE")):
            continue
        rid = os.path.basename(os.path.normpath(d))
        m = re.match(r"(.+)_s(\d+)$", rid)
        if not m:
            continue
        runs.append({
            "run_id": rid, "config_id": m.group(1), "exp": rid.split("_")[0], "seed": int(m.group(2)),
            "dir": d, "summary": json.load(open(os.path.join(d, "summary.json"))),
            "config": yaml.safe_load(open(os.path.join(d, "config.yaml"))),
        })
    return runs


def by_config(runs):
    out = {}
    for r in runs:
        out.setdefault(r["config_id"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return math.nan, math.nan, 0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else math.nan
    return m, sd, n


def check_seeds(groups, expected, allow_partial):
    short = {c: len(v) for c, v in groups.items() if len(v) < expected}
    if short and not allow_partial:
        raise SystemExit("configurations with fewer than %d seeds: %s" % (expected, short))
    return short


def write_table(path_base, header, rows):
    """CSV + LaTeX tabular with the same content."""
    os.makedirs(os.path.dirname(path_base), exist_ok=True)
    with open(path_base + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    with open(path_base + ".tex", "w") as fh:
        fh.write("\\begin{tabular}{@{}l" + "c" * (len(header) - 1) + "@{}}\n\\toprule\n")
        fh.write(" & ".join(esc(h) for h in header) + " \\\\\n\\midrule\n")
        for r in rows:
            fh.write(" & ".join(esc(c) for c in r) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def pm(m, sd, digits=2):
    if math.isnan(m):
        return ""
    return f"{m:.{digits}f} $\\pm$ {sd:.{digits}f}" if not math.isnan(sd) else f"{m:.{digits}f}"
