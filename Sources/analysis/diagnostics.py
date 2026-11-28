#!/usr/bin/env python3
"""Section VII-C drift and coverage diagnostics from the E3/E4 oracle logs.

D1  Delta_proto and kappa over rounds (edge mean): reference (alpha=0.3), alpha=1,
    p_train=0 under packet loss (E4); mean +- std band over seeds.
D2  Empirical class-loss rate of the source mask vs Proposition 2,
    p^{S_j} - (S_j/S) p^S, binned by (p, S, S_j); y = x reference line.
D3  Time-averaged Delta_proto, kappa and test accuracy vs alpha for tau = 0, 5.
D4  Spearman correlation of time-averaged Delta_proto with test accuracy across
    the E3/E4 configurations (configuration means) and across runs.
T   Fraction of episodes without a valid minibatch; share of classes restored by replay.
Outputs go to analysis/out/ (PDF figures sized for one IEEE column, CSV/LaTeX tables).
"""
import argparse
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

from common import REF_E3, by_config, load_runs, mean_std, num, read_csv, write_table  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})
COL = 3.5  # IEEE column width (inches)
D1_CONFIGS = [(REF_E3, r"reference ($\alpha=0.3$)"),
              ("E3_ehsfp_cifar100_iid_alpha1p0_tau0_oracle1", r"$\alpha=1$ (no replay)"),
              ("E4_ehsfp_cifar100_iid_p_train0p0_loss0p3_oracle1", r"$p_{\mathrm{train}}=0$, loss 0.3")]


def oracle_series(run, tier="edge"):
    """{round: (mean Delta over active learners, mean kappa over all learners)}."""
    by_round = {}
    for r in read_csv(os.path.join(run["dir"], "oracle.csv")):
        if r["tier"] == tier:
            by_round.setdefault(int(r["round"]), []).append(r)
    out = {}
    for rnd, rows in sorted(by_round.items()):
        deltas = [num(r["delta_proto"]) for r in rows if r["delta_proto"] != ""]
        kappas = [num(r["kappa"]) for r in rows if r["kappa"] != ""]
        out[rnd] = (sum(deltas) / len(deltas) if deltas else math.nan,
                    sum(kappas) / len(kappas) if kappas else math.nan)
    return out


def time_avg(run):
    s = oracle_series(run)
    d = [v[0] for v in s.values() if not math.isnan(v[0])]
    k = [v[1] for v in s.values() if not math.isnan(v[1])]
    return (sum(d) / len(d) if d else math.nan), (sum(k) / len(k) if k else math.nan)


def ival(v):
    return 0 if v in (None, "") else int(float(v))


def acc(run):
    return 100 * num(run["summary"].get("test_accuracy"))


def d1(groups, out):
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.6), sharex=True)
    for cid, label in D1_CONFIGS:
        if cid not in groups:
            continue
        series = [oracle_series(r) for r in groups[cid]]
        rounds = sorted(set().union(*series))
        for ax, idx in zip(axes, (0, 1)):
            stat = [mean_std([s[t][idx] for s in series if t in s]) for t in rounds]
            m = [x[0] for x in stat]
            sd = [0 if math.isnan(x[1]) else x[1] for x in stat]
            ax.plot(rounds, m, label=f"{label} (n={len(series)})")
            lo, hi = [a - b for a, b in zip(m, sd)], [a + b for a, b in zip(m, sd)]
            if idx == 1:  # kappa is a fraction
                lo, hi = [max(0.0, v) for v in lo], [min(1.0, v) for v in hi]
            ax.fill_between(rounds, lo, hi, alpha=0.2)
    axes[0].set_ylabel(r"$\Delta_{\mathrm{proto}}$ (edge mean)")
    axes[1].set_ylabel(r"$\kappa_t$ (edge mean)")
    axes[1].set_xlabel("round")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_d1_drift_coverage.pdf")
    plt.close(fig)


def d2(groups, out):
    cells = {}
    for cid, runs in groups.items():
        if not (cid.startswith("E3_") or cid.startswith("E4_")):
            continue
        for run in runs:
            for r in read_csv(os.path.join(run["dir"], "oracle_mask.csv")):
                key = (round(num(r["p"]), 4), int(r["S"]), int(r["S_j"]))
                c = cells.setdefault(key, [0, 0])
                c[0] += 1
                c[1] += int(r["removed"])
    rows = []
    for (p, S, Sj), (n, k) in sorted(cells.items()):
        pred = p ** Sj - (Sj / S) * p ** S
        rows.append([p, S, Sj, n, k / n, pred])
    write_table(f"{out}/d2_prop2_check", ["p", "S", "S_j", "class-episodes", "empirical", "Prop. 2"],
                [[p, S, Sj, n, f"{e:.4f}", f"{q:.4f}"] for p, S, Sj, n, e, q in rows])
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    if rows:
        sizes = [max(4, min(60, 4 * math.log10(1 + r[3]) ** 2)) for r in rows]
        ax.scatter([r[5] for r in rows], [r[4] for r in rows], s=sizes, alpha=0.7)
        top = max(max(r[4] for r in rows), max(r[5] for r in rows), 1e-3) * 1.05
        ax.plot([0, top], [0, top], "k--", lw=0.8, label="$y=x$")
        ax.legend(frameon=False)
    ax.set_xlabel(r"predicted $p^{S_j}-(S_j/S)p^S$")
    ax.set_ylabel("empirical class-loss rate")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_d2_prop2.pdf")
    plt.close(fig)


def config_points(groups, prefix):
    pts = []
    for cid, runs in sorted(groups.items()):
        if not cid.startswith(prefix):
            continue
        per = [time_avg(r) for r in runs]
        pts.append({"config": cid, "runs": runs, "delta": [x[0] for x in per], "kappa": [x[1] for x in per],
                    "acc": [acc(r) for r in runs]})
    return pts


def d3(groups, out):
    pts = config_points(groups, "E3_")
    fig, axes = plt.subplots(1, 3, figsize=(2 * COL, 2.0))
    rows = []
    for tau in (0, 5):
        sel = []
        for p in pts:
            m = re.search(r"alpha(\d+p\d+)_tau(\d+)", p["config"])
            if m and int(m.group(2)) == tau:
                sel.append((float(m.group(1).replace("p", ".")), p))
        sel.sort(key=lambda x: x[0])
        for ax, key, ylab in zip(axes, ("delta", "kappa", "acc"),
                                 (r"time-avg $\Delta_{\mathrm{proto}}$", r"time-avg $\kappa$", "test acc. (%)")):
            stat = [mean_std(p[key]) for _, p in sel]
            ax.errorbar([a for a, _ in sel], [s[0] for s in stat],
                        yerr=[0 if math.isnan(s[1]) else s[1] for s in stat], marker="o", ms=3, capsize=2,
                        label=rf"$\tau={tau}$")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(ylab)
        for a, p in sel:
            d, k, c = mean_std(p["delta"]), mean_std(p["kappa"]), mean_std(p["acc"])
            rows.append([tau, a, c[2], f"{d[0]:.4g}", f"{d[1]:.2g}", f"{k[0]:.3f}", f"{k[1]:.3f}",
                         f"{c[0]:.2f}", f"{c[1]:.2f}"])
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_d3_alpha_sweep.pdf")
    plt.close(fig)
    write_table(f"{out}/d3_alpha_sweep", ["tau", "alpha", "n", "Delta mean", "Delta std", "kappa mean",
                                         "kappa std", "Acc mean (%)", "Acc std"], rows)


def d4(groups, out):
    pts = config_points(groups, "E3_") + config_points(groups, "E4_")
    cfg_d = [mean_std(p["delta"])[0] for p in pts]
    cfg_a = [mean_std(p["acc"])[0] for p in pts]
    run_d = [d for p in pts for d in p["delta"]]
    run_a = [a for p in pts for a in p["acc"]]
    res = {}
    for name, x, y in (("configurations", cfg_d, cfg_a), ("runs", run_d, run_a)):
        keep = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
        if len(keep) >= 3:
            r = stats.spearmanr([a for a, _ in keep], [b for _, b in keep])
            res[name] = {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(keep)}
        else:
            res[name] = {"rho": None, "p": None, "n": len(keep)}
    with open(f"{out}/d4_spearman.json", "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def table(groups, out):
    rows = []
    for cid, runs in sorted(groups.items()):
        if not (cid.startswith("E3_") or cid.startswith("E4_")):
            continue
        nv = ep = rest = rcls = 0
        for run in runs:
            for r in read_csv(os.path.join(run["dir"], "metrics.csv")):
                nv += ival(r.get("no_valid_minibatch"))
                ep += ival(r.get("episodes"))
            for r in read_csv(os.path.join(run["dir"], "oracle.csv")):
                rest += ival(r.get("restored"))
                rcls += ival(r.get("replay_classes"))
        rows.append([cid, len(runs), f"{nv / ep:.4f}" if ep else "", f"{rest / rcls:.4f}" if rcls else ""])
    write_table(f"{out}/diagnostics_table", ["configuration", "n", "episodes without valid minibatch",
                                             "classes restored by replay (share)"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/revision")
    ap.add_argument("--out", default="analysis/out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups = by_config(load_runs(a.root))
    d1(groups, a.out)
    d2(groups, a.out)
    d3(groups, a.out)
    res = d4(groups, a.out)
    table(groups, a.out)
    print(f"D1-D3 figures, D2/D3/diagnostics tables and D4 {res} -> {a.out}/")


if __name__ == "__main__":
    main()
