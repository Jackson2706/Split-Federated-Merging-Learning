#!/usr/bin/env python3
"""Generate the E-HSFP journal figures from recorded runs.

    cd "Submissions/E-HSFP Journal"
    python figures/make_figures.py

Inputs (all produced by the campaign; nothing here is estimated):
  ../../Sources/artifacts/per_run_metrics.csv     (from collect_journal_v1.py)
  <run_dir>/history.csv                          per-round test accuracy, comm, tier GPU peaks
  <run_dir>/diagnostics/episodic_diagnostics.csv per-round oracle drift (Def. 6)
  <run_dir>/diagnostics/reliability_weights.csv  reliability weights and their factors

Outputs: figures/fig_<name>.pdf (+ .png) and figures/data/fig_<name>.csv holding
every plotted number (mean, sample SD with ddof=1, n seeds) for checking.
A figure whose runs have not finished is not written, so the manuscript keeps
its placeholder (see \\figorph in sections/experiments.tex).
Curves are journal-protocol v2 runs (run IDs j2_*); bands are +-1 sample SD.
Fig. 4 (stability-bound term) is intentionally not generated: the constants of
Theorem 1 are unspecified, so the bound cannot be evaluated numerically.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "Sources"
OUT, DATA = HERE, HERE / "data"
SEEDS = range(3)  # figures: seeds 0-2 (user decision 2026-10-06); tables: see fill_manuscript.py
# Run-ID prefix of the corrected method (protocol v3). Superseded v2 runs use j2_.
JP = "j3"
RUNS = {r["experiment_id"]: r for r in csv.DictReader(open(SRC / "artifacts" / "per_run_metrics.csv"))}
# Per-round curves of runs completed on the legacy host whose run dirs were not migrated, recovered from the
# committed figure data and verified against the ledger (Sources/scripts/campaign/extract_legacy_curves.py).
_LP = SRC / "results" / "journal_v1" / "legacy" / "figure_curves_oldmachine.json"
LEGACY = {(c["prefix"], c["what"]): c for c in json.load(open(_LP))["curves"]} if _LP.exists() else {}
WHAT = {"history.csv": "acc", "diagnostics/episodic_diagnostics.csv": "drift"}


class Agg:
    """A legacy series stored only as seed mean/SD (all n seeds from the legacy host)."""
    def __init__(self, mean, sd, n):
        self.mean, self.sd, self.n = np.asarray(mean), np.asarray(sd), n

    def __mul__(self, k):
        return Agg(self.mean * k, self.sd * abs(k), self.n)


plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "figure.dpi": 150})


def done(eid):
    r = RUNS.get(eid)
    return r if r and r["status"].startswith("completed-") and r.get("run_dir") else None


def series(prefix, file, col, filt=None, agg=np.mean):
    """Per-seed per-round series of `col` from <run_dir>/<file>."""
    out, agg_seeds = [], []
    lg = LEGACY.get((prefix, WHAT.get(file)))
    for s in SEEDS:
        r = done(f"{prefix}_s{s}")
        if not r:
            continue
        p = Path(r["run_dir"]) / file
        if not p.exists() and file.startswith("diagnostics/") and r.get("diag_dir"):
            p = Path(r["diag_dir"]) / file.split("/", 1)[1]
        if not p.exists():
            if lg and lg["n"] == 1 and s == 0:   # legacy n=1 series == this run's exact per-round curve
                out.append(list(lg["mean"])); _LEGACY_USED.add(f"{prefix}_s0"); continue
            if lg and lg["n"] > 1 and s < lg["n"]:
                agg_seeds.append(s); continue
            _MISSING.append(f"{prefix}_s{s}:{file}")   # completed run whose curve is unavailable here
            continue
        by = defaultdict(list)
        for row in csv.DictReader(open(p)):
            if filt and not filt(row):
                continue
            try:
                by[int(row["round"])].append(float(row[col]))
            except (ValueError, KeyError):
                pass
        if by:
            out.append([agg(by[k]) for k in sorted(by)])
    if agg_seeds:
        if agg_seeds == list(range(lg["n"])) and not out:
            _LEGACY_USED.update(f"{prefix}_s{s}" for s in agg_seeds)
            return Agg(lg["mean"], lg["sd"], lg["n"])
        _MISSING.extend(f"{prefix}_s{s}:{file} (legacy aggregate cannot be combined)" for s in agg_seeds)
    if not out:
        return None
    L = min(len(x) for x in out)
    return np.array([x[:L] for x in out])


def scalar(prefix, key, scale=1.0):
    v = []
    for s in SEEDS:
        r = done(f"{prefix}_s{s}")
        try:
            v.append(float(r[key]) * scale)
        except (TypeError, ValueError, KeyError):
            pass
    return v


def band(ax, a, label, rows, name):
    if isinstance(a, Agg):
        mu, sd, n = a.mean, a.sd, a.n
    else:
        mu = a.mean(0)
        sd = a.std(0, ddof=1) if a.shape[0] > 1 else np.zeros_like(mu)
        n = a.shape[0]
    x = np.arange(1, len(mu) + 1)
    ax.plot(x, mu, label=f"{label} (n={n})", lw=1.2)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.18)
    rows += [{"figure": name, "series": label, "round": int(i), "mean": float(m), "sd": float(d), "n": n}
             for i, m, d in zip(x, mu, sd)]


_LEGACY_USED = set()  # runs whose curve came from the legacy-host store
_MISSING = []  # filled by series(); a figure with any missing contributor is frozen, not overwritten


def save(fig, name, rows):
    if _MISSING:
        plt.close(fig)
        print(f"frozen {name}: {len(_MISSING)} completed runs lack per-round artifacts on this machine "
              f"(e.g. {_MISSING[0]}); existing figure kept")
        _MISSING.clear()
        return
    if not rows:
        plt.close(fig)
        print(f"skip {name}: no completed runs yet")
        return
    if _LEGACY_USED:
        print(f"  {name}: {len(_LEGACY_USED)} curves from legacy-host figure data")
        _LEGACY_USED.clear()
    DATA.mkdir(exist_ok=True)
    with open(DATA / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}.pdf ({len(rows)} plotted values)")


ACC = ("history.csv", "validation_metric_value")
EDGE_DRIFT = ("diagnostics/episodic_diagnostics.csv", "drift_w2", lambda r: r["tier"] == "edge")


def curve_fig(name, specs, what=ACC, ylabel="Test accuracy (%)", scale=100.0, ncols=1, titles=None):
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 2.6), squeeze=False)
    rows = []
    for ax, (title, spec) in zip(axes[0], zip(titles or [""], specs if ncols > 1 else [specs])):
        for label, prefix in spec:
            a = series(prefix, *what) if len(what) == 3 else series(prefix, what[0], what[1])
            if a is not None:
                band(ax, a * scale, label, rows, name)
        ax.set_xlabel("Communication round"); ax.set_ylabel(ylabel); ax.set_title(title)
        if ax.lines:
            ax.legend()
    save(fig, name, rows)


def main():
    # Fig. 2 — convergence, IID vs non-IID, H-SFP vs E-HSFP
    specs, titles = [], []
    for d, dn in (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("ham10000", "HAM10000")):
        sp = []
        for tag, pl in (("", "IID"), ("noniid_a0p7_", "dir(0.7)"), ("noniid_a0p3_", "dir(0.3)")):
            for m, ml in (("hsfp", "H-SFP"), ("ehsfp", "E-HSFP")):
                sp.append((f"{ml} {pl}", f"{JP}_{tag}{d}_{m}"))
        specs.append(sp); titles.append(dn)
    curve_fig("fig_conv_curves", specs, ncols=3, titles=titles)

    # Fig. 3 — oracle edge drift over rounds under the stress protocol
    curve_fig("fig_drift_rounds", [("No event loss", f"{JP}_sv_cifar100_none"), ("Combined stress", f"{JP}_sv_cifar100_combined"),
                                   (r"$\tau$=5", f"{JP}_stale_cifar100_tau5"), ("p=0.5", f"{JP}_dropout_cifar100_p0p5")],
              what=EDGE_DRIFT, ylabel=r"Edge drift $W_2^2$ (oracle)", scale=1.0)

    # Figs. 5/6 — dropout and staleness: accuracy curves + drift curves
    for name, spec in (
        ("fig_dropout_effect", [(f"p={p}", f"{JP}_cifar100_ehsfp" if p == "0.2" else f"{JP}_dropout_cifar100_p{p.replace('.', 'p')}")
                                for p in ("0.0", "0.1", "0.2", "0.3", "0.5")]),
        ("fig_staleness_effect", [(rf"$\tau$={t}", f"{JP}_cifar100_ehsfp" if t == 0 else f"{JP}_stale_cifar100_tau{t}")
                                  for t in (0, 1, 3, 5, 10)])):
        spec = [(l, p.replace("p0.0", "p0p0")) for l, p in spec]
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
        rows = []
        for label, prefix in spec:
            a = series(prefix, *ACC)
            if a is not None:
                band(axes[0], a * 100, label, rows, name + ":acc")
            b = series(prefix, *EDGE_DRIFT)
            if b is not None:
                band(axes[1], b, label, rows, name + ":drift")
        axes[0].set_ylabel("Test accuracy (%)"); axes[1].set_ylabel(r"Edge drift $W_2^2$")
        for ax in axes:
            ax.set_xlabel("Communication round")
            if ax.lines:
                ax.legend()
        save(fig, name, rows)

    # Fig. 7 — module ablation bars
    spec = [("H-SFP", f"{JP}_cifar100_hsfp"), ("+mem", f"{JP}_abl_cifar100_hsfp_memory"),
            ("+drop", f"{JP}_abl_cifar100_hsfp_memory_dropout"), ("+rel", f"{JP}_abl_cifar100_hsfp_memory_reliability"),
            ("+PRC", f"{JP}_abl_cifar100_hsfp_memory_reliability_prc"), ("Full", f"{JP}_cifar100_ehsfp")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    rows = []
    for ax, (key, lab, sc) in zip(axes, (("final_acc", "Final acc. (%)", 100), ("conv_rounds", "Rounds to conv.", 1),
                                         ("stability_final", "Stability (pp$^2$)", 1e4))):
        xs, ms, sds = [], [], []
        for l, p in spec:
            v = scalar(p, key, sc)
            if v:
                xs.append(f"{l}\nn={len(v)}"); ms.append(np.mean(v)); sds.append(np.std(v, ddof=1) if len(v) > 1 else 0)
                rows.append({"figure": "fig_ablation_modules", "series": l, "metric": key, "mean": ms[-1], "sd": sds[-1], "n": len(v)})
        ax.bar(xs, ms, yerr=sds, capsize=2); ax.set_ylabel(lab); ax.tick_params(axis="x", labelsize=6)
    save(fig, "fig_ablation_modules", rows)

    # Fig. 8 — reliability weights vs factors (edge tier, full E-HSFP runs)
    pts = defaultdict(list)
    for s in SEEDS:
        r = done(f"{JP}_cifar100_ehsfp_s{s}")
        p = (Path(r["diag_dir"]) if r.get("diag_dir") else Path(r["run_dir"]) / "diagnostics") / "reliability_weights.csv" if r else None
        if p and p.exists():
            for row in csv.DictReader(open(p)):
                if row["tier"] == "edge":
                    for k in ("support_feat", "sigma_feat", "age_feat", "distance_feat"):
                        pts[k].append((float(row[k]), float(row["normalized_weight"])))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
    rows = []
    for ax, (k, lab) in zip(axes, (("support_feat", "support"), ("sigma_feat", "variance"),
                                   ("age_feat", "age"), ("distance_feat", "ref. distance"))):
        if pts[k]:
            x, y = np.array(pts[k]).T
            bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(x, bins) - 1, 0, 9)
            for b in range(10):
                if (idx == b).any():
                    rows.append({"figure": "fig_reliability_weights", "series": lab, "bin_center": (bins[b] + bins[b + 1]) / 2,
                                 "mean": float(y[idx == b].mean()), "n": int((idx == b).sum())})
            bc = [r["bin_center"] for r in rows if r["series"] == lab]; bm = [r["mean"] for r in rows if r["series"] == lab]
            ax.plot(bc, bm, marker="o", ms=3)
        ax.set_xlabel(f"{lab} (normalised)")
    axes[0].set_ylabel("Mean normalised weight")
    save(fig, "fig_reliability_weights", rows)

    # Fig. 9 — memory effect over rounds (memory window M)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    rows = []
    for m in (0, 1, 5, 10, 20):
        a = series(f"{JP}_memM_cifar100_M{m}", *ACC)
        if a is not None:
            band(axes[0], a * 100, f"M={m}", rows, "fig_memory_effect:acc")
        b = series(f"{JP}_memM_cifar100_M{m}", *EDGE_DRIFT)
        if b is not None:
            band(axes[1], b, f"M={m}", rows, "fig_memory_effect:drift")
    axes[0].set_ylabel("Test accuracy (%)"); axes[1].set_ylabel(r"Edge drift $W_2^2$")
    for ax in axes:
        ax.set_xlabel("Communication round")
        if ax.lines:
            ax.legend()
    save(fig, "fig_memory_effect", rows)

    # Fig. 10 — communication vs accuracy (journal protocol, final round)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    rows = []
    meths = (("FedAvg", "main_{d}_federated"), ("FedProx", "fed_{d}_fedprox"), ("FedNova", "fed_{d}_fednova"),
             ("HierFL", "main_{d}_hierfl"), ("SplitFed", "main_{d}_splitfl"), ("HeteroSFL", "main_{d}_hetero-sfl"),
             ("HSFL", "main_{d}_hsfl"), ("H-SFP", JP + "_{d}_hsfp"), ("E-HSFP", JP + "_{d}_ehsfp"))
    for ax, (d, dn) in zip(axes, (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("ham10000", "HAM10000"))):
        for lab, tpl in meths:
            c, a = scalar(tpl.format(d=d), "comm_MB", 1 / 1024), scalar(tpl.format(d=d), "final_acc", 100)
            if c and a:
                n = min(len(c), len(a))
                ax.errorbar(np.mean(c), np.mean(a), yerr=np.std(a, ddof=1) if n > 1 else 0, fmt="o", ms=3, capsize=2,
                            label=lab, mec="k" if lab == "E-HSFP" else None)
                rows.append({"figure": "fig_comm_acc_tradeoff", "dataset": d, "series": lab, "comm_GB": float(np.mean(c)),
                             "acc_mean": float(np.mean(a)), "acc_sd": float(np.std(a, ddof=1)) if n > 1 else 0.0, "n": n})
        ax.set_xscale("log"); ax.set_xlabel("Total comm. (GB, 60 rounds)"); ax.set_title(dn)
    axes[0].set_ylabel("Final test acc. (%)"); axes[-1].legend(loc="lower right")
    save(fig, "fig_comm_acc_tradeoff", rows)

    # Fig. 11 — tier memory vs performance (process CUDA peak per tier phase)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    rows = []
    for tier, mk in (("client", "o"), ("edge", "s"), ("cloud", "^")):
        xs, ys = [], []
        for eid, r in RUNS.items():
            if eid.startswith(f"{JP}_") and r["status"].startswith("completed-"):
                try:
                    xs.append(float(r[f"{tier}_gpu_peak_MB"])); ys.append(float(r["final_acc"]) * 100)
                except (ValueError, KeyError):
                    pass
        if xs:
            ax.scatter(xs, ys, marker=mk, s=10, label=f"{tier} (n={len(xs)})")
            rows += [{"figure": "fig_mem_perf_tradeoff", "series": tier, "gpu_peak_MB": x, "final_acc": y} for x, y in zip(xs, ys)]
    ax.set_xlabel("CUDA peak during tier phase (MB)"); ax.set_ylabel("Final test acc. (%)")
    if rows:
        ax.legend()
    save(fig, "fig_mem_perf_tradeoff", rows)


if __name__ == "__main__":
    main()
