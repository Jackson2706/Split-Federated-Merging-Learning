#!/usr/bin/env python3
"""Deterministic MiSFP collector: completed runs -> results/measurements.csv -> table rows + figures.

    python scripts/campaign/collect_misfp.py

Only runs with a validated .done marker in results/journal_v1/.markers (gated by runmeta.py validate)
are read. For each run the final-round (round 60) composed-model test metrics are used -- never a
checkpoint selected on any split. Aggregates are mean +- sample SD (ddof=1) over the registered seeds
{1,2,3}; a cell with fewer seeds is printed with its actual n (``^{n=2}``), a single seed has no SD,
and a cell with no completed run is \\MissingResult. Nothing is imputed.

Comparability key (config_hash here) = sha256 of the run's resolved config with seed/output paths/
attempt counters removed, so seeds of the same configuration aggregate and anything else does not.

Fidelity (tab:fidelity) is computed post hoc from misfp_pca_payload.npz: held-out (val_from_train)
real features vs samples from the transmitted client->edge mixture and from its moment-collapsed
single Gaussian (same packet, so the comparison isolates the representation). SWD uses 64 random
projections; MMD uses an RBF kernel with the median-distance bandwidth of the real features; both
with a fixed RNG (seed 0). Mode coverage / minority recall need a clustering fit on TRAINING features,
which the runner does not persist -> reported as \\MissingResult with that reason.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
PAPER = SRC.parent / "Submissions" / "MiSFP Journal"
MARK = SRC / "results" / "journal_v1" / ".markers"
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/workspace/campaign/outputs_journal_v1"))
GEN = PAPER / "generated"
SEEDS = (1, 2, 3)
EID = re.compile(r"^misfp_(iid|dir0\.1|dir0\.7)_(hsfp|k1|edge|k2|k4|adaptive)(?:_(\w+?))?_s(\d+)$")
VOLATILE = {"seed", "output_dir", "misfp_attempt", "runtime_metrics_path", "runtime_counters_path",
            "history_path", "run_fingerprints_path", "prediction_artifact_dir", "resolved_config_hash",
            "diagnostics_dir", "gpu", "is_gpu"}
LABEL = {"hsfp": "H-SFP", "k1": "MiSFP-K1", "edge": "MiSFP-Edge", "k2": "MiSFP-K2", "k4": "MiSFP-K4",
         "adaptive": "MiSFP-Adaptive"}
MISSING = r"\MissingResult"


def comparable_hash(cfg):
    clean = {k: v for k, v in cfg.items() if k not in VOLATILE}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:16]


def swd(a, b, rng, n_proj=64):
    d = a.shape[1]
    P = rng.standard_normal((d, n_proj)); P /= np.linalg.norm(P, axis=0, keepdims=True)
    pa, pb = np.sort(a @ P, axis=0), np.sort(b @ P, axis=0)
    q = np.linspace(0, 1, 100)
    return float(np.mean(np.abs(np.quantile(pa, q, axis=0) - np.quantile(pb, q, axis=0))))


def mmd(a, b):
    def sq(x, y):
        return (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T
    daa = sq(a, a)
    med = np.median(daa[np.triu_indices_from(daa, 1)]) if a.shape[0] > 1 else 1.0
    g = 1.0 / max(med, 1e-12)
    k = lambda x, y: np.exp(-g * np.maximum(sq(x, y), 0))
    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def sample_mixture(counts, means, vars_, n, rng):
    w = counts / counts.sum()
    comp = rng.choice(len(w), size=n, p=w)
    return means[comp] + rng.standard_normal((n, means.shape[1])) * np.sqrt(np.maximum(vars_[comp], 0))


def fidelity(rd):
    p = os.path.join(rd, "misfp_pca_payload.npz")
    if not os.path.exists(p):
        return {}
    z = np.load(p)
    classes = sorted({k.split("_")[0] for k in z.files})
    rng = np.random.default_rng(0)
    out = defaultdict(list)
    for c in classes:
        real = z[f"{c}_real"].astype(np.float64)
        counts, means, vars_ = (z[f"{c}_{k}"].astype(np.float64) for k in ("counts", "means", "vars"))
        if real.shape[0] < 5 or counts.sum() <= 0:
            continue
        w = counts / counts.sum()
        mu = (w[:, None] * means).sum(0)
        var = (w[:, None] * (vars_ + means ** 2)).sum(0) - mu ** 2      # exact moment collapse
        n = real.shape[0]
        mix = sample_mixture(counts, means, vars_, n, rng)
        sing = mu + rng.standard_normal((n, mu.shape[0])) * np.sqrt(np.maximum(var, 0))
        out["swd_mixture"].append(swd(real, mix, rng)); out["swd_single"].append(swd(real, sing, rng))
        out["mmd_mixture"].append(mmd(real, mix)); out["mmd_single"].append(mmd(real, sing))
    return {k: float(np.mean(v)) for k, v in out.items()} | {"fidelity_classes": len(out.get("swd_mixture", []))}


def load_run(eid):
    m = EID.match(eid)
    if not m:
        return None
    split, variant, extra, seed = m.group(1), m.group(2), m.group(3) or "", int(m.group(4))
    sums = glob.glob(str(OUT_ROOT / eid / "**" / "misfp_summary.json"), recursive=True)
    if len(sums) != 1:
        return {"eid": eid, "error": f"{len(sums)} summaries"}
    rd = os.path.dirname(sums[0])
    s = json.load(open(sums[0]))
    meta = json.load(open(os.path.join(rd, "run_metadata.json")))
    cfg = meta.get("resolved_config", {})
    rounds = [json.loads(l) for l in open(os.path.join(rd, "misfp_rounds.jsonl"))] \
        if os.path.exists(os.path.join(rd, "misfp_rounds.jsonl")) else []
    hd = os.path.join(rd, "misfp_heldout_diagnostic.json")
    hd = json.load(open(hd)) if os.path.exists(hd) else {}
    last = rounds[-1] if rounds else {}
    disp = [d.get("mean_ratio") for d in last.get("dispersion", []) if d.get("mean_ratio") is not None]
    comps = [e["output"]["mean_components_per_class"] for e in last.get("edge", [])
             if isinstance(e, dict) and "output" in e]
    fr = s["test_final_round"]
    comm = s.get("communication_MB", {})
    tot = s.get("misfp_totals", {})
    r = {"eid": eid, "split": split, "variant": variant, "extra": extra, "seed": seed, "run_dir": rd,
         "config_hash": comparable_hash(cfg), "epochs": s["epochs"],
         "mode": cfg.get("misfp_extraction", "local"),
         "client_budget": str(cfg.get("misfp_client_budget")), "edge_budget": str(cfg.get("misfp_edge_budget")),
         "accuracy": 100 * fr["accuracy"], "macro_f1": 100 * fr["macro_f1"],
         "worst10": 100 * fr["worst_decile_class_accuracy"],
         "ce_bytes": tot.get("client_to_edge_payload_bytes"), "ec_bytes": tot.get("edge_to_cloud_payload_bytes"),
         "total_comm_MB": comm.get("total_comm_MB"),
         "runtime": s.get("wall_time_s"), "peak_memory": (s.get("peak_cuda_allocated_bytes") or 0) / 2 ** 20,
         "components_per_class": float(np.mean(comps)) if comps else None,
         "between_source_ratio": float(np.mean(disp)) if disp else None,
         "nll_mixture": hd.get("mean_nll_mixture_per_dim"), "nll_single": hd.get("mean_nll_single_per_dim")}
    hp = os.path.join(rd, "history.csv")
    if os.path.exists(hp):
        h = list(csv.DictReader(open(hp)))
        r["hist_round"] = [int(x["round"]) for x in h]
        r["hist_val"] = [100 * float(x["validation_metric_value"]) for x in h]
        r["hist_cum_GiB"] = list(np.cumsum([float(x["comm_total_delta_MB"] or 0) for x in h]) / 1024)
    r.update(fidelity(rd))
    if variant == "hsfp":
        r["components_per_class"] = 1.0
    return r


UNITS = {"accuracy": "%", "macro_f1": "%", "worst10": "%", "ce_bytes": "byte", "ec_bytes": "byte",
         "total_comm_MB": "MB", "runtime": "s", "peak_memory": "MiB", "components_per_class": "count",
         "between_source_ratio": "ratio", "nll_mixture": "nat/dim", "nll_single": "nat/dim",
         "swd_mixture": "feature_unit", "swd_single": "feature_unit", "mmd_mixture": "unitless",
         "mmd_single": "unitless"}


def method_key(r):
    return f"{'hsfp' if r['variant'] == 'hsfp' else 'misfp_' + r['variant']}{('_' + r['extra']) if r['extra'] else ''}"


def write_measurements(runs):
    cols = ["method", "dataset", "partition", "seed", "round", "checkpoint", "encoder_extraction_mode",
            "client_budget_bytes", "edge_budget_bytes", "metric", "value", "unit", "source_run_id",
            "config_hash", "status"]
    rows = []
    for r in runs:
        for k, u in UNITS.items():
            v = r.get(k)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            rows.append({"method": method_key(r), "dataset": "cifar100",
                         "partition": "iid" if r["split"] == "iid" else f"dirichlet_{r['split'][3:]}",
                         "seed": r["seed"], "round": r["epochs"], "checkpoint": "final",
                         "encoder_extraction_mode": r["mode"], "client_budget_bytes": r["client_budget"],
                         "edge_budget_bytes": r["edge_budget"], "metric": k, "value": repr(float(v)),
                         "unit": u, "source_run_id": r["eid"], "config_hash": r["config_hash"],
                         "status": "validated"})
    tmp = PAPER / "results" / ".measurements.csv.tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    os.replace(tmp, PAPER / "results" / "measurements.csv")
    return len(rows)


def cell(runs, metric, scale=1.0, nd=2, **match):
    """mean +- SD over seeds for runs matching `match`; refuses to mix configurations."""
    sel = [r for r in runs if all(r.get(k) == v for k, v in match.items()) and r.get(metric) is not None]
    if not sel:
        return MISSING, None
    hashes = {r["config_hash"] for r in sel}
    if len(hashes) > 1:
        return r"\textit{mixed cfg}", None
    v = [r[metric] * scale for r in sel]
    n = len(v)
    m = statistics.mean(v)
    if n == 1:
        s = f"{m:.{nd}f}$^{{n=1}}$"
    else:
        sd = statistics.stdev(v)
        s = f"{m:.{nd}f} {{\\scriptsize $\\pm$ {sd:.{nd}f}}}" + ("" if n == len(SEEDS) else f"$^{{n={n}}}$")
    return s, (m, n)


def write_rows(name, lines):
    GEN.mkdir(exist_ok=True)
    tmp = GEN / f".{name}.tmp"
    tmp.write_text("% Generated by Sources/scripts/campaign/collect_misfp.py -- do not edit by hand\n"
                   + "\n".join(lines) + "\n")
    os.replace(tmp, GEN / f"{name}.tex")


def tables(runs):
    nat = [r for r in runs if not r["extra"]]
    split_name = {"iid": "IID", "dir0.1": r"Dir($\alpha{=}0.1$)", "dir0.7": r"Dir($\alpha{=}0.7$)"}
    L = []
    for split in ("iid", "dir0.1", "dir0.7"):
        for metric, lab in (("accuracy", "accuracy (\\%)"), ("macro_f1", "macro-F1 (\\%)")):
            cells = [MISSING] * 3 + [cell(nat, metric, split=split, variant=v)[0] for v in ("hsfp", "k1", "adaptive")]
            first = "CIFAR-100" if (split == "iid" and metric == "accuracy") else ""
            L.append(f"{first} & {split_name[split]}, {lab} & " + " & ".join(cells) + r" \\")
    write_rows("misfp_main_rows", L)

    P = "dir0.1"   # primary non-IID setting for resources / ablation / sensitivity / fidelity
    L = []
    for v, formula in (("hsfp", "$O(Cd)$"), ("k1", "$8C+R(8+2pd)$ B"), ("edge", "$8C+R(8+2pd)$ B"),
                       ("adaptive", "$8C+R(8+2pd)$ B")):
        ce = cell(nat, "ce_bytes", 1 / 2 ** 20, 2, split=P, variant=v)[0] if v != "hsfp" else r"n/a$^\ast$"
        ec = cell(nat, "ec_bytes", 1 / 2 ** 20, 2, split=P, variant=v)[0] if v != "hsfp" else r"n/a$^\ast$"
        L.append(f"{LABEL[v]} & {ce} & {ec} & {MISSING} & {cell(nat, 'total_comm_MB', 1, 1, split=P, variant=v)[0]} & "
                 f"{cell(nat, 'runtime', 1, 0, split=P, variant=v)[0]} & {cell(nat, 'peak_memory', 1, 0, split=P, variant=v)[0]} & {formula}" + r" \\")
    write_rows("misfp_resources_rows", L)

    L = [f"{LABEL[v]} & {cell(nat, 'accuracy', split=P, variant=v)[0]} & {cell(nat, 'macro_f1', split=P, variant=v)[0]} & "
         f"{cell(nat, 'components_per_class', 1, 2, split=P, variant=v)[0]}" + r" \\"
         for v in ("hsfp", "k1", "edge", "k2", "k4", "adaptive")]
    write_rows("misfp_ablation_rows", L)

    L = []
    for v in ("k2", "adaptive"):
        for extra, rule, prec in (("fp16matched", r"match K1 $\times1.0$", "fp16"),
                                  ("budget1.5x", r"match K1 $\times1.5$", "fp32"),
                                  ("budget2.0x", r"match K1 $\times2.0$", "fp32")):
            sel = [r for r in runs if r["extra"] == extra]
            L.append(f"{LABEL[v]}: {rule} & {prec} & 4 & 3 & 50/comp. & {cell(sel, 'accuracy', split=P, variant=v)[0]} & "
                     f"{cell(sel, 'macro_f1', split=P, variant=v)[0]} & {cell(sel, 'total_comm_MB', 1, 1, split=P, variant=v)[0]} & "
                     f"{cell(sel, 'runtime', 1, 0, split=P, variant=v)[0]}" + r" \\")
    for v, cap in (("k1", 1), ("k2", 2), ("k4", 4)):
        L.append(f"unbounded ({LABEL[v]}) & fp32 & {cap} & 3 & 50/comp. & {cell(nat, 'accuracy', split=P, variant=v)[0]} & "
                 f"{cell(nat, 'macro_f1', split=P, variant=v)[0]} & {cell(nat, 'total_comm_MB', 1, 1, split=P, variant=v)[0]} & "
                 f"{cell(nat, 'runtime', 1, 0, split=P, variant=v)[0]}" + r" \\")
    for extra, ms, bs in (("minsup1", "1", "50/comp."), ("", "3", "50/comp."), ("minsup10", "10", "50/comp."),
                          ("bsamp10", "3", "10/comp."), ("bsamp200", "3", "200/comp.")):
        sel = [r for r in runs if r["extra"] == extra]
        L.append(f"unbounded (Adaptive) & fp32 & 4 & {ms} & {bs} & {cell(sel, 'accuracy', split=P, variant='adaptive')[0]} & "
                 f"{cell(sel, 'macro_f1', split=P, variant='adaptive')[0]} & {cell(sel, 'total_comm_MB', 1, 1, split=P, variant='adaptive')[0]} & "
                 f"{cell(sel, 'runtime', 1, 0, split=P, variant='adaptive')[0]}" + r" \\")
    write_rows("misfp_sensitivity_rows", L)

    L = []
    for extra, mode in (("", "Local"), ("snapshot", "Shared snapshot")):
        sel = [r for r in runs if r["extra"] == extra]
        for rep, suf in (("single Gaussian (collapsed K2 packet)", "single"), ("MiSFP-K2 mixture", "mixture")):
            L.append(f"{mode} & {rep} & {cell(sel, 'nll_' + suf, 1, 3, split=P, variant='k2')[0]} & "
                     f"{cell(sel, 'swd_' + suf, 1, 3, split=P, variant='k2')[0]} & {cell(sel, 'mmd_' + suf, 1, 4, split=P, variant='k2')[0]} & "
                     f"{MISSING} & {MISSING} & {cell(sel, 'between_source_ratio', 1, 3, split=P, variant='k2')[0]}" + r" \\")
    write_rows("misfp_fidelity_rows", L)


VARIANTS = ("hsfp", "k1", "edge", "k2", "k4", "adaptive")


def _pending(ax, title):
    ax.set_title(title, fontsize=7)
    ax.text(0.5, 0.5, "results pending", ha="center", va="center", transform=ax.transAxes, fontsize=7, color="0.5")


def _meansd(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def _curve(ax, sel, xkey, label):
    """Seed-mean curve (+-1 sample SD band) over runs that have per-round history; rounds aligned by index."""
    sel = [r for r in sel if r.get("hist_val")]
    if not sel:
        return False
    L = min(len(r["hist_val"]) for r in sel)
    y = np.array([r["hist_val"][:L] for r in sel]); x = np.mean([r[xkey][:L] for r in sel], axis=0)
    mu = y.mean(0); sd = y.std(0, ddof=1) if len(sel) > 1 else np.zeros(L)
    ax.plot(x, mu, lw=1, label=f"{label} (n={len(sel)})"); ax.fill_between(x, mu - sd, mu + sd, alpha=.15)
    return True


def _bars(ax, groups, metric, title, ylabel):
    """groups: [(label, runs)] -> mean +- SD bars of `metric`; n under each bar."""
    xs, ms, sds = [], [], []
    for lab, sel in groups:
        v = [r[metric] for r in sel if r.get(metric) is not None]
        if v:
            m, d = _meansd(v); xs.append(f"{lab}\nn={len(v)}"); ms.append(m); sds.append(d)
    if not xs:
        return _pending(ax, title)
    ax.bar(xs, ms, yerr=sds, capsize=2, color="C0", alpha=.8)
    ax.set_title(title, fontsize=7); ax.set_ylabel(ylabel, fontsize=7); ax.tick_params(axis="x", labelsize=5.5)
    lo = min(m - d for m, d in zip(ms, sds)); hi = max(m + d for m, d in zip(ms, sds))
    ax.set_ylim(max(0, lo - 0.15 * (hi - lo + 1e-9) - 0.5), hi + 0.15 * (hi - lo + 1e-9) + 0.5)


def figures(runs):
    """Fig. empirical (4 panels) and Fig. diagnostics (4 panels). Only validated runs are plotted; a panel
    with no completed run says 'results pending'. Bands/error bars: 1 sample SD over completed seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not runs:
        return "no completed runs; templates kept"
    P = "dir0.1"
    nat = [r for r in runs if not r["extra"]]
    sel = lambda pool, **m: [r for r in pool if all(r.get(k) == v for k, v in m.items())]
    out = []

    # ---- empirical_results.pdf -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.6))
    a, b, c, d = axs
    ok = [_curve(a, sel(nat, split=P, variant=v), "hist_cum_GiB", LABEL[v]) for v in VARIANTS]
    if any(ok):
        a.set_title("(a) Accuracy vs. communication, Dir(0.1)", fontsize=7)
        a.set_xlabel("Cumulative communication (GiB)", fontsize=7); a.set_ylabel("Validation acc. (%)", fontsize=7)
        a.legend(fontsize=4.5)
    else:
        _pending(a, "(a) Accuracy vs. communication, Dir(0.1)")
    ok = [_curve(b, sel(nat, split=P, variant=v), "hist_round", LABEL[v]) for v in VARIANTS]
    if any(ok):
        b.set_title("(b) Convergence, Dir(0.1)", fontsize=7)
        b.set_xlabel("Communication round", fontsize=7); b.set_ylabel("Validation acc. (%)", fontsize=7)
        b.legend(fontsize=4.5)
    else:
        _pending(b, "(b) Convergence, Dir(0.1)")
    # (c) held-out-style fidelity: SWD of synthetic vs real (PCA) features, single Gaussian vs mixture
    labs, sw1, sw1e, swm, swme = [], [], [], [], []
    for v in VARIANTS[1:]:
        g = [r for r in sel(nat, split=P, variant=v) if r.get("swd_mixture") is not None]
        if g:
            labs.append(f"{LABEL[v].replace('MiSFP-', '')}\nn={len(g)}")
            m, e = _meansd([r["swd_single"] for r in g]); sw1.append(m); sw1e.append(e)
            m, e = _meansd([r["swd_mixture"] for r in g]); swm.append(m); swme.append(e)
    if labs:
        x = np.arange(len(labs))
        c.bar(x - .2, sw1, .4, yerr=sw1e, capsize=2, label="single Gaussian")
        c.bar(x + .2, swm, .4, yerr=swme, capsize=2, label="mixture")
        c.set_xticks(x); c.set_xticklabels(labs, fontsize=5.5); c.legend(fontsize=5)
        c.set_title("(c) Feature fidelity, Dir(0.1)", fontsize=7); c.set_ylabel("SWD to real features (lower better)", fontsize=6)
    else:
        _pending(c, "(c) Feature fidelity, Dir(0.1)")
    # (d) allocation / budget: components per class and payload vs accuracy (per seed)
    pts = False
    for v in VARIANTS:
        g = [r for r in sel(nat, split=P, variant=v) if r.get("components_per_class") is not None]
        if g:
            d.scatter([r["components_per_class"] for r in g], [r["accuracy"] for r in g], s=12, label=LABEL[v]); pts = True
    for ex, mk in (("fp16matched", "x"), ("budget1.5x", "+"), ("budget2.0x", "*")):
        for v in ("k2", "adaptive"):
            g = [r for r in sel(runs, split=P, variant=v, extra=ex) if r.get("components_per_class") is not None]
            if g:
                d.scatter([r["components_per_class"] for r in g], [r["accuracy"] for r in g], s=18, marker=mk,
                          label=f"{LABEL[v]} {ex}"); pts = True
    if pts:
        d.set_title("(d) Allocation vs. accuracy, Dir(0.1)", fontsize=7)
        d.set_xlabel("Edge-to-cloud components / class", fontsize=7); d.set_ylabel("Final test acc. (%)", fontsize=7)
        d.legend(fontsize=4.5)
    else:
        _pending(d, "(d) Allocation vs. accuracy, Dir(0.1)")
    for ax in axs:
        ax.tick_params(labelsize=6); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "empirical_results.pdf", bbox_inches="tight"); plt.close(fig)
    out.append("figures/empirical_results.pdf")

    # ---- diagnostic_results.pdf ------------------------------------------------------------------------
    ex = lambda e, v="adaptive": sel(runs, split=P, variant=v, extra=e)
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.4))
    _bars(axs[0], [("K1", ex("", "k1")), ("K2", ex("", "k2")), ("K4", ex("", "k4")), ("Adaptive", ex(""))],
          "accuracy", "(a) Component cap", "Final test acc. (%)")
    _bars(axs[1], [("min sup. 1", ex("minsup1")), ("3 (default)", ex("")), ("10", ex("minsup10"))],
          "accuracy", "(b) Minimum support (Adaptive)", "Final test acc. (%)")
    _bars(axs[2], [("10/comp.", ex("bsamp10")), ("50 (default)", ex("")), ("200/comp.", ex("bsamp200"))],
          "accuracy", "(c) Boundary Monte Carlo (Adaptive)", "Final test acc. (%)")
    _bars(axs[3], [(f"{LABEL[v].replace('MiSFP-', '')}\n{m}", ex(e, v)) for v in ("k1", "edge", "k2")
                   for e, m in (("", "local"), ("snapshot", "snap."))],
          "accuracy", "(d) Local vs. shared-snapshot extraction", "Final test acc. (%)")
    for ax in axs:
        ax.tick_params(axis="y", labelsize=6); ax.grid(alpha=.2, axis="y")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "diagnostic_results.pdf", bbox_inches="tight"); plt.close(fig)
    out.append("figures/diagnostic_results.pdf")
    return "wrote " + ", ".join(out)


def main():
    done = sorted(p.stem for p in MARK.glob("misfp_*.done"))
    runs, errors = [], []
    for eid in done:
        try:
            r = load_run(eid)
        except Exception as e:  # a corrupt run is reported, never silently dropped
            r = {"eid": eid, "error": f"{type(e).__name__}: {e}"}
        if r is None:
            continue
        (errors if "error" in r else runs).append(r)
    n = write_measurements(runs)
    tables(runs)
    fig = figures(runs)
    status = {"completed_runs": len(runs), "measurement_rows": n, "errors": errors, "figures": fig,
              "per_run": [{k: r.get(k) for k in ("eid", "accuracy", "macro_f1", "config_hash")} for r in runs]}
    (GEN / "collector_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({k: status[k] for k in ("completed_runs", "measurement_rows", "errors", "figures")}))


if __name__ == "__main__":
    main()
