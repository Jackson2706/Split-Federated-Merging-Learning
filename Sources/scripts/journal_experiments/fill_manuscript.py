#!/usr/bin/env python3
"""Fill the E-HSFP journal manuscript tables from recorded runs.

    python scripts/journal_experiments/fill_manuscript.py            # refresh
    python scripts/journal_experiments/fill_manuscript.py --dry-run  # print only

Reads artifacts/per_run_metrics.csv (written by collect_journal_v1.py; run that
first) and rewrites the row fragments under
Submissions/E-HSFP Journal/generated/*.tex, which experiments.tex pulls in via
\\tabrows. Rules:
  * a cell is a mean {\\scriptsize $\\pm$ sample SD (ddof=1)} over the completed
    seeds of that cell; fewer than 5 seeds is marked with a superscript n;
  * a cell with no completed run stays \\pending -- nothing is estimated;
  * every completed seed is used; cells containing a seed flagged
    completed-suspect (collapsed / constant predictor) carry a section sign;
  * journal-protocol values only; conference values live in Tables I/II/VII.
Run-ID scheme for the corrected method (journal protocol v2): j2_<family>_..._s<seed>.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
PAPER = SRC.parent / "Submissions" / "E-HSFP Journal"
GEN = PAPER / "generated"
METRICS = SRC / "artifacts" / "per_run_metrics.csv"
SEEDS = range(5)
# Run-ID prefix of the corrected method (protocol v3). Superseded v2 runs use j2_.
JP = "j3"
PENDING = r"\pending"


def load():
    runs = {}
    if METRICS.exists():
        for r in csv.DictReader(open(METRICS)):
            runs[r["experiment_id"]] = r
    return runs


RUNS = load()
EXCLUDED = []
SUSPECT = set()


# Seeds a cell aggregates. Main tables: 0-4. Sweep tables (user decision
# 2026-10-06): seed 0 only, including their reference rows, so no row mixes a
# single seed with a five-seed mean.
ACTIVE_SEEDS = list(SEEDS)


def vals(prefix, key):
    out = []
    for s in ACTIVE_SEEDS:
        r = RUNS.get(f"{prefix}_s{s}")
        if not r:
            continue
        if not r["status"].startswith("completed-"):
            continue
        if r["status"] != "completed-valid":
            # Kept (no proven bug -> excluding seeds would be selection);
            # the cell is marked with a section sign instead.
            EXCLUDED.append(f"{prefix}_s{s} ({r['status']}: {r.get('notes', '')})")
            SUSPECT.add(prefix)
        try:
            v = float(r.get(key, ""))
        except ValueError:
            continue
        if math.isfinite(v):
            out.append(v)
    return out


TIMING_KEYS = ("runtime_s",)


def legacy_host(prefix):
    """True if any contributing run was measured on the previous host (legacy ledger)."""
    return any("legacy_ledger" in (RUNS.get(f"{prefix}_s{s}") or {}).get("notes", "") for s in ACTIVE_SEEDS)


_PP = Path(__file__).resolve().parents[2] / "results" / "journal_v1" / "partial_runs.json"
PARTIAL = json.load(open(_PP)) if _PP.exists() else {}   # runs stopped at the deadline (deadline_cutoff.sh)


def partial(prefix, key):
    """(value, round) for a run stopped at the deadline, when the cell has no completed run."""
    if len(ACTIVE_SEEDS) != 1 or key not in ("final_acc", "final_f1"):
        return None
    p = PARTIAL.get(f"{prefix}_s{ACTIVE_SEEDS[0]}")
    return (p[key], p["round"]) if p else None


def cell(prefix, key, scale=1.0, nd=2):
    if prefix is None:
        return PENDING
    v = vals(prefix, key)
    if not v and partial(prefix, key):
        x, k = partial(prefix, key)
        return f"{x * scale:.{nd}f}" + r"$^{r=" + str(k) + "}$"
    if not v:
        # a completed run that never met the convergence rule is a measured outcome, not a missing value
        done = [RUNS.get(f"{prefix}_s{s}") for s in ACTIVE_SEEDS]
        done = [r for r in done if r and r["status"].startswith("completed-")]
        if key == "conv_rounds" and done and all("not_converged" in r.get("notes", "") for r in done):
            return "n.c."
        return PENDING
    if key in TIMING_KEYS and legacy_host(prefix):
        # wall-clock from the previous host (RTX 3080 Ti, 4 concurrent lanes): not comparable with
        # timings taken on the RTX 6000 Ada host; flagged instead of silently mixed
        return cell_value(v, scale, nd, prefix) + r"$^\star$"
    return cell_value(v, scale, nd, prefix)


def cell_value(v, scale, nd, prefix):
    m = np.mean(v) * scale
    s = f"{m:.{nd}f}"
    if len(v) > 1:
        s += r" {\scriptsize $\pm$ " + f"{np.std(v, ddof=1) * scale:.{nd}f}" + "}"
    if len(v) < len(ACTIVE_SEEDS):
        s += r"$^{n=" + str(len(v)) + "}$"
    if prefix in SUSPECT:
        s += r"$^\S$"
    return s


def na(text="n/a"):
    return text


ACC, F1 = ("final_acc", 100, 2), ("final_f1", 100, 2)
CONV, STAB, DRIFT = ("conv_rounds", 1, 1), ("stability_final", 1e4, 2), ("drift_mean", 1, 3)
COMM = ("comm_MB", 1 / 1024, 1)


def row(label, prefix, cols):
    return " & ".join([label] + [c if isinstance(c, str) else cell(prefix, *c) for c in cols]) + r" \\"


def write(name, rows, dry):
    text = "% Generated by Sources/scripts/journal_experiments/fill_manuscript.py -- do not edit by hand\n" + "\n".join(rows) + "\n"
    if dry:
        print(f"--- {name}\n{text}")
    else:
        (GEN / f"{name}.tex").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    global ACTIVE_SEEDS
    GEN.mkdir(parents=True, exist_ok=True)
    ACTIVE_SEEDS = [0]  # ---- sweep tables (single seed) ----
    E = f"{JP}_cifar100_ehsfp"   # full E-HSFP, journal protocol v2, CIFAR-100 IID (reference cell)

    write("dropout_convergence", [
        # run IDs spell p=0 as p0p0 (j3_dropout_cifar100_p0p0_s0)
        row(f"$p={p}$", E if p == "0.2" else f"{JP}_dropout_cifar100_p{('0.0' if p == '0' else p).replace('.', 'p')}", [ACC, F1, CONV, STAB, DRIFT])
        for p in ("0", "0.1", "0.2", "0.3", "0.5")], a.dry_run)
    write("staleness_convergence", [
        row(rf"$\tau={t}$", E if t == 0 else f"{JP}_stale_cifar100_tau{t}", [ACC, F1, CONV, DRIFT, na()])
        for t in (0, 1, 3, 5, 10)], a.dry_run)

    sv = [("No event loss", "none"), ("Function timeout", "timeout"), ("Cold-start delay", "coldstart"),
          ("Partial edge execution", "partial"), ("Missing edge update", "missedge"),
          ("Combined serverless stress", "combined")]
    base = {s: RUNS.get(f"{JP}_sv_cifar100_none_s{s}") for s in ACTIVE_SEEDS}
    rows = []
    for label, k in sv:
        pre = f"{JP}_sv_cifar100_{k}"
        gaps = []
        for s in ACTIVE_SEEDS:
            r, b = RUNS.get(f"{pre}_s{s}"), base.get(s)
            if r and b and r["status"].startswith("completed-") and b["status"].startswith("completed-"):
                gaps.append((float(b["final_acc"]) - float(r["final_acc"])) * 100)
        gap = "0 (ref.)" if k == "none" else (PENDING if not gaps else
              f"{np.mean(gaps):.2f}" + (r" {\scriptsize $\pm$ " + f"{np.std(gaps, ddof=1):.2f}" + "}" if len(gaps) > 1 else "")
              + (r"$^{n=" + str(len(gaps)) + "}$" if len(gaps) < 5 else ""))
        rows.append(row(label, pre, [ACC, F1, DRIFT, CONV, gap]))
    write("serverless_robustness", rows, a.dry_run)

    rows = []
    for d, dn in (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("ham10000", "HAM10000")):
        for part, tag in (("IID", ""), ("Non-IID mild", "noniid_a0p7_"), ("Non-IID severe", "noniid_a0p3_")):
            for m, ml in (("hsfp", r"\hsfp"), ("ehsfp", r"\ehsfp")):
                rows.append(" & ".join([dn, part, ml]) + " & " +
                            " & ".join(cell(f"{JP}_{tag}{d}_{m}", *c) for c in (ACC, F1, CONV, DRIFT, COMM)) + r" \\")
    write("noniid_robustness", rows, a.dry_run)

    write("ehsfp_component_ablation", [
        row("H-SFP baseline", f"{JP}_cifar100_hsfp", [ACC, F1, COMM, CONV, STAB]),
        row(r"$+$ Episodic memory", f"{JP}_abl_cifar100_hsfp_memory", [ACC, F1, COMM, CONV, STAB]),
        row(r"$+$ Prototype dropout", f"{JP}_abl_cifar100_hsfp_memory_dropout", [ACC, F1, COMM, CONV, STAB]),
        row(r"$+$ Reliability-aware aggregation", f"{JP}_abl_cifar100_hsfp_memory_reliability", [ACC, F1, COMM, CONV, STAB]),
        row(r"$+$ Replay consistency", f"{JP}_abl_cifar100_hsfp_memory_reliability_prc", [ACC, F1, COMM, CONV, STAB]),
        row(r"$+$ Residual prototype generator", None, [ACC, F1, COMM, CONV, STAB]),
        row("Full E-HSFP", E, [ACC, F1, COMM, CONV, STAB]),
    ], a.dry_run)

    rows = []
    for mm in (0, 1, 5, 10, 20):
        rows.append(row(str(mm), f"{JP}_memM_cifar100_M{mm}", [ACC, F1, DRIFT, CONV, ("mem_overhead_MB", 1, 2)]))
    write("memory_size_ablation", rows, a.dry_run)
    # Values quoted in the prose (sections/experiments.tex) for runs that may finish after the text was
    # written; the macros expand to \pending until the run is collected.
    def pv(prefix):
        if not vals(prefix, "final_acc") and partial(prefix, "final_acc"):
            x, k = partial(prefix, "final_acc")
            return f"{100 * x:.2f}\\% (round {k} of 60; run stopped at the deadline)"
        c = cell(prefix, *ACC)
        return c if c == PENDING else c + r"\%"
    write("prose_values", [
        r"\providecommand{\PVcHundredMildEhsfp}{" + pv(f"{JP}_noniid_a0p7_cifar100_ehsfp") + "}",
        r"\providecommand{\PVcHundredSevereEhsfp}{" + pv(f"{JP}_noniid_a0p3_cifar100_ehsfp") + "}",
        r"\providecommand{\PVmemTwentyAcc}{" + pv(f"{JP}_memM_cifar100_M20") + "}",
        r"\providecommand{\PVpartialNote}{" + (r" $^{r=k}$: run stopped at the submission deadline; accuracy and F1"
                                               r" at round $k$ of 60, other metrics not reported." if any(
            e not in RUNS or not RUNS[e]["status"].startswith("completed-") for e in PARTIAL) else "") + "}",
    ], a.dry_run)

    rows = []
    for label, k in (("Average", "average"), ("Sample-count weighted", "sample_count_weighted"),
                     ("Reliability-aware", "learnable_reliability"), ("Reliability-aware + memory", "learnable_reliability_memory")):
        p = f"{JP}_aggr_cifar100_{k}"
        rows.append(" & ".join([label, cell(p + "_clean", *ACC), cell(p + "_clean", *F1), cell(p + "_drop", *ACC),
                                cell(p + "_stale", *ACC), cell(p + "_clean", *COMM)]) + r" \\")
    write("aggregation_comparison", rows, a.dry_run)

    write("synthesis_ablation", [
        row("Diagonal Gaussian", f"{JP}_synth_cifar100_diag", [ACC, F1, ("fidelity", 1, 3), COMM, ("runtime_s", 1 / 3600, 2)]),
        row("Low-rank covariance", E, [ACC, F1, ("fidelity", 1, 3), COMM, ("runtime_s", 1 / 3600, 2)]),
        row("Residual generator", None, [ACC, F1, PENDING, PENDING, PENDING]),
        row("Residual generator + consistency loss", None, [ACC, F1, PENDING, PENDING, PENDING]),
    ], a.dry_run)

    write("scalability", [
        row(str(k), f"{JP}_cifar100_hsfp" if k == 200 else f"{JP}_scale_cifar100_K{k}_hsfp",
            [ACC, F1, COMM, ("client_gpu_peak_MB", 1, 0), ("runtime_s", 1 / 3600, 2), CONV])
        for k in (20, 50, 100, 200, 500)], a.dry_run)

    ACTIVE_SEEDS = list(SEEDS)  # ---- main tables (five seeds) ----
    # Matched journal-protocol main comparison (companion to conference Table I).
    rows = []
    for label, tpl in (("FedAvg", "main_{d}_federated"), ("FedProx", "fed_{d}_fedprox"), ("FedNova", "fed_{d}_fednova"),
                       ("HierFL$^\\ddagger$", "main_{d}_hierfl"), ("SplitFed", "main_{d}_splitfl"),
                       ("HeteroSFL", "main_{d}_hetero-sfl"), ("HSFL (A)", "main_{d}_hsfl"),
                       (r"\hsfp", JP + "_{d}_hsfp"), (r"\ehsfp", JP + "_{d}_ehsfp")):
        rows.append(" & ".join([label] + [cell(tpl.format(d=d), *ACC) for d in ("cifar10", "cifar100", "ham10000")]) + r" \\")
    write("journal_main_classification", rows, a.dry_run)
    rows = []
    for label, tpl in (("FedAvg", "main_isic2018_federated"), ("HierFL$^\\ddagger$", "main_isic2018_hierfl"),
                       ("SplitFed", "main_isic2018_splitfl"), ("HeteroSFL", "main_isic2018_hetero-sfl"),
                       ("HSFL (A)", "main_isic2018_hsfl"), (r"\hsfp", "main_isic2018_hsfp"), (r"\ehsfp", "main_isic2018_ehsfp")):
        rows.append(" & ".join([label, cell(tpl, "test_iou", 100, 2), cell(tpl, "test_dice", 100, 2),
                                cell(tpl, "runtime_s", 1 / 3600, 2)]) + r" \\")
    write("journal_segmentation", rows, a.dry_run)

    filled = sum(1 for f in GEN.glob("*.tex") for line in open(f) for c in line.split("&") if r"\pending" not in c and line.strip() and not line.startswith("%"))
    print(f"fragments in {GEN}; suspect seeds included and marked: {len(set(EXCLUDED))}")
    for e in sorted(set(EXCLUDED)):
        print("  suspect:", e)


if __name__ == "__main__":
    main()
