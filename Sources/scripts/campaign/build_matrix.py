#!/usr/bin/env python3
"""Experiment matrix: every manuscript table/figure cell -> run IDs -> validated seeds -> status.

    python scripts/campaign/build_matrix.py   # writes results/campaign/EXPERIMENT_MATRIX.{csv,md}

Status vocabulary:
  complete     every required seed validated (.done after runmeta validation, or legacy ledger)
  partial      some seeds validated, the rest queued/running (cell prints its actual n)
  running      >=1 required run holds a live lock, none validated yet
  queued       in master.queue, nothing validated (fresh run; no mid-run checkpoint exists to resume)
  failed       >=1 required run has a .failed marker (needs review; never auto-retried)
  blocked      cannot run as specified (data/implementation/protocol) -- reason given
  reference    value reproduced verbatim from the accepted conference paper (no run expected)
The cell definitions mirror fill_manuscript.py (E-HSFP) and collect_misfp.py (MiSFP).
"""
import csv
import glob
import os
import re

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(SRC, "results", "journal_v1")
MARK = os.path.join(RES, ".markers")
CAMP = os.path.join(SRC, "results", "campaign")
J = "j3"

done = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(MARK, "*.done"))}
failed = {os.path.basename(p)[:-7] for p in glob.glob(os.path.join(MARK, "*.failed"))}
locks = set()
for lk in glob.glob(os.path.join(MARK, "*.lock")):
    try:
        if os.path.exists(f"/proc/{open(os.path.join(lk, 'pid')).read().strip()}"):
            locks.add(os.path.basename(lk)[:-5])
    except OSError:
        pass
queued = set()
qf = os.path.join(RES, "queue", "master.queue")
if os.path.exists(qf):
    queued = {l.split("|")[1] for l in open(qf) if l.strip() and not l.startswith("#")}
dsf = os.path.join(CAMP, "descoped.txt")
descoped = {l.strip() for l in open(dsf) if l.strip() and not l.startswith("#")} if os.path.exists(dsf) else set()
legacy = set()
lf = os.path.join(RES, "legacy", "per_run_metrics_oldmachine_20261015.csv")
if os.path.exists(lf):
    legacy = {r["experiment_id"] for r in csv.DictReader(open(lf)) if r["status"].startswith("completed")}

ROWS = []


def add(paper, item, cell, prefix, seeds, note=""):
    if prefix is None:
        ROWS.append(dict(paper=paper, item=item, cell=cell, runs="", required=0, validated=0,
                         status="blocked", note=note))
        return
    if prefix == "REF":
        ROWS.append(dict(paper=paper, item=item, cell=cell, runs="", required=0, validated=0,
                         status="reference", note=note or "accepted ECCV 2026 paper, verbatim"))
        return
    eids = [f"{prefix}_s{s}" for s in seeds]
    v = [e for e in eids if e in done]
    if any(e in failed for e in eids):
        st = "failed"
    elif len(v) == len(eids):
        st = "complete"
    elif v:
        st = "partial"
    elif any(e in locks for e in eids):
        st = "running"
    elif any(e in queued for e in eids):
        st = "queued"
    elif any(e in descoped for e in eids):
        st = "descoped (user decision 2026-10-15, cost)"
    else:
        st = "missing (not queued)"
    leg = sum(1 for e in v if e in legacy)
    ROWS.append(dict(paper=paper, item=item, cell=cell, runs=f"{prefix}_s{{{','.join(map(str, seeds))}}}",
                     required=len(eids), validated=len(v), status=st,
                     note=(note + (f"; {leg} seed(s) from legacy ledger (old machine)" if leg else "")).strip("; ")))


S5, S0, S02, S123 = range(5), [0], range(3), (1, 2, 3)
E = "E-HSFP"
for label, tpl in (("FedAvg", "main_{d}_federated"), ("FedProx", "fed_{d}_fedprox"), ("FedNova", "fed_{d}_fednova"),
                   ("HierFL", "main_{d}_hierfl"), ("SplitFed", "main_{d}_splitfl"), ("HeteroSFL", "main_{d}_hetero-sfl"),
                   ("HSFL (A)", "main_{d}_hsfl"), ("H-SFP", J + "_{d}_hsfp"), ("E-HSFP", J + "_{d}_ehsfp")):
    for d in ("cifar10", "cifar100", "ham10000"):
        add(E, "Tab journal_main_classification", f"{label} / {d}", tpl.format(d=d), S5)
for label, tpl in (("FedAvg", "main_isic2018_federated"), ("HierFL", "main_isic2018_hierfl"), ("SplitFed", "main_isic2018_splitfl"),
                   ("HeteroSFL", "main_isic2018_hetero-sfl"), ("HSFL (A)", "main_isic2018_hsfl"), ("H-SFP", "main_isic2018_hsfp"),
                   ("E-HSFP", "main_isic2018_ehsfp")):
    add(E, "Tab journal_segmentation", label, tpl, S5)
add(E, "Tab main_classification (conference)", "all baseline rows", "REF", S0)
add(E, "Tab main_classification (conference)", "E-HSFP row", None, S0,
    "no E-HSFP run exists under the conference protocol; journal-protocol values are in the matched table")
add(E, "Tab main_classification (conference)", "ImageNet-1K E-HSFP", None, S0, "blocked-data: ImageNet-1K licence-gated, not on disk")
add(E, "Tab main_classification", "FedProto/FedGen/FedDF journal rows", None, S0, "blocked-implementation: no implementation in repo")
add(E, "Tab segmentation (conference)", "E-HSFP row", None, S0, "no E-HSFP run under the conference 200-round protocol")
for p in ("0", "0.1", "0.2", "0.3", "0.5"):
    add(E, "Tab dropout_convergence", f"p={p}", f"{J}_cifar100_ehsfp" if p == "0.2" else f"{J}_dropout_cifar100_p{('0.0' if p == '0' else p).replace('.', 'p')}", S0)
for t in (0, 1, 3, 5, 10):
    add(E, "Tab staleness_convergence", f"tau={t}", f"{J}_cifar100_ehsfp" if t == 0 else f"{J}_stale_cifar100_tau{t}", S0)
add(E, "Tab staleness_convergence", "stability bound term", None, S0, "blocked: Theorem 1 constants unspecified")
for k in ("none", "timeout", "coldstart", "partial", "missedge", "combined"):
    add(E, "Tab serverless_robustness", k, f"{J}_sv_cifar100_{k}", S0)
for d in ("cifar10", "cifar100", "ham10000"):
    for part, tag in (("IID", ""), ("mild a=0.7", "noniid_a0p7_"), ("severe a=0.3", "noniid_a0p3_")):
        for m in ("hsfp", "ehsfp"):
            add(E, "Tab noniid_robustness", f"{d} {part} {m}", f"{J}_{tag}{d}_{m}", S0)
for lab, pre in (("H-SFP baseline", f"{J}_cifar100_hsfp"), ("+memory", f"{J}_abl_cifar100_hsfp_memory"),
                 ("+dropout", f"{J}_abl_cifar100_hsfp_memory_dropout"), ("+reliability", f"{J}_abl_cifar100_hsfp_memory_reliability"),
                 ("+replay consistency", f"{J}_abl_cifar100_hsfp_memory_reliability_prc"), ("Full E-HSFP", f"{J}_cifar100_ehsfp")):
    add(E, "Tab ehsfp_component_ablation", lab, pre, S0)
add(E, "Tab ehsfp_component_ablation", "+residual generator", None, S0, "blocked-implementation: generator objective unspecified (eq. 14)")
for mm in (0, 1, 5, 10, 20):
    add(E, "Tab memory_size_ablation", f"M={mm}", f"{J}_memM_cifar100_M{mm}", S0)
add(E, "Tab memory_size_ablation", "additional memory overhead column", None, S0, "collector has no mem_overhead_MB metric")
for k in ("average", "sample_count_weighted", "learnable_reliability", "learnable_reliability_memory"):
    for arm in ("clean", "drop", "stale"):
        add(E, "Tab aggregation_comparison", f"{k} {arm}", f"{J}_aggr_cifar100_{k}_{arm}", S0)
add(E, "Tab synthesis_ablation", "Diagonal Gaussian", f"{J}_synth_cifar100_diag", S0)
add(E, "Tab synthesis_ablation", "Low-rank covariance (= full E-HSFP)", f"{J}_cifar100_ehsfp", S0)
add(E, "Tab synthesis_ablation", "Residual generator (+consistency)", None, S0, "blocked-implementation")
add(E, "Tab synthesis_ablation", "Prototype fidelity column", None, S0, "no fidelity metric recorded by the runner")
for k in (20, 50, 100, 200, 500):
    add(E, "Tab scalability", f"K={k}", f"{J}_cifar100_hsfp" if k == 200 else f"{J}_scale_cifar100_K{k}_hsfp", S0)
add(E, "Tab comm_memory (conference)", "baseline rows", "REF", S0)
for fig, prefixes in (("fig_conv_curves", [f"{J}_{d}_{m}" for d in ("cifar10", "cifar100", "ham10000") for m in ("hsfp", "ehsfp")]),
                      ("fig_dropout_effect", [f"{J}_dropout_cifar100_p{p}" for p in ("0p0", "0p1", "0p3", "0p5")]),
                      ("fig_staleness_effect", [f"{J}_stale_cifar100_tau{t}" for t in (1, 3, 5, 10)]),
                      ("fig_memory_effect", [f"{J}_memM_cifar100_M{m}" for m in (0, 1, 5, 10, 20)]),
                      ("fig_drift_rounds", [f"{J}_sv_cifar100_{k}" for k in ("none", "combined")])):
    for p in prefixes:
        add(E, f"Fig {fig}", p, p, S02, "figure frozen while any contributor's per-round history is absent (legacy runs)")
add(E, "Fig fig_bound_vs_drift", "all", None, S0, "blocked: Theorem 1 constants unspecified")

M = "MiSFP"
SPL = ("iid", "dir0.1", "dir0.7")
for split in SPL:
    for v in ("hsfp", "k1", "adaptive"):
        add(M, "tab:main", f"{split} {v}", f"misfp_{split}_{v}", S123)
    add(M, "tab:main", f"{split} FedAvg/SplitFed/FedProto (dagger)", None, S0,
        "not scheduled: contextual/adapted columns; no matched-hierarchy implementation with val_from_train partitions (FedProto absent)")
for v in ("hsfp", "k1", "edge", "k2", "k4", "adaptive"):
    for split in SPL:
        add(M, "tab:ablation" if split == "dir0.1" else "fig:empirical", f"{split} {v}", f"misfp_{split}_{v}", S123)
for v in ("hsfp", "k1", "edge", "adaptive"):
    add(M, "tab:resources", f"dir0.1 {v}", f"misfp_dir0.1_{v}", S123)
add(M, "tab:resources", "model-sync column", None, S0, "runner does not separate model-sync bytes from total")
for split in SPL:
    for v in ("k2", "adaptive"):
        for ex in ("fp16matched", "budget1.5x", "budget2.0x"):
            add(M, "tab:sensitivity" if split == "dir0.1" else "sensitivity (extra partitions)", f"{split} {v} {ex}",
                f"misfp_{split}_{v}_{ex}", S123)
    for v in ("k1", "edge", "k2"):
        add(M, "tab:fidelity" if split == "dir0.1" else "fidelity (extra partitions)", f"{split} {v} shared snapshot",
            f"misfp_{split}_{v}_snapshot", S123)
for ex in ("minsup1", "minsup10", "bsamp10", "bsamp200"):
    add(M, "tab:sensitivity / fig:diagnostics", f"dir0.1 adaptive {ex}", f"misfp_dir0.1_adaptive_{ex}", S123,
        "sweep values chosen by the campaign (author confirmation pending)")
add(M, "tab:fidelity", "mode coverage / minority recall", None, S0, "needs training-feature artifacts the runner does not persist")
add(M, "tab:setup", "CIFAR-10 / HAM10000 rows", None, S0, "not integrated in the MiSFP runner (manuscript marks them planned)")

os.makedirs(CAMP, exist_ok=True)
cols = ["paper", "item", "cell", "runs", "required", "validated", "status", "note"]
with open(os.path.join(CAMP, "EXPERIMENT_MATRIX.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(ROWS)
summ = {}
for r in ROWS:
    k = (r["paper"], r["status"])
    summ[k] = summ.get(k, 0) + 1
md = ["# Experiment matrix (generated by scripts/campaign/build_matrix.py)", "",
      "| Paper | Status | Cells |", "|---|---|---|"] + [f"| {p} | {s} | {n} |" for (p, s), n in sorted(summ.items())]
md += ["", "| Paper | Table/figure | Cell | Runs | Validated/required | Status | Note |", "|---|---|---|---|---|---|---|"]
md += [f"| {r['paper']} | {r['item']} | {r['cell']} | `{r['runs']}` | {r['validated']}/{r['required']} | {r['status']} | {r['note']} |"
       for r in ROWS]
tmp = os.path.join(CAMP, ".matrix.tmp")
open(tmp, "w").write("\n".join(md) + "\n")
os.replace(tmp, os.path.join(CAMP, "EXPERIMENT_MATRIX.md"))
print({f"{p}:{s}": n for (p, s), n in sorted(summ.items())})
