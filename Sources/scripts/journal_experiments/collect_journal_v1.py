#!/usr/bin/env python3
"""Collect the frozen journal_v1 campaign into ledgers, tables and figures.

    python scripts/journal_experiments/collect_journal_v1.py

Reads ONLY recorded artifacts (manifest, markers, locks, queue files, logs,
metrics.json / history.csv / prediction NPZs / diagnostics CSVs). Nothing is
interpolated, extrapolated or copied between cells. Writes:

  manifests/run_status.csv          one row per run (completed / running / queued / ...)
  artifacts/per_run_metrics.csv     extracted metrics per completed run
  artifacts/coverage_report.csv     every manuscript cell -> runs, seed count, status
  artifacts/tables/table_*.{csv,tex}
  artifacts/figures/fig*.{png,pdf}

Metric rules (see reports/completion_audit.md):
* Classification headline = FINAL-ROUND (round 60) test accuracy / macro-F1 for
  every method. The H-SFP family's own "best validation" pick uses the test
  split as validation (get_data.py returns test_dataset twice), so it is
  reported only as a secondary, explicitly test-selected column.
* Macro-F1 is recomputed from last_round_predictions.npz when present;
  FedAvg/FedProx/FedNova/HierFL never persisted predictions or F1 -> blank.
* Segmentation = each runner's own reported test IoU/Dice with its selection
  rule recorded (runners differ; documented, not harmonised by guesswork).
* Aggregates: mean and sample SD (ddof=1) over seeds; n is always shown.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ehsfp.research_metrics import rounds_to_convergence, trailing_window_stability  # noqa: E402

RES = ROOT / "results" / "journal_v1"
LOGS = RES / "logs"
MARK = RES / ".markers"
QDIR = RES / "queue"
OUT_ROOTS = [
    Path(os.environ.get("OUT_ROOT", "/media/jackson/Data/ehsfp_campaign/outputs_journal_v1")),
    ROOT / "outputs" / "journal_v1",
]
SEG_RUN_ROOTS = [
    ROOT / "segmentation" / "H-SFP" / "Figure" / "data" / "runs",
    Path("/media/jackson/Data/ehsfp_campaign/seg_hsfp_runs"),
]
ART = ROOT / "artifacts"
TAB = ART / "tables"
FIG = ART / "figures"
SEEDS = [0, 1, 2, 3, 4]
BUDGET = 60
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "ham10000": 7}
CODE_REV = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


# ----------------------------------------------------------------- utilities
def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def mean_sd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else None
    return m, sd, len(v)


def fmt(m, sd, n, scale=1.0, nd=2, need=5):
    if m is None:
        return "—"
    s = f"{m * scale:.{nd}f}"
    if sd is not None:
        s += f" ± {sd * scale:.{nd}f}"
    if n < need:
        s += f" (n={n}/{need})"
    return s


def read_text(p):
    try:
        return Path(p).read_text(errors="replace")
    except OSError:
        return ""


def relocate(path_str):
    """Map a recorded run_dir to where it exists now (artifacts were copied
    to the data disk on 2026-10-02; originals were left in place)."""
    p = Path(path_str)
    if p.exists():
        return p
    s = str(p)
    for old in (str(ROOT / "outputs" / "journal_v1"),):
        if s.startswith(old):
            for r in OUT_ROOTS:
                q = Path(s.replace(old, str(r), 1))
                if q.exists():
                    return q
    for old in (str(SEG_RUN_ROOTS[0]),):
        if s.startswith(old):
            q = Path(s.replace(old, str(SEG_RUN_ROOTS[1]), 1))
            if q.exists():
                return q
    return None


# --------------------------------------------------------------- run universe
def load_manifest():
    rows = {}
    mf = RES / "manifest.csv"
    if mf.exists():
        for r in csv.DictReader(open(mf)):
            rows[r["experiment_id"]] = r  # latest row wins
    return rows


def load_queues():
    q = {}
    for f in sorted(QDIR.glob("*.queue")):
        lane = f.stem
        for line in open(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            F = line.split("|")
            q.setdefault(F[1], {"lane": lane, "stage": F[0], "task": F[2], "dataset": F[3],
                                "method": F[4], "cfg": F[5], "seed": F[6], "ablation": F[7],
                                "extra": " ".join(x for x in F[8:] if x)})
    return q


def lock_state(eid):
    d = MARK / f"{eid}.lock"
    if not d.is_dir():
        return None
    pid = read_text(d / "pid").strip()
    alive = False
    if pid.isdigit():
        try:
            os.kill(int(pid), 0)
            alive = True
        except OSError:
            alive = False
    return {"pid": pid, "alive": alive, "started": read_text(d / "started").strip()}


# ------------------------------------------------------------ metric extract
INTEG = re.compile(r"\[Integrity\] architecture_id=\S+ run_dir=(\S+)")
HASH = re.compile(r"\[Integrity\] resolved_config_hash=(\S+)")
RUNTIME = re.compile(r"Total Run Time: ([0-9.]+)s")
COMM = re.compile(r"total_comm_MB: ([0-9.]+) MB")
EPOCH = re.compile(r"EPOCH (\d+)/(\d+)")


def npz_metrics(p):
    if p is None or not Path(p).exists():
        return None, None
    from sklearn.metrics import accuracy_score, f1_score
    d = np.load(p)
    y, yhat = d["labels"], d["predictions"]
    return float(accuracy_score(y, yhat)), float(f1_score(y, yhat, average="macro", zero_division=0))


def find_baseline_dir(eid):
    for r in OUT_ROOTS:
        d = r / eid
        if d.is_dir():
            return d
    return None


def extract(eid, task, method, log_text):
    """Return a dict of recorded metrics for one finished (or partial) run."""
    m = {"run_dir": "", "config_hash": "", "notes": []}
    h = HASH.search(log_text)
    if h:
        m["config_hash"] = h.group(1)
    rt = RUNTIME.findall(log_text)
    m["runtime_s"] = fnum(rt[-1]) if rt else None
    cm = COMM.findall(log_text)
    m["comm_MB"] = fnum(cm[-1]) if cm else None
    curve = None
    run_dir = None
    if method == "h-sfp":
        g = INTEG.search(log_text)
        if g:
            run_dir = relocate(g.group(1))
    else:
        run_dir = find_baseline_dir(eid)
    m["run_dir"] = str(run_dir) if run_dir else ""

    if task == "classification" and method == "h-sfp" and run_dir:
        mj = run_dir / "metrics.json"
        if mj.exists():
            j = json.load(open(mj))
            acc = j.get("validation_accuracy") or []
            f1 = j.get("validation_f1") or []
            curve = [fnum(a) for a in acc]
            m["final_acc"] = fnum(acc[-1]) if acc else None
            m["final_f1"] = fnum(f1[-1]) if f1 else None
            m["best_acc_testselected"] = max(a for a in curve if a is not None) if curve else None
            m["comm_MB"] = fnum(j.get("total_comm_MB")) or m["comm_MB"]
        a, f = npz_metrics(run_dir / "last_round_predictions.npz")
        if a is not None:
            if m.get("final_acc") is not None and abs(a - m["final_acc"]) > 1e-4:
                m["notes"].append(f"npz_acc_mismatch({a:.4f})")
            m["final_acc"], m["final_f1"] = a, f
    elif task == "classification" and run_dir:
        js = [p for p in run_dir.glob("*.json")]
        j = json.load(open(js[0])) if js else {}
        a, f = npz_metrics(run_dir / "last_round_predictions.npz")
        if a is not None:
            m["final_acc"], m["final_f1"] = a, f
        elif j.get("train_accuracy") and len(j["train_accuracy"]) >= BUDGET:
            # Federated/HierFL: 'train_accuracy' is the per-round TEST accuracy
            # (classification/Federated/runner.py:121-122); F1 never recorded.
            m["final_acc"] = fnum(j["train_accuracy"][-1])
            m["final_f1"] = None
        if j.get("train_accuracy") and len(j["train_accuracy"]) >= BUDGET:
            curve = [fnum(a) for a in j["train_accuracy"]]
        m["best_acc_testselected"] = fnum(j.get("best_val_top1"))
        m["comm_MB"] = fnum(j.get("total_comm_MB")) or m["comm_MB"]
        m["runtime_s"] = fnum(j.get("runtime_s")) or m["runtime_s"]
        m["peak_vram_MB"] = fnum(j.get("peak_vram_MB"))
    elif task == "segmentation":
        if method == "h-sfp" and run_dir and (run_dir / "metrics.json").exists():
            j = json.load(open(run_dir / "metrics.json"))
            sel = j.get("selected_checkpoint", {})
            m["test_iou"], m["test_dice"] = fnum(sel.get("test_iou")), fnum(sel.get("test_dice"))
            lr = j.get("last_round_validation", {})
            m["final_iou"], m["final_dice"] = fnum(lr.get("iou")), fnum(lr.get("dice"))
            curve = [fnum(x) for x in j.get("validation_iou", [])]
            m["selection_rule"] = "best-round on evaluation split (= test split)"
            m["comm_MB"] = fnum(j.get("total_comm_MB")) or m["comm_MB"]
        else:
            pats = [
                (r"Best-checkpoint Test IoU: ([0-9.]+)%\s+Test Dice: ([0-9.]+)%", 0.01, "best-round checkpoint; validation split = test split"),
                (r"Best-checkpoint Test IoU: ([0-9.]+)%\s+Dice: ([0-9.]+)%", 0.01, "best-round checkpoint; validation split = test split"),
                (r"Best-checkpoint Test IoU: (0\.[0-9]+)\s+Dice: (0\.[0-9]+)", 1.0, "best-round checkpoint; validation split = test split"),
                (r"Final Test IoU: ([0-9.]+)%\s+Dice: ([0-9.]+)%", 0.01, "final round, test split"),
                (r"\|---- Test IoU: ([0-9.]+)%\s*\n\|---- Test Dice: ([0-9.]+)%", 0.01, "best-validation checkpoint, test split"),
            ]
            for pat, sc, rule in pats:
                g = re.findall(pat, log_text)
                if g:
                    m["test_iou"], m["test_dice"] = float(g[-1][0]) * sc, float(g[-1][1]) * sc
                    m["selection_rule"] = rule
                    break
            br = re.findall(r"Best Validation IoU: [0-9.]+%?\s+Dice: [0-9.]+%?\s+Round: (\d+)", log_text)
            if br:
                m["best_round"] = int(br[-1])
    # per-round curve-derived metrics
    if curve and all(c is not None for c in curve):
        m["rounds_done"] = len(curve)
        rc = rounds_to_convergence(curve)
        m["conv_rounds"] = rc if isinstance(rc, int) else None
        if not isinstance(rc, int):
            m["notes"].append("not_converged")
        st = trailing_window_stability(curve, 5)
        m["stability_final"] = st[-1]
        m["curve"] = curve
    # diagnostics (new instrumented runs only)
    if run_dir and (run_dir / "diagnostics" / "episodic_diagnostics.csv").exists():
        rows = list(csv.DictReader(open(run_dir / "diagnostics" / "episodic_diagnostics.csv")))
        er = [r for r in rows if r["tier"] == "edge" and fnum(r["drift_w2"]) is not None]
        by_round = defaultdict(list)
        for r in er:
            by_round[int(r["round"])].append(float(r["drift_w2"]))
        if by_round:
            m["drift_curve"] = [float(np.mean(by_round[k])) for k in sorted(by_round)]
            m["drift_mean"] = float(np.mean(m["drift_curve"]))
        m["mean_packet_age"] = float(np.mean([fnum(r["mean_packet_age"]) or 0 for r in er])) if er else None
    if run_dir and (run_dir / "history.csv").exists():
        hist = list(csv.DictReader(open(run_dir / "history.csv")))
        m["rounds_done"] = m.get("rounds_done") or len(hist)
        if hist:
            last = hist[-1]
            for t in ("client", "edge", "cloud"):
                vals = [fnum(r.get(f"{t}_gpu_peak_allocated_bytes")) for r in hist]
                vals = [v for v in vals if v is not None]
                m[f"{t}_gpu_peak_MB"] = max(vals) / 2 ** 20 if vals else None
            m["client_mem_records"] = fnum(last.get("client_memory_records"))
            m["edge_mem_records"] = fnum(last.get("edge_memory_records"))
    # degeneracy flags
    nc = NUM_CLASSES.get(eid.split("_")[1] if eid.count("_") > 1 else "", None)
    if m.get("final_acc") is not None and nc and m["final_acc"] <= 1.5 / nc:
        m["notes"].append("near_chance_accuracy")
    if m.get("best_round") == 1:
        m["notes"].append("best_at_round_1(no_learning_after_init)")
    return m


# ------------------------------------------------------------------ eid parse
def describe(eid, q=None, man=None):
    src = q or {}
    if man:
        src = {"stage": man["stage"], "task": man["task"], "dataset": man["dataset"],
               "method": man["method"], "seed": man["seed"], "ablation": man["ablation"],
               "extra": man.get("overrides", ""), "cfg": man["cfg"]}
    seed = int(src.get("seed", re.search(r"_s(\d+)$", eid).group(1) if re.search(r"_s(\d+)$", eid) else -1))
    extra = src.get("extra", "")
    part = "IID"
    if "dirichlet_alpha=0.7" in extra or "_a0p7_" in eid:
        part = "Dirichlet(0.7)"
    elif "dirichlet_alpha=0.3" in extra or "_a0p3_" in eid:
        part = "Dirichlet(0.3)"
    protocol = "frozen_journal_v1 (60 rounds, seeds 0-4)"
    budget = BUDGET
    if eid.startswith("comm200"):
        protocol, budget = "comm200 (200 rounds, single seed, Table XIII only)", 200
    if eid.startswith("parity"):
        protocol = "parity gate (60 rounds, seed 0) — duplicate of main_*_hsfp_s0"
    return {"stage": src.get("stage", ""), "task": src.get("task", ""), "dataset": src.get("dataset", ""),
            "method": src.get("method", ""), "variant": (src.get("ablation") or "none") + (f" [{extra}]" if extra and extra != f"epochs={BUDGET}" else ""),
            "partition": part, "seed": seed, "protocol": protocol, "budget": budget,
            "cfg": src.get("cfg", "")}


# -------------------------------------------------------------- cell registry
def cells():
    """Manuscript cell -> eid template (seed substituted). Reuse is explicit."""
    C = []
    t1_methods = [("FedAvg", "main_{d}_federated"), ("FedProx", "fed_{d}_fedprox"),
                  ("FedNova", "fed_{d}_fednova"), ("HierFL", "main_{d}_hierfl"),
                  ("SplitFed", "main_{d}_splitfl"), ("HeteroSFL", "main_{d}_hetero-sfl"),
                  ("HSFL (A)", "main_{d}_hsfl"), ("H-SFP (A)", "main_{d}_hsfp"),
                  ("E-HSFP", "main_{d}_ehsfp")]
    for row, tpl in t1_methods:
        for d in ("cifar10", "cifar100", "ham10000"):
            C.append(("I", row, d, tpl.format(d=d), "cls", None))
        C.append(("I", row, "imagenet1k", None, "cls", "blocked-data: data/ImageNet -> /media/jackson/Data/HSF-P/ImageNet is empty"))
    for row in ("FedProto", "FedGen", "FedDF"):
        for d in ("cifar10", "cifar100", "ham10000", "imagenet1k"):
            C.append(("I", row, d, None, "cls", "blocked-implementation: no FedProto/FedGen/FedDF implementation exists in this repo"))
    for row, tpl in [("FedAvg", "main_isic2018_federated"), ("HierFL", "main_isic2018_hierfl"),
                     ("SplitFed", "main_isic2018_splitfl"), ("HeteroSFL", "main_isic2018_hetero-sfl"),
                     ("HSFL (A)", "main_isic2018_hsfl"), ("H-SFP (A)", "main_isic2018_hsfp"),
                     ("E-HSFP", "main_isic2018_ehsfp")]:
        C.append(("II", row, "isic2018", tpl, "seg", None))
    for p, tpl in [("p=0", "dropout_cifar100_p0p0"), ("p=0.1", "dropout_cifar100_p0p1"),
                   ("p=0.2", "main_cifar100_ehsfp"), ("p=0.3", "dropout_cifar100_p0p3"),
                   ("p=0.5", "dropout_cifar100_p0p5")]:
        C.append(("III", p, "cifar100", tpl, "cls", None))
    for t, tpl in [("tau=0", "main_cifar100_ehsfp"), ("tau=1", "stale_cifar100_tau1"),
                   ("tau=3", "stale_cifar100_tau3"), ("tau=5", "stale_cifar100_tau5"),
                   ("tau=10", "stale_cifar100_tau10")]:
        C.append(("IV", t, "cifar100", tpl, "cls", None))
    for r, k in [("No event loss", "none"), ("Function timeout", "timeout"),
                 ("Cold-start delay", "coldstart"), ("Partial edge execution", "partial"),
                 ("Missing edge update", "missedge"), ("Combined serverless stress", "combined")]:
        C.append(("V", r, "cifar100", f"sv_cifar100_{k}", "cls", None))
    for d in ("cifar10", "cifar100", "ham10000"):
        for part, pre in [("IID", "main_{d}_{m}"), ("Non-IID mild", "noniid_a0p7_{d}_{m}"),
                          ("Non-IID severe", "noniid_a0p3_{d}_{m}")]:
            for m, lab in (("hsfp", "H-SFP"), ("ehsfp", "E-HSFP")):
                C.append(("VI", f"{d} | {part} | {lab}", d, pre.format(d=d, m=m), "cls", None))
    for r, tpl in [("H-SFP baseline", "main_cifar100_hsfp"), ("+ Episodic memory", "abl_cifar100_hsfp_memory"),
                   ("+ Prototype dropout", "abl_cifar100_hsfp_memory_dropout"),
                   ("+ Reliability-aware aggregation", "abl_cifar100_hsfp_memory_reliability"),
                   ("+ Replay consistency", "abl_cifar100_hsfp_memory_reliability_prc"),
                   ("+ Residual prototype generator", None), ("Full E-HSFP", "main_cifar100_ehsfp")]:
        C.append(("VIII", r, "cifar100", tpl, "cls",
                  None if tpl else "blocked-implementation: residual generator hard-coded to None in hierarchy.py; training objective for G unspecified"))
    for mm in (0, 1, 5, 10, 20):
        C.append(("IX", f"M={mm}", "cifar100", f"memM_cifar100_M{mm}", "cls", None))
    for r, k in [("Average", "average"), ("Sample-count weighted", "sample_count_weighted"),
                 ("Reliability-aware", "learnable_reliability"), ("Reliability-aware + memory", "learnable_reliability_memory")]:
        for cond in ("clean", "drop", "stale"):
            C.append(("X", f"{r} | {cond}", "cifar100", f"aggr_cifar100_{k}_{cond}", "cls", None))
    C.append(("XI", "Diagonal Gaussian", "cifar100", "main_cifar100_ehsfp", "cls", None))
    for r in ("Low-rank covariance", "Residual generator", "Residual generator + consistency loss"):
        C.append(("XI", r, "cifar100", None, "cls",
                  "blocked-implementation/protocol: not in the training pipeline (low-rank packet rank r and generator objective unspecified; dropout_consistency_loss never invoked)"))
    for k in (20, 50, 100, 200, 500):
        C.append(("XII", f"K={k}", "cifar100", "main_cifar100_hsfp" if k == 200 else f"scale_cifar100_K{k}_hsfp", "cls", None))
    C.append(("XIII", "E-HSFP (200 rounds)", "cifar100", "comm200_cifar100_ehsfp", "cls", None))
    return C


# ------------------------------------------------------------------- main
def main():
    for d in (ART, TAB, FIG, ROOT / "manifests"):
        d.mkdir(parents=True, exist_ok=True)
    man, queues = load_manifest(), load_queues()
    done = {p.stem for p in MARK.glob("*.done")}
    failed = {p.stem for p in MARK.glob("*.failed")}
    universe = sorted(set(man) | done | failed | set(queues) | {p.name[:-5] for p in MARK.glob("*.lock")})
    runs = {}
    for eid in universe:
        desc = describe(eid, queues.get(eid), man.get(eid))
        logp = LOGS / f"{eid}.log"
        lt = read_text(logp)
        lk = lock_state(eid)
        r = {"experiment_id": eid, **desc, "log_path": str(logp.relative_to(ROOT)) if logp.exists() else "",
             "runner_id": "", "failure_reason": "", "next_action": "", "progress": ""}
        attempts = sorted(LOGS.glob(f"{eid}.attempt-*.log"))
        if eid in done and man.get(eid, {}).get("status", "").startswith("PASS"):
            met = extract(eid, desc["task"], desc["method"], lt)
            r.update({k: v for k, v in met.items() if k not in ("curve", "drift_curve")})
            r["_curve"], r["_drift"] = met.get("curve"), met.get("drift_curve")
            primary = met.get("final_acc") if desc["task"] == "classification" else met.get("test_iou")
            if primary is None:
                r["status"], r["failure_reason"] = "completed-unparsed", "finished but no metric could be extracted"
                r["next_action"] = "export fix"
            elif any(n.startswith(("near_chance", "best_at_round_1")) for n in met["notes"]):
                r["status"] = "completed-suspect"
                r["failure_reason"] = ";".join(met["notes"])
                r["next_action"] = "diagnose baseline (kept; not silently dropped)"
            else:
                r["status"] = "completed-valid"
            if man[eid].get("status") == "PASS_NAN_WARNING":
                r["status"] = "completed-suspect"
                r["failure_reason"] += ";nan_in_log"
            r["progress"] = f"{met.get('rounds_done', desc['budget'])}/{desc['budget']}"
            r["checkpoint"] = str(Path(met["run_dir"]) / "checkpoint.pt") if met.get("run_dir") and (Path(met["run_dir"]) / "checkpoint.pt").exists() else ""
        elif lk and lk["alive"]:
            ep = EPOCH.findall(lt) or re.findall(r"\| *(\d+)/(\d+) \[", lt)
            r["status"] = "running-healthy"
            lane = next((f.stem for f in QDIR.glob("*.pid") if read_text(f).strip() == lk["pid"]), "?")
            r["runner_id"] = f"tmux ehsfp-lanes / {lane} / pid {lk['pid']}"
            r["progress"] = f"{ep[-1][0]}/{ep[-1][1]} (round in progress)" if ep else "starting"
            r["next_action"] = "wait (lane continues automatically)"
        elif eid in failed:
            r["status"] = "failed"
            tail = lt.strip().splitlines()[-3:] if lt else []
            r["failure_reason"] = " | ".join(tail)[:300]
            r["next_action"] = "diagnose, then ./journal_run.sh retry_failed"
        elif eid in queues:
            r["status"] = "queued"
            r["runner_id"] = f"tmux ehsfp-lanes / {queues[eid]['lane']}"
            r["next_action"] = "runs when its lane reaches it"
            if attempts:
                r["failure_reason"] = f"earlier attempt interrupted (host down 2026-08-29): {attempts[-1].name}"
        else:
            r["status"] = "pending"
            r["next_action"] = "not queued"
        if attempts and r["status"] != "queued":
            r["failure_reason"] = (r["failure_reason"] + f";earlier interrupted attempt preserved: {attempts[-1].name}").strip(";")
        if eid.startswith("parity"):
            r["status"] = r["status"] + " (duplicate of main_*_hsfp_s0; not pooled)"
        r["code_revision"] = CODE_REV + "+dirty"
        if not r.get("config_hash"):
            r["config_hash"] = hashlib.sha256(f"{desc['cfg']}|{desc['variant']}|{desc['seed']}".encode()).hexdigest()[:16] + " (launcher-args hash)"
        runs[eid] = r

    # ---- cell-level degeneracy: a metric constant across seeds (SD < 0.05 pp
    # with >=3 seeds) means the seed had no effect -> constant predictor.
    groups = defaultdict(list)
    for eid, r in runs.items():
        if r["status"].startswith("completed") and re.search(r"_s\d+$", eid):
            groups[re.sub(r"_s\d+$", "", eid)].append(r)
    for g, rs in groups.items():
        key = "test_iou" if rs[0]["task"] == "segmentation" else "final_acc"
        vals = [r.get(key) for r in rs if r.get(key) is not None]
        if len(vals) >= 3 and float(np.std(vals, ddof=1)) < 5e-4:
            for r in rs:
                r["status"] = "completed-suspect"
                r["failure_reason"] = (r.get("failure_reason", "") + ";constant_across_seeds(sd<0.05pp)").strip(";")
                r["next_action"] = "diagnose baseline (kept; not silently dropped)"

    # ---- manifests/run_status.csv
    cell_of = defaultdict(list)
    for t, row, d, tpl, kind, blk in cells():
        if tpl:
            for s in SEEDS:
                cell_of[f"{tpl}_s{s}"].append(f"Table {t}: {row}" + (f" [{d}]" if t == "I" else ""))
    cols = ["experiment_id", "table_or_figure", "dataset", "method", "variant", "partition", "seed",
            "protocol", "budget", "config_hash", "code_revision", "status", "progress", "checkpoint",
            "log_path", "runner_id", "failure_reason", "next_action"]
    with open(ROOT / "manifests" / "run_status.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for eid, r in runs.items():
            w.writerow({**r, "table_or_figure": "; ".join(cell_of.get(eid, [])) or "supporting"})

    # ---- artifacts/per_run_metrics.csv
    mcols = ["experiment_id", "status", "dataset", "method", "variant", "partition", "seed",
             "final_acc", "final_f1", "best_acc_testselected", "test_iou", "test_dice", "final_iou",
             "final_dice", "selection_rule", "comm_MB", "runtime_s", "rounds_done", "conv_rounds",
             "stability_final", "drift_mean", "mean_packet_age", "client_gpu_peak_MB",
             "edge_gpu_peak_MB", "cloud_gpu_peak_MB", "client_mem_records", "edge_mem_records",
             "peak_vram_MB", "notes", "run_dir"]
    with open(ART / "per_run_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mcols, extrasaction="ignore")
        w.writeheader()
        for eid, r in runs.items():
            if r["status"].startswith("completed"):
                w.writerow({**r, "notes": ";".join(r.get("notes", []))})

    # ---- coverage + tables
    def seed_runs(tpl):
        out = []
        for s in SEEDS:
            r = runs.get(f"{tpl}_s{s}")
            out.append(r)
        return out

    def agg(tpl, key):
        vals = [r.get(key) for r in seed_runs(tpl) if r and r["status"].startswith("completed")]
        return mean_sd(vals)

    cov_rows = []
    for t, row, d, tpl, kind, blk in cells():
        if blk:
            cov_rows.append({"table": t, "cell": row, "dataset": d, "runs": "", "seeds_completed": 0,
                             "seeds_required": 5, "status": blk.split(":")[0], "detail": blk})
            continue
        rs = seed_runs(tpl) if not tpl.startswith("comm200") else [runs.get(f"{tpl}_s0")]
        need = 1 if tpl.startswith("comm200") else 5
        comp = [r for r in rs if r and r["status"].startswith("completed")]
        susp = [r for r in comp if r["status"].startswith("completed-suspect")]
        states = sorted({(r["status"] if r else "pending") for r in rs})
        st = "complete" if len(comp) >= need else ("partial" if comp else "missing")
        if susp:
            st += " (suspect seeds: %d)" % len(susp)
        cov_rows.append({"table": t, "cell": row, "dataset": d,
                         "runs": ";".join(f"{tpl}_s{s}" for s in (SEEDS if need == 5 else [0])),
                         "seeds_completed": len(comp), "seeds_required": need, "status": st,
                         "detail": ",".join(states)})
    with open(ART / "coverage_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cov_rows[0]))
        w.writeheader()
        w.writerows(cov_rows)

    write_tables(runs, agg, seed_runs)
    write_figures(runs, seed_runs)
    write_report(runs, cov_rows)
    n = defaultdict(int)
    for r in runs.values():
        n[r["status"].split(" ")[0]] += 1
    print("run states:", dict(n))
    c = defaultdict(int)
    for r in cov_rows:
        c[r["status"].split(" ")[0]] += 1
    print("cell coverage:", dict(c))


# ------------------------------------------------------------------ tables
def save_table(name, header, rows, caption):
    with open(TAB / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    def esc(x):
        return str(x).replace("±", r"$\pm$").replace("_", r"\_").replace("%", r"\%").replace("—", "--")
    with open(TAB / f"{name}.tex", "w") as fh:
        fh.write("% Generated by scripts/journal_experiments/collect_journal_v1.py — do not edit by hand\n")
        fh.write("\\begin{table}[t]\\centering\\scriptsize\n\\caption{" + esc(caption) + "}\n")
        fh.write("\\begin{tabular}{" + "l" * len(header) + "}\\toprule\n")
        fh.write(" & ".join(esc(h) for h in header) + "\\\\\\midrule\n")
        for r in rows:
            fh.write(" & ".join(esc(x) for x in r) + "\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\\end{table}\n")


def write_tables(runs, agg, seed_runs):
    P = 100.0
    note = "Mean ± sample SD (ddof=1) over seeds 0-4; '(n=k/5)' marks incomplete cells; '—' = no completed run."
    # Table I (journal matched)
    rows = []
    for row, tpl in [("FedAvg", "main_{d}_federated"), ("FedProx", "fed_{d}_fedprox"), ("FedNova", "fed_{d}_fednova"),
                     ("HierFL", "main_{d}_hierfl"), ("SplitFed", "main_{d}_splitfl"), ("HeteroSFL", "main_{d}_hetero-sfl"),
                     ("HSFL (A)", "main_{d}_hsfl"), ("H-SFP (A)", "main_{d}_hsfp"), ("E-HSFP", "main_{d}_ehsfp")]:
        cells_ = [fmt(*agg(tpl.format(d=d), "final_acc"), scale=P) for d in ("cifar10", "cifar100", "ham10000")]
        rows.append([row] + cells_ + ["blocked-data"])
    for row in ("FedProto", "FedGen", "FedDF"):
        rows.append([row, "not implemented", "not implemented", "not implemented", "blocked-data"])
    save_table("table_I_journal_matched", ["Method", "CIFAR-10", "CIFAR-100", "HAM10000", "ImageNet-1K"], rows,
               "Journal-protocol matched classification accuracy (%), final round (60), IID, 200 clients. " + note +
               " Conference values (manuscript Table I) are NOT pooled here.")
    # Table I macro-F1 companion
    rows = []
    for row, tpl in [("SplitFed", "main_{d}_splitfl"), ("HeteroSFL", "main_{d}_hetero-sfl"), ("HSFL (A)", "main_{d}_hsfl"),
                     ("H-SFP (A)", "main_{d}_hsfp"), ("E-HSFP", "main_{d}_ehsfp")]:
        rows.append([row] + [fmt(*agg(tpl.format(d=d), "final_f1"), scale=P) for d in ("cifar10", "cifar100", "ham10000")])
    save_table("table_I_macro_f1", ["Method", "CIFAR-10", "CIFAR-100", "HAM10000"], rows,
               "Final-round macro-F1 (%) recomputed from saved predictions. FedAvg/FedProx/FedNova/HierFL never persisted predictions or F1. " + note)
    # Table I secondary: test-selected best round (H-SFP family runner's own rule)
    rows = []
    for row, tpl in [("FedAvg", "main_{d}_federated"), ("HierFL", "main_{d}_hierfl"), ("H-SFP (A)", "main_{d}_hsfp"), ("E-HSFP", "main_{d}_ehsfp")]:
        rows.append([row] + [fmt(*agg(tpl.format(d=d), "best_acc_testselected"), scale=P) for d in ("cifar10", "cifar100", "ham10000")])
    save_table("table_I_best_round_test_selected", ["Method", "CIFAR-10", "CIFAR-100", "HAM10000"], rows,
               "SECONDARY ONLY: max-over-rounds accuracy where the selection split IS the test split (optimistically biased). " + note)
    # Table II
    rows = []
    for row, tpl in [("FedAvg", "main_isic2018_federated"), ("HierFL", "main_isic2018_hierfl"), ("SplitFed", "main_isic2018_splitfl"),
                     ("HeteroSFL", "main_isic2018_hetero-sfl"), ("HSFL (A)", "main_isic2018_hsfl"), ("H-SFP (A)", "main_isic2018_hsfp"),
                     ("E-HSFP", "main_isic2018_ehsfp")]:
        rs = [r for r in seed_runs(tpl) if r and r["status"].startswith("completed")]
        rule = rs[0].get("selection_rule", "") if rs else ""
        th = mean_sd([(r.get("runtime_s") or 0) / 3600 for r in rs]) if rs else (None, None, 0)
        flag = "SUSPECT: " + rs[0].get("failure_reason", "") if rs and any(r["status"].startswith("completed-suspect") for r in rs) else ""
        rows.append([row, fmt(*agg(tpl, "test_iou"), scale=P), fmt(*agg(tpl, "test_dice"), scale=P),
                     fmt(*th, scale=1.0), rule, flag])
    save_table("table_II_journal_matched", ["Method", "IoU (%)", "Dice (%)", "Time (h, host wall)", "Selection rule", "Flag"], rows,
               "Journal-protocol ISIC-2018 (60 rounds, 50 clients). Selection rules differ by runner (recorded per row). " + note)

    def stress_table(name, header, spec, caption):
        rows = []
        for label, tpl in spec:
            rows.append([label, fmt(*agg(tpl, "final_acc"), scale=P), fmt(*agg(tpl, "final_f1"), scale=P),
                         fmt(*agg(tpl, "conv_rounds"), nd=1), fmt(*agg(tpl, "stability_final"), scale=1e4, nd=3),
                         fmt(*agg(tpl, "drift_mean"), nd=3)])
        save_table(name, header, rows, caption)
    H = ["Setting", "Final acc (%)", "Final macro-F1 (%)", "Rounds to conv.", "Stability W=5 (x1e-4)", "Edge drift W2^2 (oracle)"]
    stress_table("table_III_dropout", H, [("p=0", "dropout_cifar100_p0p0"), ("p=0.1", "dropout_cifar100_p0p1"),
                  ("p=0.2 (=main E-HSFP)", "main_cifar100_ehsfp"), ("p=0.3", "dropout_cifar100_p0p3"),
                  ("p=0.5", "dropout_cifar100_p0p5")],
                 "Packet-loss (prototype dropout, whole client packets) robustness, CIFAR-100 E-HSFP. Drift = Def. 6 oracle diagnostic (blank for runs that predate instrumentation). " + note)
    stress_table("table_IV_staleness", H, [("tau=0 (=main E-HSFP)", "main_cifar100_ehsfp"), ("tau=1", "stale_cifar100_tau1"),
                  ("tau=3", "stale_cifar100_tau3"), ("tau=5", "stale_cifar100_tau5"), ("tau=10", "stale_cifar100_tau10")],
                 "Bounded staleness: client packets delivered after d~U{0..tau} rounds (real delayed statistics). Stability-bound term NOT computed: Theorem 1 constants are unspecified. " + note)
    rows = []
    base_vals = {s: (runs.get(f"sv_cifar100_none_s{s}") or {}).get("final_acc") for s in SEEDS}
    for label, k in [("No event loss", "none"), ("Function timeout (client p=0.10)", "timeout"),
                     ("Cold-start delay (p=0.30, packet deferred 1 round)", "coldstart"),
                     ("Partial edge execution (q=0.30, 50% of batches)", "partial"),
                     ("Missing edge update (edge timeout p=0.10)", "missedge"), ("Combined serverless stress", "combined")]:
        tpl = f"sv_cifar100_{k}"
        gaps = []
        for s in SEEDS:
            r = runs.get(f"{tpl}_s{s}")
            if r and r["status"].startswith("completed") and base_vals.get(s) is not None and r.get("final_acc") is not None:
                gaps.append(base_vals[s] - r["final_acc"])
        rows.append([label, fmt(*agg(tpl, "final_acc"), scale=P), fmt(*agg(tpl, "final_f1"), scale=P),
                     fmt(*agg(tpl, "drift_mean"), nd=3), fmt(*agg(tpl, "conv_rounds"), nd=1),
                     fmt(*mean_sd(gaps), scale=P) if k != "none" else "0 (reference)"])
    save_table("table_V_serverless", ["Stress condition", "Acc (%)", "Macro-F1 (%)", "Edge drift W2^2", "Rounds to conv.", "Recovery gap (pp, seed-paired)"], rows,
               "Serverless-style events on CIFAR-100 E-HSFP (prototype dropout p=0.2 kept as the method component). Latencies are SIMULATED time, not host wall time. " + note)
    rows = []
    for d, dn in (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("ham10000", "HAM10000")):
        for part, pre in [("IID", "main_{d}_{m}"), ("Non-IID mild dir(0.7)", "noniid_a0p7_{d}_{m}"), ("Non-IID severe dir(0.3)", "noniid_a0p3_{d}_{m}")]:
            for m, lab in (("hsfp", "H-SFP"), ("ehsfp", "E-HSFP")):
                tpl = pre.format(d=d, m=m)
                rows.append([dn, part, lab, fmt(*agg(tpl, "final_acc"), scale=P), fmt(*agg(tpl, "final_f1"), scale=P),
                             fmt(*agg(tpl, "conv_rounds"), nd=1), fmt(*agg(tpl, "drift_mean"), nd=3),
                             fmt(*agg(tpl, "comm_MB"), scale=1 / 1024, nd=1)])
    save_table("table_VI_noniid", ["Dataset", "Partition", "Method", "Acc (%)", "Macro-F1 (%)", "Rounds to conv.", "Edge drift", "Comm (GB)"], rows,
               "Journal-protocol IID / non-IID robustness (conference-ported values excluded). " + note)
    rows = []
    for label, tpl in [("H-SFP baseline", "main_cifar100_hsfp"), ("+ Episodic memory", "abl_cifar100_hsfp_memory"),
                       ("+ Prototype dropout (memory+dropout+serverless sim)", "abl_cifar100_hsfp_memory_dropout"),
                       ("+ Reliability-aware aggregation (memory+reliability)", "abl_cifar100_hsfp_memory_reliability"),
                       ("+ Replay consistency (memory+reliability+PRC)", "abl_cifar100_hsfp_memory_reliability_prc"),
                       ("+ Residual prototype generator", None), ("Full E-HSFP", "main_cifar100_ehsfp")]:
        if tpl is None:
            rows.append([label, "blocked", "blocked", "blocked", "blocked", "blocked"])
            continue
        rows.append([label, fmt(*agg(tpl, "final_acc"), scale=P), fmt(*agg(tpl, "final_f1"), scale=P),
                     fmt(*agg(tpl, "comm_MB"), scale=1 / 1024, nd=1), fmt(*agg(tpl, "conv_rounds"), nd=1),
                     fmt(*agg(tpl, "stability_final"), scale=1e4, nd=3)])
    save_table("table_VIII_ablation", ["Variant", "Acc (%)", "Macro-F1 (%)", "Comm (GB)", "Rounds to conv.", "Stability (x1e-4)"], rows,
               "E-HSFP module ablation, CIFAR-100, frozen protocol. Rows are the repository presets (not strictly cumulative). Full E-HSFP's residual generator is a no-op. " + note)
    rows = []
    for mm in (0, 1, 5, 10, 20):
        tpl = f"memM_cifar100_M{mm}"
        ov = []
        for r in seed_runs(tpl):
            if r and r["status"].startswith("completed") and r.get("client_mem_records") is not None:
                # CIFAR split: client prototype 128-d, edge prototype 256-d; mu+sigma fp32
                ov.append((r["client_mem_records"] * 2 * 128 + (r.get("edge_mem_records") or 0) * 2 * 256) * 4 / 2 ** 20)
        rows.append([f"M={mm}", fmt(*agg(tpl, "final_acc"), scale=P), fmt(*agg(tpl, "final_f1"), scale=P),
                     fmt(*agg(tpl, "drift_mean"), nd=3), fmt(*agg(tpl, "conv_rounds"), nd=1), fmt(*mean_sd(ov), nd=2)])
    save_table("table_IX_memory", ["Memory size (per class)", "Acc (%)", "Macro-F1 (%)", "Edge drift", "Rounds to conv.", "Memory overhead (MB, final round)"], rows,
               "Per-class FIFO memory capacity (eq. 7) under dropout p=0.2 + staleness tau=3, CIFAR-100. " + note)
    rows = []
    for label, k in [("Average", "average"), ("Sample-count weighted", "sample_count_weighted"),
                     ("Reliability-aware", "learnable_reliability"), ("Reliability-aware + memory", "learnable_reliability_memory")]:
        c = f"aggr_cifar100_{k}_clean"
        rows.append([label, fmt(*agg(c, "final_acc"), scale=P), fmt(*agg(c, "final_f1"), scale=P),
                     fmt(*agg(f"aggr_cifar100_{k}_drop", "final_acc"), scale=P),
                     fmt(*agg(f"aggr_cifar100_{k}_stale", "final_acc"), scale=P),
                     fmt(*agg(c, "comm_MB"), scale=1 / 1024, nd=1)])
    save_table("table_X_aggregation", ["Aggregation rule", "Acc clean (%)", "F1 clean (%)", "Acc @ dropout p=0.3", "Acc @ staleness tau=5", "Comm (GB)"], rows,
               "Aggregation rules, CIFAR-100, no PRC/serverless sim. Reliability without memory uses an untrained reliability net (bootstrap needs memory). " + note)
    rows = [["Diagonal Gaussian (=main E-HSFP)", fmt(*agg("main_cifar100_ehsfp", "final_acc"), scale=P),
             fmt(*agg("main_cifar100_ehsfp", "final_f1"), scale=P), "not computed", "0 (baseline)",
             fmt(*agg("main_cifar100_ehsfp", "runtime_s"), scale=1 / 3600, nd=2) + " h"]]
    for r in ("Low-rank covariance", "Residual generator", "Residual generator + consistency loss"):
        rows.append([r, "blocked", "blocked", "blocked", "blocked", "blocked"])
    save_table("table_XI_synthesis", ["Synthesis module", "Acc (%)", "Macro-F1 (%)", "Prototype fidelity", "Comm overhead", "Training time"], rows,
               "Only the diagonal Gaussian row exists in the pipeline; the others need implementation decisions (see completion audit). " + note)
    rows = []
    for k in (20, 50, 100, 200, 500):
        tpl = "main_cifar100_hsfp" if k == 200 else f"scale_cifar100_K{k}_hsfp"
        rows.append([str(k), fmt(*agg(tpl, "final_acc"), scale=P), fmt(*agg(tpl, "final_f1"), scale=P),
                     fmt(*agg(tpl, "comm_MB"), scale=1 / 1024, nd=1), fmt(*agg(tpl, "client_gpu_peak_MB"), nd=0),
                     fmt(*agg(tpl, "runtime_s"), scale=1 / 3600, nd=2), fmt(*agg(tpl, "conv_rounds"), nd=1)])
    save_table("table_XII_scalability", ["Clients", "Acc (%)", "Macro-F1 (%)", "Comm (GB)", "Client-phase GPU peak (MB)", "Time (h)", "Rounds to conv."], rows,
               "H-SFP(A), CIFAR-100 IID, journal protocol (60 rounds). Conference accuracies are not pooled. " + note)
    r = runs.get("comm200_cifar100_ehsfp_s0")
    val = f"{r['comm_MB'] / 1024:.2f}" if r and r.get("comm_MB") else "—"
    save_table("table_XIII_comm", ["Method", "Comm (GB), 200 rounds", "Source"], [["E-HSFP", val, "comm200_cifar100_ehsfp_s0 (single seed)"]],
               "E-HSFP literal 200-round communication (measured, not extrapolated).")


# ------------------------------------------------------------------ report
def write_report(runs, cov_rows):
    """reports/simulation_completion_report.md, regenerated from the exports."""
    import datetime
    (ROOT / "reports").mkdir(exist_ok=True)
    st = defaultdict(int)
    for r in runs.values():
        st[r["status"].split(" ")[0]] += 1
    L = ["# E-HSFP journal campaign — simulation completion report", "",
         f"_Auto-generated by `scripts/journal_experiments/collect_journal_v1.py` at "
         f"{datetime.datetime.now().isoformat(timespec='seconds')}. Do not edit by hand; re-run the collector._", "",
         "Read with `reports/completion_audit.md` (causes, fixes, metric rules, blockers). Every number below is "
         "computed from recorded run artifacts; empty cells (—) have no completed run and are NOT estimated.", "",
         "## Run states", "", "| state | runs |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(st.items())]
    L += ["", "## Coverage by manuscript table", "", "| Table | complete | partial | missing | blocked |", "|---|---|---|---|---|"]
    by = defaultdict(lambda: defaultdict(int))
    for c in cov_rows:
        k = c["status"].split(" ")[0]
        by[c["table"]]["blocked" if k.startswith("blocked") else k] += 1
    order = ["I", "II", "III", "IV", "V", "VI", "VIII", "IX", "X", "XI", "XII", "XIII"]
    for t in order:
        d = by.get(t, {})
        L.append(f"| {t} | {d.get('complete', 0)} | {d.get('partial', 0)} | {d.get('missing', 0)} | {d.get('blocked', 0)} |")
    L += ["", "Tables VII (conference H-SFP ablation) and the conference rows of I/II/VI/XII/XIII are verbatim "
          "conference values; they are not regenerated and never pooled with journal-protocol numbers.", ""]
    running = [r for r in runs.values() if r["status"].startswith("running")]
    if running:
        L += ["## Running now", "", "| run | progress | runner |", "|---|---|---|"]
        L += [f"| {r['experiment_id']} | {r['progress']} | {r['runner_id']} |" for r in running]
        L.append("")
    for f in sorted(TAB.glob("table_*.csv")):
        rows = list(csv.reader(open(f)))
        if not rows:
            continue
        L += [f"## {f.stem}", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        L += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        L.append("")
    sus = [r for r in runs.values() if r["status"].startswith("completed-suspect")]
    if sus:
        L += ["## Completed-suspect runs (kept, flagged)", "", "| run | reason |", "|---|---|"]
        L += [f"| {r['experiment_id']} | {r.get('failure_reason', '')} |" for r in sus]
        L.append("")
    L += ["## Figures", ""] + [f"- `artifacts/figures/{p.name}`" for p in sorted(FIG.glob("*.png"))]
    (ROOT / "reports" / "simulation_completion_report.md").write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ figures
def write_figures(runs, seed_runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def band(ax, tpl, label, key="_curve", scale=100.0):
        cs = [r[key] for r in seed_runs(tpl) if r and r["status"].startswith("completed") and r.get(key)]
        if not cs:
            return False
        L = min(len(c) for c in cs)
        a = np.array([c[:L] for c in cs]) * scale
        x = np.arange(1, L + 1)
        mu = a.mean(0)
        ax.plot(x, mu, label=f"{label} (n={len(cs)})")
        if len(cs) > 1:
            sd = a.std(0, ddof=1)
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
        return True

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG / f"{name}.{ext}", dpi=150)
        plt.close(fig)

    # Fig 2 — convergence IID / non-IID
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (d, dn) in zip(axes, (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("ham10000", "HAM10000"))):
        for pre, pl in (("main_{d}_{m}", "IID"), ("noniid_a0p7_{d}_{m}", "dir(0.7)"), ("noniid_a0p3_{d}_{m}", "dir(0.3)")):
            for m, ml in (("hsfp", "H-SFP"), ("ehsfp", "E-HSFP")):
                band(ax, pre.format(d=d, m=m), f"{ml} {pl}")
        ax.set_title(dn); ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy (%)"); ax.legend(fontsize=6)
    save(fig, "fig2_convergence")

    def sweep_fig(name, spec, key="_curve", ylabel="Test accuracy (%)", scale=100.0):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        any_ = False
        for label, tpl in spec:
            any_ |= band(ax, tpl, label, key=key, scale=scale)
        ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
        if any_:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no completed runs yet", ha="center", transform=ax.transAxes)
        save(fig, name)

    sweep_fig("fig3_drift_over_rounds", [("No event loss", "sv_cifar100_none"), ("Combined stress", "sv_cifar100_combined"),
                                         ("tau=5", "stale_cifar100_tau5"), ("p=0.5", "dropout_cifar100_p0p5")],
              key="_drift", ylabel="Edge drift W2^2 (oracle)", scale=1.0)
    sweep_fig("fig5_dropout", [("p=0", "dropout_cifar100_p0p0"), ("p=0.1", "dropout_cifar100_p0p1"), ("p=0.2", "main_cifar100_ehsfp"),
                               ("p=0.3", "dropout_cifar100_p0p3"), ("p=0.5", "dropout_cifar100_p0p5")])
    sweep_fig("fig6_staleness", [("tau=0", "main_cifar100_ehsfp"), ("tau=1", "stale_cifar100_tau1"), ("tau=3", "stale_cifar100_tau3"),
                                 ("tau=5", "stale_cifar100_tau5"), ("tau=10", "stale_cifar100_tau10")])
    sweep_fig("fig9_memory", [(f"M={m}", f"memM_cifar100_M{m}") for m in (0, 1, 5, 10, 20)])

    # Fig 7 — module ablation bars
    spec = [("H-SFP", "main_cifar100_hsfp"), ("+mem", "abl_cifar100_hsfp_memory"), ("+drop", "abl_cifar100_hsfp_memory_dropout"),
            ("+rel", "abl_cifar100_hsfp_memory_reliability"), ("+PRC", "abl_cifar100_hsfp_memory_reliability_prc"), ("Full", "main_cifar100_ehsfp")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (key, lab, sc) in zip(axes, (("final_acc", "Final acc (%)", 100), ("conv_rounds", "Rounds to conv.", 1), ("stability_final", "Stability (x1e-4)", 1e4))):
        xs, ms, sds = [], [], []
        for l, tpl in spec:
            vals = [r.get(key) for r in seed_runs(tpl) if r and r["status"].startswith("completed")]
            m, sd, n = mean_sd(vals)
            xs.append(f"{l}\n(n={n})"); ms.append((m or 0) * sc); sds.append((sd or 0) * sc)
        ax.bar(xs, ms, yerr=sds, capsize=3); ax.set_ylabel(lab); ax.tick_params(axis="x", labelsize=7)
    save(fig, "fig7_module_ablation")

    # Fig 8 — reliability weights vs factors
    rows = []
    for r in runs.values():
        if r["status"].startswith("completed") and r.get("run_dir"):
            p = Path(r["run_dir"]) / "diagnostics" / "reliability_weights.csv"
            if p.exists():
                rows += [x for x in csv.DictReader(open(p)) if x["tier"] == "edge"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (k, lab) in zip(axes, (("support_feat", "support (norm.)"), ("sigma_feat", "variance |sigma| (norm.)"),
                                   ("age_feat", "age (norm.)"), ("distance_feat", "reference distance (norm.)"))):
        if rows:
            x = np.array([float(z[k]) for z in rows]); y = np.array([float(z["normalized_weight"]) for z in rows])
            bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(x, bins) - 1, 0, 9)
            means = [y[idx == b].mean() if (idx == b).any() else np.nan for b in range(10)]
            ax.plot((bins[:-1] + bins[1:]) / 2, means, marker="o")
        else:
            ax.text(0.5, 0.5, "no instrumented runs yet", ha="center", transform=ax.transAxes)
        ax.set_xlabel(lab); ax.set_ylabel("mean normalized weight")
    save(fig, "fig8_reliability_weights")

    # Fig 10 — communication vs accuracy (journal matched, CIFAR-100 + others)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, d in zip(axes, ("cifar10", "cifar100", "ham10000")):
        for lab, tpl in [("FedAvg", "main_{d}_federated"), ("FedProx", "fed_{d}_fedprox"), ("FedNova", "fed_{d}_fednova"),
                         ("HierFL", "main_{d}_hierfl"), ("SplitFed", "main_{d}_splitfl"), ("HeteroSFL", "main_{d}_hetero-sfl"),
                         ("HSFL", "main_{d}_hsfl"), ("H-SFP", "main_{d}_hsfp"), ("E-HSFP", "main_{d}_ehsfp")]:
            rs = [r for r in seed_runs(tpl.format(d=d)) if r and r["status"].startswith("completed") and r.get("comm_MB") and r.get("final_acc") is not None]
            if rs:
                cx = mean_sd([r["comm_MB"] / 1024 for r in rs]); cy = mean_sd([r["final_acc"] * 100 for r in rs])
                ax.errorbar(cx[0], cy[0], xerr=cx[1] or 0, yerr=cy[1] or 0, fmt="o", capsize=3, label=f"{lab} (n={len(rs)})")
        ax.set_xscale("log"); ax.set_xlabel("Total communication (GB, 60 rounds, log)"); ax.set_ylabel("Final acc (%)")
        ax.set_title(d); ax.legend(fontsize=6)
    save(fig, "fig10_comm_vs_accuracy")

    # Fig 11 — tier memory vs performance (instrumented runs only)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    pts = 0
    for tier, mk in (("client", "o"), ("edge", "s"), ("cloud", "^")):
        xs, ys = [], []
        for r in runs.values():
            if r["status"].startswith("completed") and r.get(f"{tier}_gpu_peak_MB") and r.get("final_acc") is not None:
                xs.append(r[f"{tier}_gpu_peak_MB"]); ys.append(r["final_acc"] * 100)
        if xs:
            ax.scatter(xs, ys, marker=mk, label=f"{tier} phase (n={len(xs)})"); pts += len(xs)
    ax.set_xlabel("Process CUDA peak during tier phase (MB)"); ax.set_ylabel("Final acc (%)")
    if pts:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no instrumented runs yet", ha="center", transform=ax.transAxes)
    save(fig, "fig11_memory_vs_performance")


if __name__ == "__main__":
    main()
