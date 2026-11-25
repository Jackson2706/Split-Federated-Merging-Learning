#!/usr/bin/env python3
"""E5 accumulation pilot: seed 0 of the four Dir(0.3) variants with edge accumulation on vs off.

Reads only results/revision/E5_*_s0/{summary.json,config.yaml,dispersion.csv};
writes analysis/out/pilot_e5.csv and prints a table. Single seed: indicative only.

    python analysis/pilot_e5.py [--root results/revision]
"""
import argparse
import csv
import glob
import json
import os

import yaml

FIELDS = ["dataset", "variant", "accumulation", "test_accuracy", "test_macro_f1", "final_accuracy",
          "best_accuracy", "relative_rounds", "gain_vs_ref_pp", "late_dispersion", "status", "run"]


def describe(cfg):
    sampler = cfg.get("sampler", "gaussian") or "gaussian"
    fix = bool(cfg.get("scatter_fix", 0))
    variant = {("gaussian", False): "reference", ("mixture", False): "mixture sampler",
               ("gaussian", True): "scatter fix", ("mixture", True): "both"}[(sampler, fix)]
    on = cfg.get("memory_mode") == "statistics" and cfg.get("stat_accumulation_decay", None) not in (0, 0.0)
    on = on or float(cfg.get("stat_accumulation_decay", 0) or 0) > 0
    return variant, "on" if on else "off"


def late_dispersion(path, last=20):
    if not os.path.exists(path):
        return ""
    rows = [r for r in csv.DictReader(open(path)) if r["mean_dispersion"]]
    if not rows:
        return ""
    cut = max(int(r["round"]) for r in rows) - last
    vals = [float(r["mean_dispersion"]) for r in rows if int(r["round"]) > cut]
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/revision")
    ap.add_argument("--out", default="analysis/out/pilot_e5.csv")
    a = ap.parse_args()
    table = []
    for d in sorted(glob.glob(os.path.join(a.root, "E5_*_s0"))):
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        variant, acc = describe(cfg)
        row = {"dataset": cfg["dataset"], "variant": variant, "accumulation": acc, "run": os.path.basename(d),
               "late_dispersion": late_dispersion(os.path.join(d, "dispersion.csv"))}
        if os.path.exists(os.path.join(d, "DONE")):
            s = json.load(open(os.path.join(d, "summary.json")))
            row.update({k: s.get(k) for k in ("test_accuracy", "test_macro_f1", "final_accuracy",
                                               "best_accuracy", "relative_rounds")}, status="done")
        else:
            row["status"] = "pending"
        table.append(row)
    ref = {(r["dataset"], r["accumulation"]): r.get("test_accuracy") for r in table if r["variant"] == "reference"}
    for r in table:
        base = ref.get((r["dataset"], r["accumulation"]))
        if r.get("test_accuracy") is not None and base is not None:
            r["gain_vs_ref_pp"] = round(100 * (r["test_accuracy"] - base), 2)
    order = ["reference", "mixture sampler", "scatter fix", "both"]
    table.sort(key=lambda r: (r["dataset"], order.index(r["variant"]), r["accumulation"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    fmt = lambda v: "" if v in (None, "") else (f"{100 * v:.2f}" if isinstance(v, float) and v <= 1 else str(v))
    print(f"{'dataset':10} {'variant':16} {'accum':5} {'test acc':>8} {'F1':>6} {'gain pp':>7} {'disp':>8} status")
    for r in table:
        disp = f"{r['late_dispersion']:.3g}" if isinstance(r["late_dispersion"], float) else ""
        print(f"{r['dataset']:10} {r['variant']:16} {r['accumulation']:5} {fmt(r.get('test_accuracy')):>8} "
              f"{fmt(r.get('test_macro_f1')):>6} {str(r.get('gain_vs_ref_pp', '')):>7} {disp:>8} {r['status']}")
    print(f"\nwrote {a.out}  (one seed per cell: use as a pilot, not as a result)")


if __name__ == "__main__":
    main()
