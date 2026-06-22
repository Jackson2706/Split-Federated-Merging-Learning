#!/usr/bin/env python3
"""
Aggregate camera-ready experiment logs into summary CSVs (mean +/- std over seeds).

Scans results/camera_ready/logs/ for:
  hetero_<method>_ae<AE>_ac<AC>_s<seed>.log
  partial_<method>_frac<F>_<dist>_s<seed>.log
extracts the final test F1, and writes:
  results/camera_ready/hetero/summary_hetero.csv
  results/camera_ready/partial/summary_partial.csv
  results/camera_ready/{hetero,partial}/raw_*.csv

  python scripts/camera_ready/collect_camera_ready.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from camera_ready.io_utils import CR_ROOT, cr_dir, write_csv, aggregate_seeds

LOGS_DIR = os.path.join(CR_ROOT, "logs")

# Test F1 (%) across all method runners; Best Validation F1 for H-SFP.
_TEST_F1 = re.compile(r"(?:Final\s+)?Test F1(?:\s+Score)?\s*[:=]\s*([\d.]+)\s*%")
_BEST_F1 = re.compile(r"Best Validation F1\s*[:=]\s*([\d.]+)\s*%")

_HETERO = re.compile(r"^hetero_(?P<method>.+?)_ae(?P<ae>[\d.]+)_ac(?P<ac>[\d.]+)_s(?P<seed>\d+)$")
_PARTIAL = re.compile(r"^partial_(?P<method>.+?)_frac(?P<frac>[\d.]+)_(?P<dist>[^_]+)_s(?P<seed>\d+)$")


def parse_metrics(log_path):
    test_f1 = best_f1 = None
    with open(log_path, "r", errors="replace") as f:
        text = f.read()
    m = _TEST_F1.findall(text)
    if m:
        test_f1 = float(m[-1]) / 100.0
    b = _BEST_F1.findall(text)
    if b:
        best_f1 = float(b[-1]) / 100.0
    return test_f1, best_f1


def collect():
    if not os.path.isdir(LOGS_DIR):
        print(f"No logs dir: {LOGS_DIR}")
        return
    hetero_rows, partial_rows = [], []
    for fn in sorted(os.listdir(LOGS_DIR)):
        if not fn.endswith(".log"):
            continue
        eid = fn[:-4]
        path = os.path.join(LOGS_DIR, fn)
        test_f1, best_f1 = parse_metrics(path)

        mh = _HETERO.match(eid)
        if mh:
            d = mh.groupdict()
            hetero_rows.append({
                "method": d["method"], "alpha_edge": d["ae"], "alpha_client": d["ac"],
                "seed": d["seed"], "test_f1": test_f1, "best_f1": best_f1,
            })
            continue
        mp = _PARTIAL.match(eid)
        if mp:
            d = mp.groupdict()
            partial_rows.append({
                "method": d["method"], "frac": d["frac"], "dist": d["dist"],
                "seed": d["seed"], "test_f1": test_f1, "best_f1": best_f1,
            })

    if hetero_rows:
        hdir = cr_dir("hetero")
        write_csv(hetero_rows, os.path.join(hdir, "raw_hetero.csv"))
        agg = aggregate_seeds(hetero_rows,
                              group_keys=["method", "alpha_edge", "alpha_client"],
                              metric_keys=["test_f1", "best_f1"])
        write_csv(agg, os.path.join(hdir, "summary_hetero.csv"))
        print(f"[collect] hetero: {len(hetero_rows)} runs -> {len(agg)} conditions")

    if partial_rows:
        pdir = cr_dir("partial")
        write_csv(partial_rows, os.path.join(pdir, "raw_partial.csv"))
        agg = aggregate_seeds(partial_rows,
                              group_keys=["method", "frac", "dist"],
                              metric_keys=["test_f1", "best_f1"])
        write_csv(agg, os.path.join(pdir, "summary_partial.csv"))
        print(f"[collect] partial: {len(partial_rows)} runs -> {len(agg)} conditions")

    if not hetero_rows and not partial_rows:
        print("[collect] no matching hetero_/partial_ logs found.")


if __name__ == "__main__":
    collect()
