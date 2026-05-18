#!/usr/bin/env python3
"""
collect_results.py — Aggregate journal experiment results into summary CSVs.

Scans results/journal/logs/ for experiment log files, extracts final metrics,
and produces:
  - results/journal/summary_convergence.csv
  - results/journal/summary_ablation.csv
  - results/journal/summary_dropout.csv
  - results/journal/summary_staleness.csv
  - results/journal/summary_intervals.csv

Usage:
    python collect_results.py                   # scan & produce all CSVs
    python collect_results.py --results-dir /path/to/results/journal
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_log_metrics(log_path: str) -> dict:
    """Extract final-line metrics from a log file.

    Looks for patterns like:
        [Summary] test_f1=0.8523
        [Summary] test_iou=0.7123 test_dice=0.8045
        best_f1: 0.8523
        best_iou: 0.7123
    """
    metrics = {}
    if not os.path.isfile(log_path):
        return metrics

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            # Pattern: key=value or key: value
            for m in re.finditer(r"(test_f1|best_f1|test_iou|test_dice|best_iou|best_dice|total_time_s|train_loss)\s*[=:]\s*([\d.]+)", line):
                key, val = m.group(1), m.group(2)
                try:
                    metrics[key] = float(val)
                except ValueError:
                    pass
    return metrics


def parse_experiment_id(eid: str) -> dict:
    """Parse experiment ID into components.

    Examples:
        conv_hsfp_classification_h-sfp_cifar_our_resnet50_5_10_s0
        abl_classification_h-sfp_cifar_our_resnet50_5_10_full_e_hsfp_s2
        dropout_cls_dr0p3_s1
        staleness_seg_tau5_s0
        intv_hsfp_classification_h-sfp_cifar_our_resnet50_5_10_s3
    """
    parts = {"experiment_id": eid, "seed": None, "group": None}

    # Extract seed
    seed_match = re.search(r"_s(\d+)$", eid)
    if seed_match:
        parts["seed"] = int(seed_match.group(1))

    # Determine group
    if eid.startswith("conv_"):
        parts["group"] = "convergence"
    elif eid.startswith("abl_"):
        parts["group"] = "ablation"
    elif eid.startswith("dropout_"):
        parts["group"] = "dropout"
    elif eid.startswith("staleness_"):
        parts["group"] = "staleness"
    elif eid.startswith("intv_"):
        parts["group"] = "intervals"
    elif eid.startswith("smoke_"):
        parts["group"] = "smoke"

    return parts


def collect_all_logs(results_dir: str) -> list:
    """Scan logs directory and return list of (experiment_id, metrics) tuples."""
    logs_dir = os.path.join(results_dir, "logs")
    if not os.path.isdir(logs_dir):
        print(f"No logs directory found at {logs_dir}", file=sys.stderr)
        return []

    results = []
    for fname in sorted(os.listdir(logs_dir)):
        if not fname.endswith(".log"):
            continue
        eid = fname[:-4]  # strip .log
        metrics = parse_log_metrics(os.path.join(logs_dir, fname))
        info = parse_experiment_id(eid)
        info.update(metrics)
        results.append(info)

    return results


def aggregate_seeds(results: list) -> dict:
    """Group results by experiment (minus seed) and compute mean/std."""
    grouped = defaultdict(list)
    for r in results:
        # Key = experiment_id without seed suffix
        key = re.sub(r"_s\d+$", "", r["experiment_id"])
        grouped[key].append(r)

    aggregated = {}
    for key, runs in grouped.items():
        agg = {"experiment_id": key, "n_seeds": len(runs)}
        metric_keys = [k for k in runs[0] if k not in ("experiment_id", "seed", "group")]
        for mk in metric_keys:
            vals = [r[mk] for r in runs if mk in r and r[mk] is not None]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                import statistics
                agg[f"{mk}_mean"] = statistics.mean(vals)
                if len(vals) > 1:
                    agg[f"{mk}_std"] = statistics.stdev(vals)
                else:
                    agg[f"{mk}_std"] = 0.0
        agg["group"] = runs[0].get("group", "unknown")
        aggregated[key] = agg

    return aggregated


def write_csv(rows: list, output_path: str, fieldnames: list = None):
    """Write list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect E-HSFP journal experiment results")
    parser.add_argument("--results-dir", default=None, help="Path to results/journal/")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "..", "..", "results", "journal")

    results_dir = os.path.abspath(results_dir)
    print(f"Scanning: {results_dir}")

    all_results = collect_all_logs(results_dir)
    if not all_results:
        print("No log files found. Run experiments first.")
        return

    print(f"Found {len(all_results)} log files")

    aggregated = aggregate_seeds(all_results)

    # Split by group and write CSVs
    for group_name in ("convergence", "ablation", "dropout", "staleness", "intervals", "smoke"):
        group_rows = [v for v in aggregated.values() if v.get("group") == group_name]
        if group_rows:
            out_path = os.path.join(results_dir, f"summary_{group_name}.csv")
            write_csv(group_rows, out_path)

    # Also write a combined CSV
    all_rows = sorted(aggregated.values(), key=lambda x: x.get("experiment_id", ""))
    if all_rows:
        write_csv(all_rows, os.path.join(results_dir, "summary_all.csv"))

    print("\nDone! Summary CSVs written to:", results_dir)


if __name__ == "__main__":
    main()
