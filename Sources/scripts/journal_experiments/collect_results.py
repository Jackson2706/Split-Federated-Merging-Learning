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
    """Extract explicitly labelled final metrics from a runner log."""
    metrics = {}
    if not os.path.isfile(log_path):
        return metrics

    with open(log_path, "r", errors="replace") as f:
        text = f.read()

    number = r"([0-9]+(?:\.[0-9]+)?)"
    patterns = {
        "best_val_top1": rf"(?m)^\s*\|----\s+Best Validation Acc:\s*{number}%\s*$",
        "test_top1": rf"(?m)^\s*(?:\|----\s+)?(?:Final\s+)?Test Acc:\s*{number}%(?:\s+\(F1:\s*[0-9]+(?:\.[0-9]+)?%\))?\s*$",
        "best_iou": rf"(?m)^\s*(?:\|----\s+)?Best Validation IoU:\s*{number}%?(?:\s+Dice:\s*[0-9]+(?:\.[0-9]+)?%?)?(?:\s+Round:\s*\d+)?\s*$",
        "test_iou": rf"(?m)^\s*(?:\|----\s+)?(?:Best-checkpoint\s+)?Test IoU:\s*{number}%?(?:\s+(?:Test\s+)?Dice:\s*[0-9]+(?:\.[0-9]+)?%?)?\s*$",
        "runtime_s": rf"(?m)^\s*Total Run Time:\s*{number}s\s*$",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])

    # Accept Dice only on the Test IoU line or one of the next two labelled lines.
    test_iou_line = re.compile(
        rf"^\s*(?:\|----\s+)?(?:Best-checkpoint\s+)?Test IoU:\s*{number}%?"
    )
    dice_label = re.compile(rf"\b(?:Test\s+)?Dice:\s*{number}%?")
    standalone_dice = re.compile(
        rf"^\s*(?:\|----\s+)?(?:Test\s+)?Dice:\s*{number}%?\s*$"
    )
    lines = text.splitlines()
    for index, line in enumerate(lines):
        iou_match = test_iou_line.match(line)
        if not iou_match:
            continue
        dice_match = dice_label.search(line, iou_match.end())
        if dice_match:
            metrics["test_dice"] = float(dice_match.group(1))
            continue
        for nearby_line in lines[index + 1:index + 3]:
            nearby_match = standalone_dice.fullmatch(nearby_line)
            if nearby_match:
                metrics["test_dice"] = float(nearby_match.group(1))
                break

    comm_keys = (
        "total_comm_MB",
        "client_to_edge_data_MB",
        "edge_to_cloud_data_MB",
        "client_model_upload_MB",
        "client_model_download_MB",
        "edge_model_upload_MB",
        "edge_model_download_MB",
    )
    comm_pattern = re.compile(
        rf"^\s*({'|'.join(map(re.escape, comm_keys))}):\s*{number}\s*MB\s*$"
    )
    any_comm_line = re.compile(
        rf"^\s*[a-z][a-z0-9_]*:\s*{number}\s*MB\s*$"
    )
    in_report = False
    for line in text.splitlines():
        if re.fullmatch(r"\s*=== Communication Report ===\s*", line):
            in_report = True
            continue
        if not in_report:
            continue
        match = comm_pattern.fullmatch(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
        elif line.strip() and not any_comm_line.fullmatch(line):
            in_report = False

    runtime_matches = list(re.finditer(patterns["runtime_s"], text))
    traceback_matches = list(
        re.finditer(r"(?m)^Traceback \(most recent call last\):\s*$", text)
    )
    ended_with_traceback = bool(
        traceback_matches
        and (
            not runtime_matches
            or traceback_matches[-1].start() > runtime_matches[-1].start()
        )
    )
    metrics["status"] = (
        "no_final_metric" if not runtime_matches or ended_with_traceback else "complete"
    )
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
        metric_keys = sorted(
            set().union(*(run.keys() for run in runs))
            - {"experiment_id", "seed", "group", "status"}
        )
        for mk in metric_keys:
            vals = [r[mk] for r in runs if mk in r and r[mk] is not None]
            if vals and all(isinstance(v, (int, float)) for v in vals):
                import statistics
                agg[f"{mk}_mean"] = statistics.mean(vals)
                if len(vals) > 1:
                    agg[f"{mk}_std"] = statistics.stdev(vals)
                else:
                    agg[f"{mk}_std"] = 0.0
        agg["status"] = (
            "complete"
            if all(run.get("status") == "complete" for run in runs)
            else "no_final_metric"
        )
        agg["group"] = runs[0].get("group", "unknown")
        aggregated[key] = agg

    return aggregated


def write_csv(rows: list, output_path: str, fieldnames: list = None):
    """Write list of dicts to CSV."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
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
