#!/usr/bin/env python3
"""Aggregate fair-comparison run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "results/fair_comparison_cifar100.csv"
OUTPUTS = {
    "cifar100": DEFAULT_CSV,
    "cifar100_full60": ROOT / "results/fair_comparison_cifar100_full60.csv",
    "ham10000": ROOT / "results/fair_comparison_ham10000.csv",
    "isic2018": ROOT / "results/fair_comparison_isic2018.csv",
}
COLUMNS = [
    "method", "config", "seed", "dataset", "rounds", "best_val_top1",
    "macro_f1", "total_comm_MB", "peak_vram_MB", "runtime_s", "partition_hash",
    "config_hash", "status",
]
KNOWN_PARTITIONS = {
    20260714: "93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1",
    20260715: "40fc6ab8f6157ba946806476ca077da6d1c4197808313b66980e3c973fd2a470",
    20260716: "e13b513ace31b630e008d51bf7edd2ca5f3f899d8ba04ea481463bf23325ddc3",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("base", None)
    if base:
        parent = resolve_yaml((path.parent / base).resolve())
        parent.update(data)
        return parent
    return data


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def infer_method(path: Path, metric: dict[str, Any]) -> str:
    if "best_f1" in metric and "comm_report" in metric:
        return "h-sfp"
    text = path.name.lower()
    if "hsfp" in text or "h-sfp" in text:
        return "h-sfp"
    if "heterosfl" in text or "hetero-sfl" in text:
        return "heterosfl"
    for method in ("splitfl", "federated", "hierfl", "hsfl"):
        if method in text:
            return method
    return "unknown"


def infer_seed(path: Path, metadata: dict[str, Any]) -> int | str:
    seed = metadata.get("resolved_config", {}).get("seed")
    if seed is not None:
        return int(seed)
    for candidate in KNOWN_PARTITIONS:
        if str(candidate) in str(path):
            return candidate
    return ""


def comparison_config(method: str, dataset: str) -> Path | None:
    cifar_names = {
        "splitfl": "plan2_cifar100_splitfl.yaml",
        "federated": "plan2_cifar100_federated.yaml",
        "hierfl": "plan15_cifar100_hierfl.yaml",
        "hsfl": "plan15_cifar100_hsfl.yaml",
        "heterosfl": "plan15_cifar100_heterosfl.yaml",
    }
    ham_names = {
        "h-sfp": "plan_ham10000_hsfp.yaml",
        "splitfl": "plan_ham10000_splitfl.yaml",
        "federated": "plan_ham10000_federated.yaml",
        "hierfl": "plan_ham10000_hierfl.yaml",
        "hsfl": "plan_ham10000_hsfl.yaml",
        "heterosfl": "plan_ham10000_heterosfl.yaml",
    }
    isic_names = {
        "h-sfp": "plan_isic2018_hsfp.yaml",
        "federated": "plan_isic2018_federated.yaml",
        "hierfl": "plan_isic2018_hierfl.yaml",
        "heterosfl": "plan_isic2018_heterosfl.yaml",
    }
    if dataset in ("isic2018", "isic-2018"):
        names = isic_names
    elif dataset == "ham10000":
        names = ham_names
    else:
        names = cifar_names
    return ROOT / "configs/proxy" / names[method] if method in names else None


def first_metric_json(run_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    preferred = sorted(run_dir.rglob("metrics.json"))
    candidates = preferred or [p for p in sorted(run_dir.rglob("*.json")) if p.name != "run_metadata.json"]
    for path in candidates:
        data = load_json(path)
        if data:
            return path, data
    return None, {}


def best_accuracy(metric: dict[str, Any]) -> float | str:
    for key in (
        "best_val_top1", "best_f1", "best_iou", "test_iou",
        "final_test_f1", "test_accuracy",
    ):
        if metric.get(key) is not None:
            value = float(metric[key])
            return value * 100 if value <= 1 else value
    for key in ("validation_f1", "train_accuracy", "f1"):
        values = metric.get(key)
        if isinstance(values, list) and values:
            value = float(max(values))
            return value * 100 if value <= 1 else value
    return ""


def macro_f1(metric: dict[str, Any]) -> float | str:
    """Read a scalar or history while avoiding legacy accuracy-named-as-F1 fields."""
    for key in ("macro_f1", "best_val_macro_f1", "final_test_f1", "test_macro_f1"):
        if metric.get(key) is not None:
            value = metric[key]
            if isinstance(value, list):
                value = max(value) if value else None
            if value is not None:
                value = float(value)
                return value * 100 if value <= 1 else value
    for key in ("validation_macro_f1", "macro_f1_history"):
        values = metric.get(key)
        if isinstance(values, list) and values:
            value = float(max(values))
            return value * 100 if value <= 1 else value
    return ""


def total_comm(metric: dict[str, Any]) -> float | str:
    report = metric.get("comm_report", {})
    if report.get("total_comm_MB") is not None:
        return float(report["total_comm_MB"])
    if metric.get("total_comm_MB") is not None:
        return float(metric["total_comm_MB"])
    comm_keys = [k for k in metric if k.endswith("_MB") and ("upload" in k or "download" in k)]
    return sum(float(metric[k]) for k in comm_keys) if comm_keys else ""


def aggregate_dir(raw_dir: str, dataset_hint: str = "cifar100") -> dict[str, Any]:
    run_dir = Path(raw_dir).resolve()
    metric_path, metric = first_metric_json(run_dir)
    metadata_paths = sorted(run_dir.rglob("run_metadata.json"))
    metadata = load_json(metadata_paths[0]) if metadata_paths else {}
    method = infer_method(run_dir, metric)
    seed = infer_seed(run_dir, metadata)
    resolved = metadata.get("resolved_config", {})
    normalized_hint = "cifar100" if dataset_hint == "cifar100_full60" else dataset_hint
    detected_dataset = resolved.get("dataset", normalized_hint)
    config_path = comparison_config(method, detected_dataset)
    config_label = str(config_path.relative_to(ROOT)) if config_path else ""
    if method == "h-sfp" and detected_dataset != "ham10000":
        config_label = "configs/proxy/plan2_cifar100_hsfp_baseline.yaml+centered_cosine+freeze_backbone+num_workers=0"
    config_hash = metadata.get("resolved_config_hash", "")
    if not resolved and config_path and config_path.exists():
        resolved = resolve_yaml(config_path)
        if seed != "":
            resolved["seed"] = seed
        resolved["output_dir"] = str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir)
        config_hash = stable_hash(resolved)
    peak_bytes = metadata.get("peak_vram_bytes", metric.get("peak_vram_bytes"))
    peak_mb = metric.get("peak_vram_MB", "")
    row = {
        "method": method,
        "config": config_label,
        "seed": seed,
        "dataset": resolved.get("dataset", dataset_hint),
        "rounds": resolved.get("epochs", len(metric.get("validation_f1", metric.get("train_accuracy", []))) or 10),
        "best_val_top1": best_accuracy(metric),
        "macro_f1": macro_f1(metric),
        "total_comm_MB": total_comm(metric),
        "peak_vram_MB": (float(peak_bytes) / 1024**2) if peak_bytes is not None else peak_mb,
        "runtime_s": metric.get("runtime_s", metadata.get("runtime_s", "")),
        "partition_hash": metadata.get("partition_hash", KNOWN_PARTITIONS.get(seed, "")),
        "config_hash": config_hash,
        "status": "complete" if metric_path and best_accuracy(metric) != "" else "incomplete",
    }
    missing = [k for k in ("total_comm_MB", "peak_vram_MB", "runtime_s") if row[k] == ""]
    if row["status"] == "complete" and missing:
        row["status"] = "complete_missing_" + "+".join(missing)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Run output directories to scan recursively")
    parser.add_argument("--dataset", choices=tuple(OUTPUTS), default="cifar100")
    parser.add_argument("--output", help="Destination CSV (defaults according to --dataset)")
    args = parser.parse_args()
    output = Path(args.output) if args.output else OUTPUTS[args.dataset]
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    if output.exists():
        with output.open(newline="") as handle:
            for row in csv.DictReader(handle):
                existing[(row["method"], row["seed"], row["config"])] = row
    for directory in args.run_dirs:
        row = aggregate_dir(directory, args.dataset)
        existing[(str(row["method"]), str(row["seed"]), str(row["config"]))] = row
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(existing.values(), key=lambda r: (r["method"], str(r["seed"]), r["config"])))
    print(f"Wrote {len(existing)} rows to {output}")


if __name__ == "__main__":
    main()
