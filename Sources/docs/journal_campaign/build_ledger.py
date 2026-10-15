#!/usr/bin/env python3
"""Generate ALL_SIMULATION_RESULTS.md from the matrix + source CSVs.
Full machine precision preserved; no rounding before aggregation."""
import csv
import subprocess
import statistics as st
import datetime

REPO = "/home/jackson/Desktop/Split-Federated-Merging-Learning"
OUT = f"{REPO}/ALL_SIMULATION_RESULTS.md"

commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
now = datetime.datetime.now().isoformat(timespec="seconds")

matrix = list(csv.DictReader(open(f"{REPO}/docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv")))

status_counts = {}
for r in matrix:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1


def ci95(vals):
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0, mean, mean
    sd = st.stdev(vals)
    # t-critical for small n (two-sided 95%); table for df=1..4 (n=2..5 seeds)
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
    t = t_table.get(n - 1, 1.96)
    half = t * sd / (n ** 0.5)
    return mean, sd, mean - half, mean + half


# ---- Build aggregate bank from existing verified/completed_unvalidated proxy+legacy cells ----
def load_seed_group(path, method_field, value_field, filter_fn=None):
    groups = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if filter_fn and not filter_fn(row):
                continue
            v = row.get(value_field)
            if not v:
                continue
            groups.setdefault(row[method_field], []).append(float(v))
    return groups

isic_groups = load_seed_group(f"{REPO}/results/fair_comparison_isic2018.csv", "method", "best_iou_pct")
cifar_proxy_groups = load_seed_group(f"{REPO}/results/fair_comparison_cifar100.csv", "method", "best_val_top1")
cifar_legacy_groups = load_seed_group(f"{REPO}/results/fair_comparison_cifar100_full60.csv", "method", "test_top1")
ham_proxy_groups = load_seed_group(
    f"{REPO}/results/fair_comparison_ham10000.csv", "method", "best_val_top1",
    filter_fn=lambda r: r["method"] != "unknown",
)

cell_bank_rows = []
agg_bank_rows = []
cid = 0
aid = 0


def add_cell(prospective_table, protocol_id, dataset, task, method, variant, condition, seed,
             checkpoint, metric, value, unit, status, raw_path):
    global cid
    cid += 1
    cell_bank_rows.append([
        f"cell_{cid:04d}", prospective_table, protocol_id, dataset, task, method, variant,
        condition, seed, checkpoint, metric, repr(value) if isinstance(value, float) else value,
        unit, status, raw_path,
    ])


def add_agg(prospective_table, protocol_id, dataset, method, variant, condition, metric, seed_list, values, unit, status):
    global aid
    aid += 1
    mean, sd, lo, hi = ci95(values)
    agg_bank_rows.append([
        f"agg_{aid:04d}", prospective_table, protocol_id, dataset, method, variant, condition, metric,
        ";".join(str(s) for s in seed_list), len(values),
        repr(mean), repr(sd), repr(lo), repr(hi), unit, status,
    ])


# ISIC-2018 cells + aggregates
isic_seed_map = {"federated": [20260714, 20260715, 20260716], "hierfl": [20260714, 20260715, 20260716],
                  "heterosfl": [20260714, 20260715, 20260716], "h-sfp": [20260714, 20260715, 20260716],
                  "splitfl": [20260714, 20260715, 20260716], "hsfl": [20260714, 20260715, 20260716]}
for method, vals in isic_groups.items():
    seeds = isic_seed_map.get(method, list(range(len(vals))))
    for s, v in zip(seeds, vals):
        add_cell("main_matched_comparison_isic2018", "proxy_10round_dateseed", "isic2018", "segmentation",
                  method, "baseline", "clean_iid", s, "best_validation_iou", "test_iou", v, "%",
                  "VERIFIED", "results/fair_comparison_isic2018.csv")
    add_agg("main_matched_comparison_isic2018", "proxy_10round_dateseed", "isic2018", method, "baseline",
             "clean_iid", "test_iou", seeds, vals, "%", "VERIFIED_PROXY_NOT_FINAL")

# ISIC ablation
isic_abl = [("baseline_hsfp", 50.28, 64.95), ("hsfp_memory", 47.92, 62.63),
            ("hsfp_memory_dropout", 47.83, 62.37), ("hsfp_memory_reliability", 46.74, 61.69)]
for preset, iou, dice in isic_abl:
    add_cell("ehsfp_ablation_isic2018", "proxy_10round_dateseed", "isic2018", "segmentation", "h-sfp",
              preset, "clean_iid", 20260714, "best_validation_iou", "test_iou", iou, "%",
              "COMPLETED_UNVALIDATED", "docs/optimization_loop/logs/plan42_isic_hsfp_ablation/")
    add_cell("ehsfp_ablation_isic2018", "proxy_10round_dateseed", "isic2018", "segmentation", "h-sfp",
              preset, "clean_iid", 20260714, "best_validation_iou", "test_dice", dice, "%",
              "COMPLETED_UNVALIDATED", "docs/optimization_loop/logs/plan42_isic_hsfp_ablation/")

# CIFAR-100 proxy
cifar_proxy_seed_map = {m: [20260714, 20260715, 20260716] for m in cifar_proxy_groups}
for method, vals in cifar_proxy_groups.items():
    seeds = cifar_proxy_seed_map[method]
    variant = "centered_cosine+FREEZE+EVALFIX" if method == "h-sfp" else "baseline"
    for s, v in zip(seeds, vals):
        add_cell("main_matched_comparison_cifar100_proxy", "proxy_10round_dateseed", "cifar100",
                  "classification", method, variant, "clean_iid", s, "best_val_top1", "accuracy", v, "%",
                  "VERIFIED", "results/fair_comparison_cifar100.csv")
    add_agg("main_matched_comparison_cifar100_proxy", "proxy_10round_dateseed", "cifar100", method, variant,
             "clean_iid", "accuracy", seeds, vals, "%", "VERIFIED_PROXY_NOT_FINAL")

# CIFAR-100 legacy60
for method, vals in cifar_legacy_groups.items():
    seeds = [20260714, 20260715, 20260716]
    for s, v in zip(seeds, vals):
        add_cell("main_matched_comparison_cifar100_legacy60", "legacy_60round_dateseed", "cifar100",
                  "classification", method, "baseline", "clean_iid", s, "test_top1_over_60rounds", "accuracy",
                  v, "%", "COMPLETED_UNVALIDATED", "results/fair_comparison_cifar100_full60.csv")
    add_agg("main_matched_comparison_cifar100_legacy60", "legacy_60round_dateseed", "cifar100", method,
             "baseline", "clean_iid", "accuracy", seeds, vals, "%", "COMPLETED_UNVALIDATED")

# CIFAR-100 ablation
add_cell("ehsfp_ablation_cifar100", "legacy_60round_intseed_PRE_CF6454C", "cifar100",
          "classification", "h-sfp", "baseline_hsfp", "clean_iid", 0, "best_val_top1", "accuracy",
          8.10, "%", "SUPERSEDED",
          "results/journal/logs/abl_classification_h-sfp_cifar_our_resnet50_5_10_baseline_hsfp_s0_RERUN.log")
add_cell("ehsfp_ablation_cifar100", "legacy_60round_intseed", "cifar100", "classification", "h-sfp",
          "baseline_hsfp", "clean_iid", 0, "best_val_top1", "accuracy", 11.18, "%", "VERIFIED_NEW",
          "docs/optimization_loop/logs/plan43_cifar100_baseline_hsfp_rerun/run.log")
cifar_abl = [
    ("hsfp_memory", 11.28, 1430.47, 6489),
    ("hsfp_memory_dropout", 11.20, 1426.39, 6175),
    ("hsfp_memory_reliability", 11.26, 1430.47, 6820),
]
for preset, acc, comm, rt in cifar_abl:
    add_cell("ehsfp_ablation_cifar100", "legacy_60round_intseed", "cifar100", "classification",
              "h-sfp", preset, "clean_iid", 0, "best_val_top1", "accuracy", acc, "%",
              "COMPLETED_UNVALIDATED",
              "results/journal/logs/abl_classification_h-sfp_cifar_our_resnet50_5_10_%s_s0_RERUN.log" % preset)

# HAM10000 proxy (H-SFP flagged INVALID)
ham_proxy_seed_map = {m: [20260714, 20260715, 20260716] for m in ham_proxy_groups}
for method, vals in ham_proxy_groups.items():
    seeds = ham_proxy_seed_map[method]
    status = "INVALID" if method == "h-sfp" else "VERIFIED"
    for s, v in zip(seeds, vals):
        add_cell("main_matched_comparison_ham10000_proxy", "proxy_10round_dateseed", "ham10000",
                  "classification", method, "baseline", "clean_iid", s, "best_val_top1", "accuracy", v, "%",
                  status, "results/fair_comparison_ham10000.csv")
    if method != "h-sfp":
        add_agg("main_matched_comparison_ham10000_proxy", "proxy_10round_dateseed", "ham10000", method,
                 "baseline", "clean_iid", "accuracy", seeds, vals, "%", "VERIFIED_PROXY_NOT_FINAL")
    else:
        add_agg("main_matched_comparison_ham10000_proxy", "proxy_10round_dateseed", "ham10000", method,
                 "baseline", "clean_iid", "accuracy", seeds, vals, "%",
                 "INVALID_SEED_NOT_APPLIED_BUG")

# HAM10000 ablation
ham_abl = [
    ("hsfp_memory", 54.72, 16063.54, "27/60"),
    ("hsfp_memory_dropout", 50.77, 15473.14, "42/60"),
    ("hsfp_memory_reliability", 55.82, 16063.54, "56/60"),
]
for preset, acc, comm, best_epoch in ham_abl:
    add_cell("ehsfp_ablation_ham10000", "legacy_60round_intseed", "ham10000", "classification", "h-sfp",
              preset, "clean_iid", 0, "best_val_top1", "accuracy", acc, "%",
              "COMPLETED_UNVALIDATED", "docs/optimization_loop/logs/plan39_ablation_rerun_ham_v2/")

# Conference references (as cells, status REFERENCE_ONLY, protocol_id distinct)
conf_refs = [
    ("cifar10", "classification", "accuracy", 67.44, 0.85), ("cifar100", "classification", "accuracy", 55.10, 0.95),
    ("ham10000", "classification", "accuracy", 80.86, 0.75), ("imagenet1k", "classification", "accuracy", 27.9, 0.3),
    ("isic2018", "segmentation", "iou", 68.3, 0.5), ("isic2018", "segmentation", "dice", 79.5, 0.4),
]
for dataset, task, metric, mean, std in conf_refs:
    add_cell("conference_reference", "conference_accepted_eccv2026", dataset, task, "h-sfp",
              "conference_accepted", "clean_iid", "N/A", "N/A", metric, mean, "%", "REFERENCE_ONLY",
              "accepted ECCV 2026 manuscript (external)")

print("CELL_BANK_ROWS", len(cell_bank_rows))
print("AGG_BANK_ROWS", len(agg_bank_rows))

import json
with open("/home/jackson/.claude/jobs/317bbf05/tmp/cell_bank.json", "w") as f:
    json.dump(cell_bank_rows, f)
with open("/home/jackson/.claude/jobs/317bbf05/tmp/agg_bank.json", "w") as f:
    json.dump(agg_bank_rows, f)
with open("/home/jackson/.claude/jobs/317bbf05/tmp/meta.json", "w") as f:
    json.dump({"commit": commit, "dirty": dirty, "now": now, "status_counts": status_counts, "total_rows": len(matrix)}, f)
