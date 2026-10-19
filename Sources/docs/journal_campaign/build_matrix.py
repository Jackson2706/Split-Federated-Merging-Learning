#!/usr/bin/env python3
"""Generate docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv from existing source-of-truth
CSVs/logs. Read-only against source data; never fabricates a value not present in a source file.
"""
import csv
import statistics as st

REPO = "/home/jackson/Desktop/Split-Federated-Merging-Learning"
OUT = f"{REPO}/docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv"

COLUMNS = [
    "experiment_id", "priority", "protocol_id", "dataset", "task", "backbone", "method",
    "variant", "condition", "rounds", "local_epochs", "client_count", "edge_count",
    "participation_rate", "aggregation_intervals", "seed", "config_path", "config_hash",
    "partition_hash", "code_commit", "status", "attempt", "runtime_s", "raw_result_path",
    "checkpoint_path", "metric_validation", "communication_validation", "notes",
]

rows = []


def add(**kw):
    r = {c: kw.get(c, "MISSING") for c in COLUMNS}
    rows.append(r)


# ============================================================================
# 1. ISIC-2018 fair comparison — proxy protocol, 10 rounds, seeds 20260714/15/16
# ============================================================================
isic_fc = {}
with open(f"{REPO}/results/fair_comparison_isic2018.csv") as f:
    for row in csv.DictReader(f):
        isic_fc.setdefault(row["method"], []).append(row)

for method, entries in isic_fc.items():
    for e in entries:
        add(
            experiment_id=f"isic2018_proxy_{method}_s{e['seed']}",
            priority="P1_reuse",
            protocol_id="proxy_10round_dateseed",
            dataset="isic2018", task="segmentation", backbone="resnet50",
            method=method, variant="baseline", condition="clean_iid",
            rounds=10, seed=e["seed"],
            config_path=f"configs/proxy/plan_isic2018_{method if method!='h-sfp' else 'hsfp'}.yaml",
            code_commit="pre-session (various, not individually hashed)",
            status="VERIFIED" if e.get("status") == "complete" else "COMPLETED_UNVALIDATED",
            runtime_s=e.get("runtime_s") or "MISSING",
            raw_result_path="results/fair_comparison_isic2018.csv",
            metric_validation="best_iou/test_iou present; macro-F1 n/a for segmentation",
            communication_validation="MISSING (not yet reconciled packet-by-packet for this row)",
            notes=e.get("notes", ""),
        )

# ============================================================================
# 2. ISIC-2018 H-SFP ablation — proxy protocol, 10 rounds, single seed 20260714
# ============================================================================
isic_abl = [
    ("baseline_hsfp", 50.28, 64.95, 3371),
    ("hsfp_memory", 47.92, 62.63, 3367),
    ("hsfp_memory_dropout", 47.83, 62.37, 3216),
    ("hsfp_memory_reliability", 46.74, 61.69, 3349),
]
for preset, iou, dice, rt in isic_abl:
    add(
        experiment_id=f"isic2018_proxy_hsfp_abl_{preset}_s20260714",
        priority="P1_reuse",
        protocol_id="proxy_10round_dateseed",
        dataset="isic2018", task="segmentation", backbone="resnet50",
        method="h-sfp", variant=preset, condition="clean_iid",
        rounds=10, seed=20260714,
        config_path="configs/proxy/plan_isic2018_hsfp.yaml",
        code_commit="pre-session (2026-07-22 ablation campaign, commit 9e785b9 tree + uncommitted SplitFL fix)",
        status="COMPLETED_UNVALIDATED",
        runtime_s=rt,
        raw_result_path="docs/optimization_loop/logs/plan42_isic_hsfp_ablation/isic_hsfp_abl_%s_seed20260714.log" % preset,
        metric_validation=f"test_iou={iou}, test_dice={dice}; single seed, not independently re-derived from saved predictions",
        communication_validation="MISSING",
        notes="Single seed (n=1), proxy protocol only. NOT comparable to Section 8.2's 5-seed frozen-protocol requirement.",
    )

# ============================================================================
# 3. CIFAR-100 fair comparison — proxy protocol, 10 rounds
# ============================================================================
with open(f"{REPO}/results/fair_comparison_cifar100.csv") as f:
    for e in csv.DictReader(f):
        add(
            experiment_id=f"cifar100_proxy_{e['method']}_s{e['seed']}",
            priority="P1_reuse",
            protocol_id="proxy_10round_dateseed",
            dataset="cifar100", task="classification", backbone="resnet50 (proxy; frozen protocol calls for resnet18)",
            method=e["method"], variant=("centered_cosine+FREEZE+EVALFIX" if e["method"] == "h-sfp" else "baseline"),
            condition="clean_iid", rounds=e["rounds"], seed=e["seed"],
            config_path=e["config"], config_hash=e.get("config_hash", "MISSING"),
            partition_hash=e.get("partition_hash", "MISSING"),
            code_commit="pre-session (various)",
            status="VERIFIED" if e.get("status", "").startswith("complete") else "COMPLETED_UNVALIDATED",
            runtime_s=e.get("runtime_s") or "MISSING",
            raw_result_path="results/fair_comparison_cifar100.csv",
            metric_validation=f"best_val_top1={e['best_val_top1']}; macro_f1={e.get('macro_f1') or 'MISSING'}",
            communication_validation="MISSING",
            notes=e.get("status", ""),
        )

# ============================================================================
# 4. CIFAR-100 fair comparison — legacy 60-round protocol
# ============================================================================
with open(f"{REPO}/results/fair_comparison_cifar100_full60.csv") as f:
    for e in csv.DictReader(f):
        add(
            experiment_id=f"cifar100_legacy60_{e['method']}_s{e['seed']}",
            priority="P1_reuse",
            protocol_id="legacy_60round_dateseed",
            dataset="cifar100", task="classification", backbone="resnet50 (frozen protocol calls for resnet18)",
            method=e["method"], variant="baseline", condition="clean_iid",
            rounds=60, seed=e["seed"],
            code_commit="pre-session (various)",
            status="COMPLETED_UNVALIDATED",
            runtime_s=e.get("runtime_s"),
            raw_result_path="results/fair_comparison_cifar100_full60.csv",
            metric_validation=f"test_top1={e['test_top1']}; macro_f1=MISSING (never computed for this table)",
            communication_validation=f"total_comm_MB={e.get('total_comm_MB')} (not yet packet-reconciled)",
            notes="COMPLETE_UNVALIDATED per repo convention: per-seed std/F1-validity/comm-accounting checks not yet independently applied.",
        )

# ============================================================================
# 5. CIFAR-100 H-SFP ablation — legacy 60-round protocol, single seed s0
# ============================================================================
add(
    experiment_id="cifar100_legacy60_hsfp_abl_baseline_hsfp_s0_PRE_CF6454C",
    priority="P3_diagnostic", protocol_id="legacy_60round_intseed_PRE_CF6454C",
    dataset="cifar100", task="classification", backbone="resnet50 (frozen protocol calls for resnet18)",
    method="h-sfp", variant="baseline_hsfp", condition="clean_iid",
    rounds=60, seed=0,
    config_path="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml",
    code_commit="pre-cf6454c (July 9, OLD [64,32,32] client feature boundary)",
    status="SUPERSEDED",
    runtime_s=25000,
    raw_result_path="results/journal/logs/abl_classification_h-sfp_cifar_our_resnet50_5_10_baseline_hsfp_s0_RERUN.log",
    metric_validation="best_val_top1=8.10; macro_f1=MISSING",
    communication_validation="total_comm_MB=30414.23; STALE -- pre-cf6454c architecture, not comparable to the memory-preset rows' post-cf6454c architecture (512x per-class-entry size difference, definitively traced 2026-07-23, Section 5.4 CLOSED)",
    notes="SUPERSEDED 2026-07-24 by cifar100_legacy60_hsfp_abl_baseline_hsfp_s0 (rerun on current code). Preserved for provenance only, not reusable -- measured on the pre-cf6454c client architecture, not the same model as the memory-preset rows.",
)
add(
    experiment_id="cifar100_legacy60_hsfp_abl_baseline_hsfp_s0",
    priority="P1_reuse", protocol_id="legacy_60round_intseed",
    dataset="cifar100", task="classification", backbone="resnet50 (frozen protocol calls for resnet18)",
    method="h-sfp", variant="baseline_hsfp", condition="clean_iid",
    rounds=60, seed=0,
    config_path="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml",
    code_commit="2026-07-24 rerun, current code (post cf6454c, pooled [128,1,1] client boundary)",
    status="VERIFIED_NEW",
    runtime_s=6491,
    raw_result_path="docs/optimization_loop/logs/plan43_cifar100_baseline_hsfp_rerun/run.log",
    checkpoint_path="outputs/proxy/plan43_cifar100_baseline_hsfp_rerun_s0/*/*/checkpoint.pt (best epoch 56)",
    metric_validation="best_val_top1=11.18; test_acc=11.18; macro_f1=MISSING",
    communication_validation="total_comm_MB=1450.13 (client_to_edge 726.88 / edge_to_client 620.11 / edge_to_cloud 80.48 / cloud_to_edge 22.66) -- now genuinely comparable in scale to the memory-preset rows' ~1430 MB (previously 30414.23 MB pre-cf6454c), confirming the Section 5.4 architecture-change explanation",
    notes="RESTORES the ablation comparator retracted 2026-07-23. Rerun 2026-07-24 on current code, same config/seed/ablation-flag as the 3 memory-preset rows below. Valid comparison: baseline 11.18% vs hsfp_memory 11.28% / hsfp_memory_dropout 11.20% / hsfp_memory_reliability 11.26% -- a ~0.02-0.10pp gap, essentially within single-seed noise, NOT the previously-reported (and retracted) +3.18pp 'memory improvement' which was entirely a cross-architecture artifact. Single seed (n=1) -- no meaningful memory-vs-baseline conclusion can be drawn from one seed; the 5-seed frozen protocol (Section 8.2) is required for a real ablation finding.",
)
cifar_abl_memory = [
    ("hsfp_memory", 11.28, 1430.47, 6489),
    ("hsfp_memory_dropout", 11.20, 1426.39, 6175),
    ("hsfp_memory_reliability", 11.26, 1430.47, 6820),
]
for preset, acc, comm, rt in cifar_abl_memory:
    add(
        experiment_id=f"cifar100_legacy60_hsfp_abl_{preset}_s0",
        priority="P1_reuse",
        protocol_id="legacy_60round_intseed",
        dataset="cifar100", task="classification", backbone="resnet50 (frozen protocol calls for resnet18)",
        method="h-sfp", variant=preset, condition="clean_iid",
        rounds=60, seed=0,
        config_path="configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml",
        code_commit="pre-session (plan37 rerun, post cf6454c ablation-precedence fix, pooled [128,1,1] client boundary)",
        status="COMPLETED_UNVALIDATED",
        runtime_s=rt,
        raw_result_path=f"results/journal/logs/abl_classification_h-sfp_cifar_our_resnet50_5_10_{preset}_s0_RERUN.log",
        metric_validation=f"best_val_top1={acc}; macro_f1=MISSING",
        communication_validation=f"total_comm_MB={comm}; now confirmed genuinely comparable to the restored baseline_hsfp row (1450.13 MB) -- same client architecture, Section 5.4 CLOSED",
        notes="Single seed (n=1). Requires 5-seed frozen rerun (Section 8.2). Now has a genuinely valid baseline_hsfp comparator (cifar100_legacy60_hsfp_abl_baseline_hsfp_s0, rerun 2026-07-24) -- see that row for the restored comparison.",
    )

# ============================================================================
# 6. HAM10000 fair comparison — proxy protocol, 10 rounds (H-SFP row INVALID)
# ============================================================================
with open(f"{REPO}/results/fair_comparison_ham10000.csv") as f:
    for e in csv.DictReader(f):
        if e["method"] == "unknown":
            add(
                experiment_id=f"ham10000_proxy_UNKNOWN_s{e['seed']}",
                priority="P3_diagnostic",
                protocol_id="proxy_10round_dateseed",
                dataset="ham10000", task="classification", method="unknown",
                seed=e["seed"], status="FAILED_INFRA",
                raw_result_path="results/fair_comparison_ham10000.csv",
                notes="incomplete row in source CSV: config hash present, no metric captured",
            )
            continue
        is_hsfp = e["method"] == "h-sfp"
        add(
            experiment_id=f"ham10000_proxy_{e['method']}_s{e['seed']}",
            priority="P0_repair_needed" if is_hsfp else "P1_reuse",
            protocol_id="proxy_10round_dateseed",
            dataset="ham10000", task="classification", backbone="resnet50",
            method=e["method"], variant="baseline", condition="clean_iid",
            rounds=e["rounds"], seed=e["seed"],
            config_path=e["config"], config_hash=e.get("config_hash", "MISSING"),
            partition_hash=e.get("partition_hash", "MISSING"),
            code_commit="pre-session (various)",
            status="INVALID" if is_hsfp else "VERIFIED",
            runtime_s=e.get("runtime_s") or "MISSING",
            raw_result_path="results/fair_comparison_ham10000.csv",
            metric_validation=f"best_val_top1={e['best_val_top1']}; macro_f1={e.get('macro_f1') or 'MISSING'}",
            communication_validation="MISSING",
            notes=("INVALID: bit-identical best_val_top1 (10.983524712930603) across all 3 different seeds. "
                   "ROOT CAUSE CONFIRMED 2026-07-23: training collapse (fp16-autocast-overflow in prototype "
                   "std computation -> NaN cloud logits -> argmax always class 0 -> accuracy locks to the "
                   "class-0 validation fraction, 220/2003), NOT a seed-plumbing bug -- archived partition "
                   "hashes already differed by seed. Same fp16 mechanism fixed 2026-07-21 for the reliability "
                   "path; this campaign predates that fix. FIX VERIFIED 2026-07-23 via GPU rerun (2 seeds x 3 "
                   "epochs): finite losses, predictions spanning all 7 classes, genuinely distinct accuracy "
                   "per seed (11.98%/11.93%), partition/client-sampling/cloud-init hashes differ by seed "
                   "(client/edge hashes match across seeds by design -- pretrained ImageNet ResNet50 weights, "
                   "not seed-derived). See DECISIONS.md. Section 5.2 CLOSED. This row remains INVALID as "
                   "reported (not retroactively promoted) -- a full 3-seed 10-round proxy rerun with the fix "
                   "is still required before a new valid aggregate exists." if is_hsfp else e.get("status", "")),
        )

# ============================================================================
# 7. HAM10000 H-SFP ablation — legacy 60-round protocol, single seed s0
# ============================================================================
ham_abl = [
    ("hsfp_memory", 54.72, 16063.54, "27/60"),
    ("hsfp_memory_dropout", 50.77, 15473.14, "42/60"),
    ("hsfp_memory_reliability", 55.82, 16063.54, "56/60"),
]
for preset, acc, comm, best_epoch in ham_abl:
    add(
        experiment_id=f"ham10000_legacy60_hsfp_abl_{preset}_s0",
        priority="P1_reuse",
        protocol_id="legacy_60round_intseed",
        dataset="ham10000", task="classification", backbone="resnet50",
        method="h-sfp", variant=preset, condition="clean_iid",
        rounds=60, seed=0,
        config_path="configs/classification/h-sfp/ham10000_our_resnet50_5_10.yaml",
        code_commit="pre-session (plan39_v2, post fp16-prototype-std fix)",
        status="COMPLETED_UNVALIDATED",
        runtime_s="MISSING (best-epoch only recorded)",
        raw_result_path="docs/optimization_loop/logs/plan39_ablation_rerun_ham_v2/",
        checkpoint_path=f"best epoch {best_epoch}",
        metric_validation=f"best_val_top1={acc}; macro_f1=MISSING",
        communication_validation=f"total_comm_MB={comm}",
        notes="Single seed (n=1). No baseline_hsfp comparator in this pass (HAM10000 baseline row is INVALID above). Needs 5-seed frozen rerun + a valid baseline.",
    )

# ============================================================================
# 8. Historical registry entry (kept as reference, not reusable as frozen data)
# ============================================================================
add(
    experiment_id="plan14_freeze_centered_cifar100_hsfp",
    priority="P3_diagnostic",
    protocol_id="legacy_10round_dateseed_pre_evalfix_variant",
    dataset="cifar100", task="classification", method="h-sfp",
    variant="baseline_hsfp+centered_cosine+FREEZE+EVALFIX",
    rounds=10, seed="20260714;20260715;20260716",
    code_commit="pre-cf6454c",
    status="VERIFIED",
    raw_result_path="docs/optimization_loop/EXPERIMENT_REGISTRY.csv",
    metric_validation="best_val_top1 mean=10.29, std=0.37 (n=3)",
    notes="Same underlying data family as row 3's H-SFP proxy result above; kept for provenance, not double-counted.",
)

# ============================================================================
# 9. Conference reference values (never reruns; clearly separate protocol_id)
# ============================================================================
conf_refs = [
    ("cifar10", "classification", "alexnet", "accuracy", 67.44, 0.85, "%"),
    ("cifar100", "classification", "resnet18", "accuracy", 55.10, 0.95, "%"),
    ("ham10000", "classification", "resnet50", "accuracy", 80.86, 0.75, "%"),
    ("imagenet1k", "classification", "resnet50", "accuracy", 27.9, 0.3, "%"),
    ("isic2018", "segmentation", "resnet50-unet", "iou", 68.3, 0.5, "%"),
    ("isic2018", "segmentation", "resnet50-unet", "dice", 79.5, 0.4, "%"),
]
for dataset, task, backbone, metric, mean, std, unit in conf_refs:
    add(
        experiment_id=f"conf_ref_{dataset}_{metric}",
        priority="REFERENCE_ONLY",
        protocol_id="conference_accepted_eccv2026",
        dataset=dataset, task=task, backbone=backbone, method="h-sfp", variant="conference_accepted",
        condition="clean_iid", seed="N/A (published aggregate)",
        code_commit="N/A (published, predates this repo's journal-extension work)",
        status="VERIFIED",
        raw_result_path="accepted ECCV 2026 H-SFP manuscript (external, not in this repo)",
        metric_validation=f"{metric}=mean {mean} +- std {std} {unit}",
        notes="Reference value only. Never to be combined with any local rerun's communication/accuracy from a different configuration.",
    )
add(
    experiment_id="conf_ref_comm_hsfp_A",
    priority="REFERENCE_ONLY", protocol_id="conference_accepted_eccv2026",
    dataset="cifar_or_ham (config A)", task="classification", method="h-sfp", variant="H-SFP(A)",
    rounds=200, condition="clean_iid", seed="N/A",
    status="VERIFIED", raw_result_path="accepted ECCV 2026 manuscript (external)",
    communication_validation="13.8 GB over 200 rounds",
    notes="Reference communication total for configuration (A). Do not pair with configuration (C)'s accuracy or vice versa.",
)
add(
    experiment_id="conf_ref_comm_hsfp_C",
    priority="REFERENCE_ONLY", protocol_id="conference_accepted_eccv2026",
    dataset="cifar_or_ham (config C)", task="classification", method="h-sfp", variant="H-SFP(C)",
    rounds=200, condition="clean_iid", seed="N/A",
    status="VERIFIED", raw_result_path="accepted ECCV 2026 manuscript (external)",
    communication_validation="10.94 GB over 200 rounds",
    notes="Reference communication total for configuration (C). Do not pair with configuration (A)'s accuracy or vice versa.",
)

# ============================================================================
# 10. Known blockers as explicit rows (not silently omitted)
# ============================================================================
add(
    experiment_id="cifar100_proxy_hsfp_full_e_verification_s20260714",
    priority="P1_reuse", protocol_id="proxy_10round_dateseed",
    dataset="cifar100", task="classification", backbone="resnet50",
    method="h-sfp", variant="full_e_hsfp", condition="clean_iid",
    rounds=10, seed=20260714,
    config_path="configs/proxy/plan2_cifar100_hsfp_full_e.yaml",
    code_commit="9e785b9 tree + uncommitted 2026-07-23 fixes",
    status="COMPLETED_UNVALIDATED",
    runtime_s=4052.31,
    raw_result_path="docs/optimization_loop/logs/deadlock_repro_2026-07-23/run_full_10epoch.log",
    checkpoint_path="outputs/deadlock_repro_full/full_e_hsfp_seed20260714/*/*/checkpoint.pt (best epoch 9)",
    metric_validation="best_val_top1=7.20; test_acc=7.20; macro_f1=MISSING",
    communication_validation="total_comm_MB=129.37, route breakdown present (client_to_edge 68.32 / edge_to_client 51.68 / edge_to_cloud 9.38)",
    notes="DEADLOCK VERIFICATION RUN, not a reused result: this is the first-ever completed end-to-end full_e_hsfp run at 200-client proxy scale. Exit 0, finite losses all 10 rounds, all 4 components (memory/PRC/reliability/dropout) confirmed genuinely active via runtime_counters.jsonl (prc.calls=180960 100% nonzero, memory.reads=112250, reliability.aggregation_calls=58 nonuniform, dropout.apply_calls=48). See DECISIONS.md 'DEADLOCK DOES NOT REPRODUCE' entry. Single seed/dataset -- needs 5-seed + multi-dataset extension before VERIFIED.",
)
add(
    experiment_id="BLOCKER_full_e_hsfp_autograd_deadlock",
    priority="P1_reuse", protocol_id="N/A",
    method="h-sfp", variant="full_e_hsfp / hsfp_memory_reliability_prc",
    status="RESOLVED_PENDING_FULL_VERIFICATION",
    notes="RESOLVED 2026-07-23 (was BLOCKED_RESOURCE): full_e_hsfp completed a clean 10-round, 200-client CIFAR-100 proxy run end-to-end (exit 0, finite losses, all components genuinely active -- see isic2018_proxy_hsfp_full_e_verification_s20260714 row and DECISIONS.md). The deadlock does not reproduce with current code; most likely already fixed by 3 earlier rounds (PRC batching, .item()-sync removal, max_replay_batch capping) and simply never re-verified end-to-end, or resolved by environment/library drift since 2026-07-14/16. No longer a structural/sudo-dependent blocker. Remaining work is ordinary verification, not repair: extend to HAM10000/ISIC-2018, the other 4 final seeds, and the full 5-seed frozen protocol; independently confirm hsfp_memory_reliability_prc alone (not yet run, shares the same PRC code path as full_e_hsfp so very likely fine, but unverified).",
)
add(
    experiment_id="BLOCKER_metric_checkpoint_correctness",
    priority="P1_reuse", protocol_id="N/A",
    method="all classification + segmentation methods", variant="Section 5.3 infrastructure",
    status="RESOLVED_PENDING_FULL_VERIFICATION",
    notes="RESOLVED 2026-07-23 (was PARTIAL): classification/training_metrics.py saves raw predictions/labels (.npz, no training-loop metric values -- enforced by a key-set check) at best_val_predictions.npz / last_round_predictions.npz for all 4 methods with real macro-F1 (H-SFP/HSFL/SplitFL/HeteroSFL), none skipped. Standalone scripts/journal_experiments/validate_metrics.py independently recomputes accuracy/macro-F1/per-class-recall/confusion-matrix from the raw arrays only. All 4 classification + 6 segmentation output JSONs now expose separate last_round_validation/best_validation/selected_checkpoint records (fixed a real conflation: HeteroSFL's 'best_f1' key actually held accuracy, renamed to best_val_top1); checkpoint-selection policies unchanged. Segmentation's 5 duplicate compute_iou_and_dice copies de-duplicated into one canonical, docstring-frozen function (threshold>0.5, whole-batch flattened pixel aggregate, empty union/denominator->0/0); all 6 call sites confirmed importing the shared function. Independently re-verified: 15/15 relevant tests genuinely pass on rerun (test_classification_metric_reporting, test_segmentation_metric_protocol, test_research_metrics), including the deliberately-imbalanced accuracy!=macro-F1 case and the empty-mask edge case. CPU-only, no GPU needed. Status is RESOLVED_PENDING_FULL_VERIFICATION not VERIFIED because the new artifact-saving capability has not yet been exercised in a real full campaign run -- historical CSV rows' MISSING macro_f1 fields are not retroactively filled.",
)
add(
    experiment_id="BLOCKER_communication_ledger_reconciliation",
    priority="P1_reuse", protocol_id="N/A",
    method="all methods", variant="Section 5.4 infrastructure",
    status="RESOLVED_PENDING_FULL_VERIFICATION",
    notes="RESOLVED 2026-07-23 (was PARTIAL): definitively traced the CIFAR-100 ablation's 30414->1430 MB total_comm_MB 'collapse' to a cross-code-version comparison artifact -- baseline_hsfp predates commit cf6454c's client-architecture change ([64,32,32] unpooled -> [128,1,1] pooled, exactly 512x per-class-entry size reduction), NOT a memory-driven communication saving or a silently-skipped transmission. Confirmed via archived route-subtotal arithmetic AND a CPU regression invoking the real _client_ssl_extraction_phase with memory on/off proving byte-identical output. Added a from-first-principles packet-by-packet reconciliation test (tests/test_communication_accounting.py). Independently re-verified: 6/6 tests genuinely pass (bare pytest-style functions, invoked directly since unittest doesn't discover them). Consequence: cifar100_legacy60_hsfp_abl_baseline_hsfp_s0 is marked INVALID pending a rerun on current code -- see that row. full_e_hsfp's nonzero-module-counters requirement was already satisfied by the Section 5.1 GPU verification run.",
)
add(
    experiment_id="BLOCKER_per_round_history_logging",
    priority="P1_reuse", protocol_id="N/A",
    method="h-sfp (classification + segmentation)", variant="Section 5.5 infrastructure",
    status="RESOLVED_PENDING_FULL_VERIFICATION",
    notes="RESOLVED 2026-07-23 (was NOT_STARTED, the last Section 5 blocker): added shared ehsfp/history_logger.py -- one durable, flushed+fsynced history.csv row per completed round (round, elapsed_time_s, train_loss, validation metric name+value, per-round communication deltas by route reusing ehsfp/communication.py, phase CPU/RSS/GPU-peak columns, episodic memory occupancy, event column), frozen schema documented in its docstring. Wired into classification/H-SFP/hierarchy.py and segmentation/H-SFP/hierarchy.py's training loops, snapshotting communication state at each round's start so deltas (not just cumulative totals) are captured. H-SFP does not currently track phase-level CPU/RSS/GPU-peak history (unlike some other methods) -- those columns are left present but empty, an honest reported gap, not fabricated. Independently re-verified: a mid-write-interruption test (2 of 5 intended rounds, logger reference dropped mid-run) confirms the partial file stays valid/parseable with a reopenable schema-checked header; 10/10 relevant tests genuinely pass on rerun (test_history_logger + test_research_metrics), plus the 6 Section 5.4 communication tests still pass unaffected. Confirmed the 5 frozen metric definitions (stability/convergence/drift/recovery-gap/fidelity) are genuinely documented in docs/JOURNAL_SIMULATION_RESULTS.md Section 3.9 and ehsfp/research_metrics.py -- no formulas changed. ALL 5 SECTION 5 VALIDITY BLOCKERS NOW CLOSED. Caveat: this closes the infrastructure gap only -- history.csv only populates for NEW runs launched after this fix; Section 8's figure-series bank remains empty until such runs are actually executed.",
)
# ============================================================================
# 11. Pre-flight smoke gate (campaign brief Section 6 item 1) — 2026-07-24
# ============================================================================
_smoke_runs = [
    ("classification", "cifar10", "federated", "alexnet", "configs/classification/federated/cifar_fedavg_alexnet.yaml"),
    ("classification", "cifar10", "hierfl", "alexnet", "configs/classification/hierfl/cifar_hierfl_alexnet.yaml"),
    ("classification", "cifar10", "hsfl", "alexnet", "configs/classification/hsfl/cifar_our_alexnet_5_10.yaml"),
    ("classification", "cifar10", "splitfl", "alexnet", "configs/classification/splitfl/cifar_splitfed_alexnet.yaml"),
    ("classification", "cifar10", "hetero-sfl", "alexnet", "configs/classification/hetero-sfl/cifar_heteroSFL_alexnet.yaml"),
    ("classification", "cifar10", "h-sfp", "alexnet", "configs/classification/h-sfp/cifar_our_alexnet_5_10.yaml"),
    ("classification", "cifar100", "federated", "resnet50", "configs/classification/federated/cifar_fedavg_resnet50.yaml"),
    ("classification", "cifar100", "hierfl", "resnet50", "configs/classification/hierfl/cifar_hierfl_resnet50.yaml"),
    ("classification", "cifar100", "hsfl", "resnet50", "configs/classification/hsfl/cifar_our_resnet50_5_10.yaml"),
    ("classification", "cifar100", "splitfl", "resnet50", "configs/classification/splitfl/cifar_splitfed_resnet50.yaml"),
    ("classification", "cifar100", "hetero-sfl", "resnet18", "configs/classification/hetero-sfl/cifar_heteroSFL_resnet18.yaml"),
    ("classification", "cifar100", "h-sfp", "resnet50", "configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml"),
    ("classification", "ham10000", "federated", "resnet50", "configs/classification/federated/ham10000_fedavg_resnet50.yaml"),
    ("classification", "ham10000", "hierfl", "resnet50", "configs/classification/hierfl/ham10000_hierfl_resnet50.yaml"),
    ("classification", "ham10000", "hsfl", "resnet50", "configs/classification/hsfl/ham10000_our_resnet50_5_10.yaml"),
    ("classification", "ham10000", "splitfl", "resnet50", "configs/classification/splitfl/ham10000_splitfed_resnet50.yaml"),
    ("classification", "ham10000", "hetero-sfl", "resnet50", "configs/classification/hetero-sfl/ham10000_heteroSFL_resnet50.yaml"),
    ("classification", "ham10000", "h-sfp", "resnet50", "configs/classification/h-sfp/ham10000_our_resnet50_5_10.yaml"),
    ("segmentation", "isic2018", "federated", "resnet50", "configs/segmentation/federated/isic_fedavg_resnet50.yaml"),
    ("segmentation", "isic2018", "hierfl", "resnet50", "configs/segmentation/hierfl/isic_hierfl_resnet50.yaml"),
    ("segmentation", "isic2018", "hsfl", "resnet50", "configs/segmentation/hsfl/isic_our_resnet50_5_10.yaml"),
    ("segmentation", "isic2018", "splitfl", "resnet50", "configs/segmentation/splitfl/isic_splitfed_resnet50.yaml"),
    ("segmentation", "isic2018", "hetero-sfl", "resnet50", "configs/segmentation/hetero-sfl/isic_heteroSFL_resnet50.yaml"),
    ("segmentation", "isic2018", "h-sfp", "resnet50", "configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml"),
]
for task, dataset, method, backbone, cfg in _smoke_runs:
    eid = f"smoke_{task}_{dataset}_{method}"
    add(
        experiment_id=eid,
        priority="P1_reuse", protocol_id="preflight_smoke_2epoch",
        dataset=dataset, task=task, backbone=backbone, method=method, variant="baseline",
        condition="clean_iid", rounds=2, seed="MISSING (config default, not overridden)",
        config_path=cfg, code_commit="2026-07-24, current code (post all Section 5 fixes)",
        status="COMPLETED_UNVALIDATED",
        raw_result_path=f"docs/optimization_loop/logs/preflight_smoke/{eid}.log",
        metric_validation="NOT RECORDED -- this is a 2-round crash-freedom smoke test only, per campaign brief Section 6 item 1 ('run two-round smoke tests for every dataset/method family... before expensive final runs'). Any accuracy/IoU value in the log is NOT a reportable result -- too few rounds, not the frozen protocol, seed not controlled.",
        communication_validation="MISSING (not the focus of this smoke pass)",
        notes=f"PASS -- exit 0, no crash, no NaN. Part of the 24/24-clean pre-flight gate run 2026-07-24 (scripts/journal_experiments/run_preflight_smoke.sh). {'CIFAR-10: first-ever execution of this (dataset, method) combination in this repository -- previously zero results of any kind.' if dataset == 'cifar10' else 'Confirms config/dataset/method combination still loads and trains cleanly on current code.'}",
    )

add(
    experiment_id="EXCLUDED_imagenet1k",
    priority="OUT_OF_SCOPE", protocol_id="N/A",
    dataset="imagenet1k", status="EXCLUDED_BY_USER_DECISION",
    notes="data/ImageNet is an empty directory (symlink to /media/jackson/HSF-P/ImageNet target has no files). User explicitly excluded ImageNet-1K from campaign scope 2026-07-23 -- no further recovery action pending, not tracked as an open blocker. All ImageNet-1K rows/cells omitted from the required matrix going forward; the conference reference value (27.9 +- 0.3%) is retained for context only, not as a target to reproduce.",
)
add(
    experiment_id="BLOCKER_ham10000_hsfp_seed_bug",
    priority="P1_reuse", protocol_id="proxy_10round_dateseed",
    dataset="ham10000", method="h-sfp", status="REPAIRED_AWAITING_RERUN",
    notes="RESOLVED 2026-07-23 (was P0_repair_needed): root cause was training collapse (fp16 overflow -> NaN -> argmax always class 0), not a seed-plumbing bug. Fixed 2026-07-21 (shared fp16 fix) and independently GPU-verified 2026-07-23 (2 seeds x 3 epochs: finite losses, full class-spread predictions, distinct per-seed accuracy, differing partition/client-sampling/cloud-init hashes). The original 3-seed 10-round campaign row remains INVALID (not retroactively promoted) and needs a full rerun with the fix before a new valid aggregate exists -- that rerun is the only remaining action, not further repair.",
)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows to {OUT}")
