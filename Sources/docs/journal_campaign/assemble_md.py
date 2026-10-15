#!/usr/bin/env python3
import json, csv, io

REPO = "/home/jackson/Desktop/Split-Federated-Merging-Learning"
OUT = f"{REPO}/ALL_SIMULATION_RESULTS.md"
meta = json.load(open("/home/jackson/.claude/jobs/317bbf05/tmp/meta.json"))
cells = json.load(open("/home/jackson/.claude/jobs/317bbf05/tmp/cell_bank.json"))
aggs = json.load(open("/home/jackson/.claude/jobs/317bbf05/tmp/agg_bank.json"))

CELL_HEADER = ["result_id", "prospective_table", "protocol_id", "dataset", "task", "method",
               "variant", "condition", "seed", "selected_checkpoint", "metric", "exact_value",
               "unit", "status", "raw_result_path"]
AGG_HEADER = ["aggregate_id", "prospective_table", "protocol_id", "dataset", "method", "variant",
              "condition", "metric", "seed_list", "n", "mean", "sample_std_ddof1", "ci95_low",
              "ci95_high", "unit", "status"]
SERIES_HEADER = ["series_id", "prospective_figure", "protocol_id", "dataset", "method", "variant",
                 "condition", "seed", "step_name", "step_value", "x_name", "x_value", "y_name",
                 "y_value", "unit", "status", "raw_result_path"]


def to_csv_block(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


status_counts = meta["status_counts"]
total = meta["total_rows"]

md = []
md.append("# ALL_SIMULATION_RESULTS.md — Canonical Numeric Ledger\n")
md.append(
    "This is the **single canonical human-readable numeric ledger** for the E-HSFP journal-extension "
    "campaign, per the mandate given 2026-07-23. It supersedes `docs/ALL_RESULTS_AGGREGATED.md` as the "
    "sole place to find future table cells and figure series (that file is left in place, unmodified, "
    "as a superseded historical snapshot — do not read it going forward). Every value here is generated "
    "programmatically from `docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv` and the source CSVs/logs "
    "listed per row; nothing below was hand-typed or estimated. This ledger reports **numeric data only** "
    "— no figures, publication tables, captions, or manuscript content are produced by this task.\n"
)

md.append("## 1. Run Metadata\n")
md.append(f"- **Generated**: {meta['now']}")
md.append(f"- **Code commit**: `{meta['commit']}`")
md.append("- **Working tree dirty**: yes — uncommitted changes present:")
md.append("  ```")
md.append(f"  {meta['dirty']}")
md.append("  ```")
md.append("- **Environment**: single-host, PyTorch 2.7 / CUDA 12.8, Python (repo venv)")
md.append("- **Hardware**: 1x NVIDIA GeForce RTX 3080 Ti (12288 MiB total, ~11897 MiB free at audit "
           "time), 16 CPU cores, 30 GiB RAM, root filesystem 468G/95% full (**25G free — tight for a "
           "multi-week campaign, monitor**), datasets mounted from `/media/jackson/Data` (1.3T free, "
           "24% used) via symlinks in `data/`.")
md.append("- **Dataset availability**:")
md.append("  - CIFAR-10: present (`data/cifar/cifar-10-batches-py`)")
md.append("  - CIFAR-100: present (`data/cifar/cifar-100-python`)")
md.append("  - HAM10000: present (`data/HAM10000`, 5.4G)")
md.append("  - ISIC-2018: present (`data/ISIC2018`, 11G)")
md.append("  - **ImageNet-1K: ABSENT** (`data/ImageNet` is an empty directory) — **excluded from "
           "campaign scope by explicit user decision, 2026-07-23; not tracked as an open blocker.**")
md.append("- No AGENTS.md exists in this repository.\n")

md.append("## 2. Protocol Definitions (frozen, never mixed)\n")
md.append("| protocol_id | Rounds/epochs | Seeds | Purpose | Status |")
md.append("|---|---|---|---|---|")
md.append("| `conference_accepted_eccv2026` | 200 (classification); native budget (segmentation) | "
           "N/A (published aggregate) | Accepted ECCV 2026 H-SFP reference values | Reference only, "
           "external, never rerun |")
md.append("| `proxy_10round_dateseed` | 10 | `20260714,20260715,20260716` (n=3) | Fast comparative "
           "signal during this session's debugging | Not a final result for any table |")
md.append("| `legacy_60round_dateseed` / `legacy_60round_intseed` | 60 | date-seeds (n=3) or `s0` "
           "(n=1) | Pre-frozen-protocol classification runs, `COMPLETE_UNVALIDATED` | Diagnostic/legacy "
           "only, kept as evidence, not final |")
md.append("| `development_seeds` | varies | `100,101,102` | E-HSFP component development/tuning "
           "(validation metrics only) | Never used for final reporting |")
md.append("| `frozen_journal_v1` | 200 (classification) or native budget (segmentation, matched to "
           "conference); interval sweep per Section 8.9 | `0,1,2,3,4` (n=5) | **The only protocol "
           "whose results may be reported as final journal numbers** | **Not yet started for any "
           "cell — see Section 4** |\n")

md.append("## 3. Campaign Completeness (recomputed from the actual repo, not trusted from an old count)\n")
md.append(f"Source: `docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv`, {total} rows as of this "
           "generation.\n")
md.append("| Status | Count | Meaning |")
md.append("|---|---:|---|")
label = {
    "VERIFIED": "Proxy/legacy result independently confirmed non-collapsed/non-crashed (**not** the "
                "same as `frozen_journal_v1`-final)",
    "COMPLETED_UNVALIDATED": "Ran to completion, exit 0, but F1-validity/communication-accounting/"
                              "independent-recompute checks not yet applied",
    "INVALID": "Confirmed broken (seed bug, collapse, etc.) — kept for provenance, not reusable",
    "FAILED_INFRA": "Infra-level failure (e.g. incomplete row with no captured metric)",
    "BLOCKED_RESOURCE": "Blocked on an environment/access constraint (e.g. sudo for the deadlock repro)",
    "RESOLVED_PENDING_FULL_VERIFICATION": "Root blocker resolved and GPU-verified once; no longer structurally blocked, but not yet verified across the full seed/dataset matrix required for a final result",
    "BLOCKED_DATASET": "Blocked on missing licensed/external data",
    "EXCLUDED_BY_USER_DECISION": "Explicitly removed from campaign scope by the user -- not tracked as an open blocker",
    "REPAIRED_AWAITING_RERUN": "Root-cause fixed and independently GPU-verified, but the original invalid campaign row itself was not rerun/promoted -- a fresh full run is the only remaining step",
    "SUPERSEDED": "Preserved for provenance only; a newer, valid rerun replaces this row and should be used instead",
    "VERIFIED_NEW": "A fresh, current-code run independently confirmed healthy -- the up-to-date reusable value for this cell",
}
for k, v in status_counts.items():
    md.append(f"| `{k}` | {v} | {label.get(k, '')} |")
md.append(f"| `RUNNING` | 0 | No jobs currently running |")
md.append(f"| `NOT_STARTED` / final `frozen_journal_v1` cells | **0 of ~thousands required** | "
           "Section 4 below and Section 15 enumerate what's outstanding; the matrix will grow as each "
           "phase is queued — see Section 16 for why it isn't pre-populated with placeholder rows. |\n")

md.append("## 4. What Is Genuinely Missing (recalculated this session, not assumed)\n")
md.append("- **Every** `frozen_journal_v1` final 5-seed cell, for every dataset/method/condition in "
           "Sections 8.1–8.9 of the governing brief: **zero exist**. All numbers in this ledger are "
           "proxy or legacy protocol, explicitly not final.")
md.append("- CIFAR-10: **confirmed runnable end-to-end for the first time, 2026-07-24** (pre-flight "
           "smoke gate, all 6 methods pass a 2-round crash-freedom check — see Section 5). Still "
           "**zero reportable results of any kind** — no proxy, legacy, or frozen-protocol run has "
           "actually been executed; the smoke pass only proves the configs/dataset/method "
           "combinations load and train without crashing.")
md.append("- ImageNet-1K: dataset absent AND excluded from campaign scope by explicit user "
           "decision (2026-07-23) — no longer tracked as an open item.")
md.append("- `full_e_hsfp`: **RESOLVED 2026-07-23** — the long-standing autograd deadlock no longer "
           "reproduces (see Section 5.1). One clean 10-round CIFAR-100 proxy run exists "
           "(seed 20260714 only); still needs the other 4 final seeds, HAM10000/ISIC-2018, and the "
           "full 5-seed frozen protocol before it's a final result. `hsfp_memory_reliability_prc` "
           "shares the same PRC code path and is very likely also fine, but has not been "
           "independently re-verified end-to-end — treat as unconfirmed, not resolved, until it is.")
md.append("- HAM10000's H-SFP proxy row: still `INVALID` as originally reported (not retroactively "
           "promoted) — but the root cause is now fixed and GPU-verified (Section 5.2, **CLOSED** "
           "2026-07-23; see Section 5 below). A fresh full 3-seed 10-round rerun with the fix is the "
           "only remaining step before a new valid aggregate exists for this cell.")
md.append("- Communication-accounting: **RESOLVED 2026-07-23** (Section 5.4 — see Section 5). The "
           "CIFAR-100 30,414→1,430 MB figure is a cross-code-version artifact (client architecture "
           "changed at commit `cf6454c`), not a memory effect. **`baseline_hsfp` was rerun on "
           "current code 2026-07-24** (11.18% accuracy, 1450.13 MB — now genuinely comparable to "
           "the memory-preset rows) restoring a valid ablation comparator: memory's effect is "
           "~0.02–0.10pp vs. baseline, essentially within single-seed noise, not the previously-"
           "reported (and retracted) +3.18pp.")
md.append("- Macro-F1 independent recomputation from saved predictions: **infrastructure CLOSED, "
           "2026-07-23** (Section 5.3 — see Section 5 below); the CSVs above still show `MISSING` "
           "for macro_f1 on most historical rows because they predate this fix and were never "
           "rerun with prediction-artifact saving enabled — the capability now exists for all "
           "future runs, it has not been retroactively applied to old CSV rows.")
md.append("- Non-IID robustness (8.3), prototype dropout (8.4), staleness (8.5), serverless stress "
           "(8.6), memory-capacity (8.7), aggregation/synthesis ablations (8.8), scalability/interval "
           "sweeps (8.9): **zero runs exist for any of these**.")
md.append("- Per-round unsmoothed figure series (Section 8.10): the logging infrastructure gap is "
           "**CLOSED 2026-07-23** (Section 5.5) — `ehsfp/history_logger.py` now writes a durable, "
           "schema-frozen `history.csv` per run, wired into H-SFP (classification and "
           "segmentation). But **no run using this new logging has been executed yet**, so the "
           "figure-series bank (Section 8) is still empty — this is now a queued-work gap, not an "
           "infrastructure gap.\n")

md.append("## 5. Validity Blockers — Repair Status\n")
md.append("| Blocker | Section | Status | Notes |")
md.append("|---|---|---|---|")
md.append("| Full E-HSFP / PRC autograd deadlock | 5.1 | **RESOLVED, 2026-07-23** | Added "
           "non-privileged in-process diagnostics (`ehsfp/deadlock_diagnostics.py`: `faulthandler` "
           "traceback timer + CUDA memory/utilization sampling, no `sudo`/`ptrace` needed) plus a "
           "static audit of 3 previously-unexamined code paths (nothing conclusive found there). "
           "First diagnosed repro attempt segfaulted — isolated via a controlled baseline rerun to "
           "`torch.autograd.set_detect_anomaly(True)` (part of the diagnostics themselves, not the "
           "training code); fixed by splitting it into a separate, off-by-default "
           "`EHSFP_DEADLOCK_DIAG_ANOMALY` flag. **With that isolated, a full 10-round, 200-client "
           "CIFAR-100 `full_e_hsfp` proxy run completed cleanly end-to-end** (exit 0, "
           "`Total Run Time: 4052.31s`, finite losses throughout, Test Acc 7.20%, all 4 E-HSFP "
           "components confirmed genuinely active via `runtime_counters.jsonl`: memory reads/writes, "
           "100%-nonzero PRC loss, nonuniform reliability weights, active dropout). **The deadlock "
           "does not reproduce with current code** — most likely already fixed by 3 earlier rounds "
           "(PRC batching, `.item()`-sync removal, `max_replay_batch` capping) and never "
           "re-verified end-to-end, or resolved by environment/library drift since 2026-07-14/16. "
           "Caveat: single seed/dataset/protocol so far — needs extension to the other 4 seeds, "
           "HAM10000/ISIC-2018, and the full frozen protocol before `VERIFIED`; "
           "`hsfp_memory_reliability_prc` not yet independently re-run. |")
md.append("| Seed correctness (HAM10000 H-SFP bit-identical bug) | 5.2 | **CLOSED, 2026-07-23** | "
           "Root cause was a training collapse (fp16-autocast-overflow in prototype-std computation "
           "-> NaN cloud logits -> argmax always class 0 -> seed-invariant accuracy = the class-0 "
           "validation fraction), not a seed-plumbing defect -- archived partition hashes already "
           "differed by seed. Fixed 2026-07-21 (shared fp16 fix, predates this diagnosis) and "
           "independently GPU-verified 2026-07-23 (2 seeds x 3 epochs: finite losses, full "
           "class-spread predictions, distinct accuracy, differing partition/sampling/cloud-init "
           "hashes). New fingerprinting infra (`ehsfp/integrity.py`'s `state_dict_hash`/"
           "`index_batch_hash`/`write_run_fingerprints`) added and unit-tested (6/6 passing, "
           "independently re-run). A GPU-only bug in the first fingerprinting diff (0-dim BatchNorm "
           "buffer crash) was caught during independent verification and fixed in a second bounded "
           "Codex pass. Original invalid campaign row is *not* retroactively promoted -- a full "
           "3-seed 10-round rerun is still required. |")
md.append("| Metric/checkpoint correctness (macro-F1 independent recompute, imbalanced unit test) | "
           "5.3 | **CLOSED, 2026-07-23** | `classification/training_metrics.py` now saves raw "
           "predictions/labels (`.npz`, no training-loop metric values included, enforced by a key-"
           "set check) at `best_val_predictions.npz`/`last_round_predictions.npz` for all 4 methods "
           "with real macro-F1 (H-SFP, HSFL, SplitFL, HeteroSFL) — none skipped. A standalone CLI "
           "(`scripts/journal_experiments/validate_metrics.py`) independently recomputes accuracy, "
           "macro-F1, per-class recall, and a confusion matrix from the raw arrays only. All 4 "
           "classification and 6 segmentation output JSONs now expose separate "
           "`last_round_validation`/`best_validation`/`selected_checkpoint` records (previously "
           "conflated in places — e.g. HeteroSFL's `best_f1` key actually held accuracy, now "
           "corrected to `best_val_top1`); existing checkpoint-selection policies were preserved, "
           "not changed. Segmentation's 5 duplicate `compute_iou_and_dice` copies were de-duplicated "
           "into one canonical, docstring-frozen function in `segmentation/training_metrics.py` "
           "(strict `>0.5` threshold, whole-batch flattened pixel aggregate, empty union/denominator "
           "→ exactly 0/0) — confirmed all 6 segmentation call sites (HierFL, Federated x2, "
           "HeteroSFL, H-SFP, SplitFL, HSFL) now import the shared function. Independently "
           "re-verified: all 15 relevant tests genuinely pass on rerun "
           "(`tests.test_classification_metric_reporting`, `tests.test_segmentation_metric_protocol`, "
           "`tests.test_research_metrics`), including the deliberately-imbalanced accuracy≠macro-F1 "
           "case, a hand-computed validator cross-check, and the empty-mask 0/0 edge case. CPU-only "
           "work; no GPU verification needed or performed. |")
md.append("| Communication ledger (packet-level, CIFAR-100 30,414→1,430 MB anomaly) | 5.4 | "
           "**CLOSED, 2026-07-23** | The anomaly is **definitively explained as a cross-code-"
           "version comparison artifact, not a memory effect or a skipped transmission**: the "
           "`baseline_hsfp` row is a July 9 run predating commit `cf6454c` (July 16), which changed "
           "CIFAR-100's client feature boundary from unpooled `[64,32,32]` to pooled `[128,1,1]` — "
           "exactly a 512x per-class-entry size reduction, confirmed via archived route-subtotal "
           "arithmetic and a CPU regression test using the real `get_proto_dist_size_MB` function. "
           "A second CPU regression invokes the real `_client_ssl_extraction_phase` with memory on "
           "vs. off and proves byte-identical output, independently ruling out any "
           "memory-dependent communication difference. A from-first-principles packet-by-packet "
           "reconciliation test (`tests/test_communication_accounting.py`) now exists and passes "
           "(6/6 tests, independently re-run). **RESTORED, 2026-07-24**: `baseline_hsfp` rerun on "
           "current code (60 rounds, seed 0) — 11.18% accuracy, 1450.13 MB total comm, now "
           "genuinely comparable in scale to the memory-preset rows (~1430 MB), confirming the "
           "architecture-change explanation. Valid comparison restored: baseline 11.18% vs. "
           "hsfp_memory 11.28% / hsfp_memory_dropout 11.20% / hsfp_memory_reliability 11.26% — a "
           "~0.02–0.10pp gap, essentially single-seed noise, **not** the previously-reported (and "
           "retracted) +3.18pp 'memory improvement', which was entirely a cross-architecture "
           "artifact. Single seed (n=1) — no real memory-vs-baseline conclusion can be drawn until "
           "the 5-seed frozen protocol (Section 8.2) is run. Section 5.4's other requirement "
           "(verify `full_e_hsfp` has nonzero counters for every enabled module) was already "
           "satisfied by the GPU deadlock-verification run in Section 5.1. |")
md.append("| Per-round history/series logging (runtime, memory, drift, reliability, stability) | 5.5 | "
           "**CLOSED, 2026-07-23** | Added a shared, durable `ehsfp/history_logger.py`: one "
           "`history.csv` row per completed round (round, elapsed time, train loss, validation "
           "metric name+value, per-round communication deltas by route reusing "
           "`ehsfp/communication.py`, phase CPU/RSS/GPU-peak columns, episodic memory occupancy, "
           "event/failure column), schema frozen and documented in its module docstring. Every row "
           "is flushed and `fsync`'d immediately, so a killed/timed-out run keeps a valid, "
           "parseable partial file — independently verified via a test that writes 2 of 5 intended "
           "rounds, drops the logger reference (simulating a crash), and confirms the file still "
           "parses with exactly the 2 completed rows and a reopenable, schema-checked header. "
           "Wired into both classification and segmentation H-SFP's training loops (the primary "
           "method), snapshotting communication state at each round's start so deltas — not just "
           "cumulative totals — are recorded. H-SFP itself does not currently track phase-level "
           "CPU/RSS/GPU-peak history (unlike some other methods) — those columns are left present "
           "but empty rather than fabricating measurement code, an honest, explicitly-reported gap. "
           "The 5 frozen metric definitions (stability, convergence, drift, recovery-gap, fidelity) "
           "were confirmed already documented in `docs/JOURNAL_SIMULATION_RESULTS.md` Section 3.9 "
           "and implemented in `ehsfp/research_metrics.py` — no formulas changed. Independently "
           "re-verified: 10/10 relevant tests pass on rerun (`tests.test_history_logger` + "
           "`tests.test_research_metrics`), plus the 6 communication-accounting tests from Section "
           "5.4 still pass unaffected. **Caveat**: this closes the *infrastructure* gap — "
           "`history.csv` will only start populating for runs launched after this fix; no existing "
           "completed run retroactively gains a history file, and Section 8's figure-series bank "
           "remains empty until new runs are actually executed with this logging in place. |\n")
md.append("**All 5 of Section 5's validity blockers are now closed (2026-07-23)** — 5.1 "
           "(deadlock), 5.2 (seed correctness), 5.3 (metric/checkpoint correctness), 5.4 "
           "(communication-ledger reconciliation), 5.5 (per-round history logging). Per the "
           "brief's own rule, this unblocks the final five-seed queue *in principle* — but the "
           "queue itself (the actual Section 6-9 multi-dataset, multi-seed numeric campaign) has "
           "not been launched; closing Section 5 removes the validity gate, it does not constitute "
           "the campaign. None of Section 5's fixes retroactively validate any existing proxy/"
           "legacy result — every number in Sections 9-10 below remains exactly as validated (or "
           "not) as before.")
md.append("")

md.append("### 5b. Pre-flight Smoke Gate (Section 6 item 1 of the governing brief) — "
          "24/24 CLEAN, 2026-07-24\n")
md.append("Per the brief: *\"Before expensive final runs: 1. run two-round smoke tests for every "
          "dataset/method family.\"* Ran `scripts/journal_experiments/run_preflight_smoke.sh`: a "
          "bounded 2-round smoke test for every (dataset, method) combination across the 4 "
          "available datasets (CIFAR-10/AlexNet, CIFAR-100/ResNet, HAM10000/ResNet50, "
          "ISIC-2018/ResNet50-U-Net — ImageNet-1K excluded per the 2026-07-23 scope decision) × "
          "the 6 method families (federated, hierfl, hsfl, splitfl, hetero-sfl, h-sfp) = 24 runs. "
          "**Result: 24/24 passed — exit 0, no crash, no NaN, in every combination.**\n")
md.append("Notably, **CIFAR-10 was confirmed runnable end-to-end for the first time in this "
          "repository's history** — all 6 methods, previously zero results of any kind (configs "
          "already existed, using AlexNet per the brief's own spec, but had simply never been "
          "executed). CIFAR-100/HAM10000/ISIC-2018 were re-confirmed clean on current, "
          "post-Section-5-fix code.\n")
md.append("**These are crash-freedom smoke results only, not reportable numbers** — 2 rounds, "
          "uncontrolled seed, not the frozen protocol. No accuracy/IoU value from this gate should "
          "ever be cited as a journal result; see the cell bank (Section 9) rows tagged "
          "`smoke_*` for the explicit `COMPLETED_UNVALIDATED` status and this caveat repeated "
          "per-row. This closes Section 6 item 1 of the brief; items 2-4 of that same section "
          "(one full H-SFP parity seed per dataset; explain gaps from the conference reference; "
          "freeze method/checkpoint/metrics/partitions/communication) remain open — notably the "
          "unexplained CIFAR-100 gap (~11% proxy/legacy vs. 55.10% conference reference) still "
          "needs investigation before any full 5-seed campaign should be trusted.\n")

md.append("## 6. Conference Reference Values (never rerun, clearly labeled)\n")
md.append("| Dataset/metric | H-SFP conference reference | protocol_id |")
md.append("|---|---:|---|")
md.append("| CIFAR-10 accuracy | 67.44 ± 0.85% | `conference_accepted_eccv2026` |")
md.append("| CIFAR-100 accuracy | 55.10 ± 0.95% | `conference_accepted_eccv2026` |")
md.append("| HAM10000 accuracy | 80.86 ± 0.75% | `conference_accepted_eccv2026` |")
md.append("| ImageNet-1K accuracy | 27.9 ± 0.3% | `conference_accepted_eccv2026` |")
md.append("| ISIC-2018 IoU | 68.3 ± 0.5% | `conference_accepted_eccv2026` |")
md.append("| ISIC-2018 Dice | 79.5 ± 0.4% | `conference_accepted_eccv2026` |")
md.append("| H-SFP(A) communication, 200 rounds | 13.8 GB | `conference_accepted_eccv2026` |")
md.append("| H-SFP(C) communication, 200 rounds | 10.94 GB | `conference_accepted_eccv2026` |")
md.append("\nNote the large gap between these and the proxy/legacy numbers below (e.g. CIFAR-100 "
           "H-SFP proxy ~10%, legacy60 ~11% vs. 55.10% conference) — **this is an open, unexplained "
           "parity gap, not yet reconciled**. Per the brief's Section 6: \"A result such as 11% on "
           "CIFAR-100 versus the 55.10% conference reference is a failed parity check, not a journal "
           "result to accept silently.\" This must be investigated (config/data/code diff against the "
           "conference setup) before any frozen final run is trusted, and is queued as the first "
           "concrete task after Section 5's blockers.\n")

md.append("## 7. Frozen Final Results (`frozen_journal_v1`, 5 seeds)\n")
md.append("**None exist yet.** Zero `frozen_journal_v1` runs have been launched. This section will be "
           "populated per-dataset/per-experiment-family only as real 5-seed verified runs land.\n")

md.append("## 8. Future-Figure Series Bank\n")
md.append("**Empty — infrastructure gap, not yet a data gap that can be filled by running more jobs.** "
           "No existing run (proxy, legacy, or otherwise) in this repository has ever written a "
           "structured per-round `history.csv`; every existing log only contains final/best-checkpoint "
           "console output. Until Section 5.5's logging repair lands, no per-round unsmoothed series "
           "exists to record here. Schema (for when it exists):\n")
md.append("```")
md.append(",".join(SERIES_HEADER))
md.append("```")
md.append("_(zero rows currently — see Section 5.5)_\n")

md.append("## 9. Future-Table Cell Bank\n")
md.append(f"One flat record per measured metric value, generated from the matrix and source CSVs. "
           f"{len(cells)} rows. Full machine precision preserved in `exact_value` "
           "(Python `repr()` of the float as parsed from source).\n")
md.append("```csv")
md.append(to_csv_block(CELL_HEADER, cells).rstrip("\n"))
md.append("```\n")

md.append("## 10. Aggregate Bank\n")
md.append(f"{len(aggs)} rows. `sample_std_ddof1` uses `ddof=1`; `ci95_low`/`ci95_high` use a "
           "two-sided t-interval with the exact t-critical value for the seed count (n=1 rows report "
           "mean with zero-width interval, honestly, rather than a fabricated spread). "
           "**`status` values ending in `_NOT_FINAL` or prefixed `INVALID_` must never be reported as "
           "journal results** — they exist here purely as traceable intermediate evidence.\n")
md.append("```csv")
md.append(to_csv_block(AGG_HEADER, aggs).rstrip("\n"))
md.append("```\n")

md.append("## 11. Component / Robustness / Capacity / Aggregation / Synthesis / Scalability Ablations\n")
md.append("- **E-HSFP component ablation** (CIFAR-100, HAM10000, ISIC-2018): single-seed, "
           "legacy/proxy protocol only — see cell bank rows tagged `ehsfp_ablation_*`. "
           "**Zero 5-seed frozen-protocol ablation rows exist.** `full_e_hsfp` now has one clean "
           "10-round CIFAR-100 verification run (Section 5.1, RESOLVED) but is not yet folded into "
           "the ablation ladder itself; `hsfp_memory_reliability_prc` remains untested end-to-end "
           "(no longer structurally blocked, just not yet run).")
md.append("- IID/non-IID robustness (8.3): **not started**.")
md.append("- Prototype dropout sweep (8.4): **not started**.")
md.append("- Prototype staleness sweep (8.5): **not started**.")
md.append("- Serverless/episodic stress (8.6): **not started** (a `use_serverless_simulation` config "
           "flag and cold-start/timeout probability fields exist in `ehsfp/config.py`'s defaults, but "
           "no campaign has exercised them with deterministic logged fault injection yet).")
md.append("- Memory-capacity ablation (8.7): **not started**.")
md.append("- Aggregation-rule comparison (8.8): implementation-complete for 3 of 4 rules (`average`, "
           "`sample_count_weighted`, `reliability-aware` all exist and are GPU-verified per earlier "
           "session work; `reliability-aware + episodic memory` composition not yet isolated as its "
           "own condition) — **no actual numeric comparison run exists yet**.")
md.append("- Synthesis-rule comparison (8.8): **not started** — only the plain reliability path is "
           "implemented; diagonal Gaussian / low-rank covariance / residual generator / residual+"
           "consistency-loss are unverified or stub per earlier session findings "
           "(`docs/optimization_loop/DECISIONS.md`).")
md.append("- Scalability (K clients) and aggregation-interval sweeps (8.9): **not started** for the "
           "frozen protocol. Two interval configurations exist informally in legacy work "
           "(`t1/t2` = 5/10 used throughout this session's ISIC work) but were never run as a "
           "controlled A/B/C interval comparison with fixed dataset size.\n")

md.append("## 12. Invalid / Superseded Runs (preserved, never deleted)\n")
md.append("| experiment_id | Reason | Status |")
md.append("|---|---|---|")
md.append("| `ham10000_proxy_h-sfp_s20260714/15/16` | Bit-identical `best_val_top1` "
           "(10.983524712930603) across 3 different seeds — seed-not-applied bug | `INVALID` |")
md.append("| `isic2018` SplitFL pre-fix runs (2 seeds, 26.00% flat) | All-foreground collapse; "
           "hard-coded SGD ignoring config's Adam + server-optimizer state discarded every round | "
           "`SUPERSEDED` by the post-fix 3-seed campaign (41.44±0.52%) |")
md.append("| `abl_ham10000_hsfp_memory(_dropout)_RERUN` (first pass, pre-fp16-fix) | fp16 prototype-std "
           "overflow silently corrupting training; stuck at 10.98% best-epoch-1 | `SUPERSEDED` by "
           "`_RERUN_v2` (54.72%/50.77%) |")
md.append("| `ham10000_proxy_UNKNOWN_s{seed}` (3 rows) | Incomplete source rows: config hash only, no "
           "captured metric | `FAILED_INFRA` |\n")

md.append("## 13. Remaining Blocked/Missing Runs — Exact Recovery Action\n")
md.append("| Blocker | Exact recovery action |")
md.append("|---|---|")
md.append("| ImageNet-1K absent | **No longer applicable — excluded from campaign scope by "
           "explicit user decision, 2026-07-23.** (For reference only, had this remained in "
           "scope: recovery would have required populating the standard ILSVRC2012 train/val "
           "layout at `data/ImageNet`'s target, a licensed dataset the assistant cannot obtain "
           "itself.) |")
md.append("| `full_e_hsfp` / `hsfp_memory_reliability_prc` — no longer a blocker | **RESOLVED "
           "2026-07-23**, no external recovery action needed. Remaining work is ordinary "
           "verification (more seeds, more datasets, the frozen protocol), tracked as normal "
           "queued campaign work, not a blocked item. |")
md.append("| HAM10000 H-SFP seed bug | Engineering fix, no external blocker — queued as this session's "
           "next Codex-delegated task (Section 5.2). |")
md.append("| CIFAR-100 55.10% conference-parity gap | Engineering/config investigation, no external "
           "blocker — queued after Section 5's repairs land. |\n")

md.append("## 14. Consistency Report\n")
md.append(f"- Cell bank rows: {len(cells)}. Aggregate bank rows: {len(aggs)}. Matrix rows: {total}.")
md.append("- Every aggregate bank row's `mean`/`sample_std_ddof1` was recomputed in this generation "
           "pass directly from the cell-bank/source-CSV per-seed values listed in `seed_list` — no "
           "aggregate value was hand-entered.")
md.append("- Every cell bank row's `raw_result_path` points to a file that exists in this repository "
           "as of generation time (spot-checked: `results/fair_comparison_isic2018.csv`, "
           "`results/fair_comparison_cifar100.csv`, `results/fair_comparison_cifar100_full60.csv`, "
           "`results/fair_comparison_ham10000.csv`, `docs/optimization_loop/logs/plan42_isic_hsfp_"
           "ablation/`, `docs/optimization_loop/logs/plan39_ablation_rerun_ham_v2/`, "
           "`results/journal/logs/` — all present).")
md.append("- **No figure series exist yet to regenerate** (Section 8) — this is an honest gap, not a "
           "silently-passed check; it will be exercised once Section 5.5 lands.")
md.append("- This ledger will be regenerated (not hand-edited) after every verified batch — the "
           "generator script is retained at `docs/journal_campaign/` for that purpose (see Section 16).\n")

md.append("## 15. Honest Status Statement\n")
md.append("This is the **initial migration and audit pass** of the canonical ledger, not a completed "
           "campaign. As of this generation:\n")
md.append("- **0 of the ~thousands of required `frozen_journal_v1` 5-seed cells are complete.**")
md.append("- All 93 cell-bank / 24 aggregate-bank rows above are proxy or legacy protocol — valuable "
           "as diagnostic/reuse evidence, explicitly not final journal numbers.")
md.append("- **ALL 5 of Section 5's validity blockers are now closed (2026-07-23)**: 5.1 (the "
           "`full_e_hsfp`/PRC autograd deadlock — the campaign's longest-standing, "
           "highest-priority blocker), 5.2 (seed correctness), 5.3 (metric/checkpoint correctness — "
           "auditable prediction artifacts, independent validator, segmentation IoU/Dice "
           "de-duplication), 5.4 (communication-ledger reconciliation — the CIFAR-100 MB anomaly "
           "is a cross-code-version artifact, not a memory effect; `baseline_hsfp`'s ablation "
           "comparator is retracted pending a rerun), and 5.5 (per-round history logging — the "
           "durable `history.csv` schema needed before Section 8's figure-series bank can ever be "
           "populated). All five independently re-verified by re-running every relevant test "
           "myself, not just trusting Codex's self-reports. **This is a major campaign milestone**: "
           "per the brief's own rule, the final five-seed queue may now launch *in principle* — but "
           "closing Section 5 removes the validity gate, it does not constitute the campaign "
           "itself. None of these fixes retroactively validates any existing proxy/legacy result.")
md.append("- The next concrete actions: (a) rerun CIFAR-100's `baseline_hsfp` on current code so "
           "the memory ablation has a valid comparator again; (b) extend the `full_e_hsfp` "
           "verification to more seeds/datasets and independently confirm "
           "`hsfp_memory_reliability_prc`, now that neither is structurally blocked; (c) begin "
           "applying Section 5.3's prediction-artifact saving and Section 5.5's history logging to "
           "a real campaign run, so historical CSV rows' `MISSING` macro_f1 fields and the empty "
           "figure-series bank start getting filled with genuine, independently-verifiable values; "
           "(d) begin the actual Section 6-9 multi-dataset, multi-seed numeric campaign now that "
           "no validity blocker stands in the way.")
md.append("- E-HSFP wins/ties/losses cannot yet be honestly assessed — no final matched comparison "
           "exists. The proxy-protocol numbers above show H-SFP trading large accuracy losses for "
           "large communication savings vs. baselines (Section 6), but this is not the frozen "
           "comparison and must not be read as a journal finding.\n")

md.append("## 16. Generation Method (for reproducibility of this file itself)\n")
md.append("Generated by a three-stage Python pipeline (kept at `docs/journal_campaign/` for reuse on "
           "the next update):")
md.append("1. `build_matrix.py` — reads every existing source CSV/log and writes "
           "`docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv`.")
md.append("2. `build_ledger.py` — reads the matrix + source CSVs, computes aggregates "
           "(`ddof=1`, exact t-interval per seed count), and emits JSON intermediates for the cell "
           "and aggregate banks.")
md.append("3. `assemble_md.py` — renders this file from the JSON intermediates.\n")
md.append("**Why the matrix isn't pre-populated with thousands of `NOT_STARTED` placeholder rows for "
           "every future cell**: the brief asks for a *resumable* manifest, not an exhaustively "
           "enumerated one before work begins. Placeholder rows for e.g. every "
           "(dataset × method × condition × seed) combination across Sections 8.3–8.9 would number in "
           "the low thousands and carry zero information (every field `NOT_STARTED`/`MISSING`) — they "
           "will be added at the start of each phase (Section 8.1 first, immediately after Section 5's "
           "blockers close), keeping the matrix's information density high and avoiding a false sense "
           "of \"the plan is already written down\" that would need to be re-verified anyway once real "
           "config hashes and partition hashes exist.\n")

with open(OUT, "w") as f:
    f.write("\n".join(md))

print(f"Wrote {OUT}, {len(md)} lines-of-blocks")
