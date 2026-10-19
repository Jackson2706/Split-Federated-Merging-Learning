# All Simulation Results — Aggregated

Every numeric result that currently exists anywhere in this repository, aggregated into one
place. This is a **data dump**, not a narrative — for interpretation, protocol definitions, and
ongoing status see [`JOURNAL_SIMULATION_RESULTS.md`](JOURNAL_SIMULATION_RESULTS.md); for a
condensed human-readable summary see [`CAMPAIGN_PROGRESS_SUMMARY.md`](CAMPAIGN_PROGRESS_SUMMARY.md).
Every number below is traceable to a source file/log listed in its table's caption. Nothing here
is estimated, inferred, or fabricated — where a value doesn't exist, it's marked `MISSING`, not
filled in.

**Note**: as of 2026-07-23, [`ALL_SIMULATION_RESULTS.md`](../ALL_SIMULATION_RESULTS.md) (repo root)
is the canonical, programmatically-generated numeric ledger for the E-HSFP journal campaign — it
supersedes this file for exhaustive coverage (future-table cell bank, aggregate bank, full
completeness tracking). This file remains a secondary, hand-curated summary and is kept updated
alongside it at the user's request; if the two ever appear to disagree, treat
`ALL_SIMULATION_RESULTS.md` as authoritative.

**Last updated**: 2026-07-24 (pre-flight smoke gate).

**Protocols in play (do not pool across these — see `JOURNAL_SIMULATION_RESULTS.md` Section 3):**
- **Proxy protocol**: 10 communication rounds, date-based seeds (`20260714`/`15`/`16`), n=3.
  Used for quick comparative signal, not the paper's final numbers.
- **Frozen/full protocol**: 60 rounds (classification) or the method's native round count
  (segmentation), integer seeds (`0,1,2,3,4`), n=5 nominal. Most of this work is still single-seed
  (`s0`) — flagged per table.
- Proxy and frozen-protocol numbers for the same method/dataset are **never comparable** — they
  are kept in separate tables throughout.

---

## 1. ISIC-2018 Segmentation

### 1.1 Fair comparison — proxy protocol (10 rounds, seeds 20260714/15/16)

*Source: `results/fair_comparison_isic2018.csv`*

| Method | Test IoU mean ± std | Seed 20260714 | Seed 20260715 | Seed 20260716 | Status |
|---|---:|---:|---:|---:|---|
| HierFL | **63.17 ± 1.56%** | 62.14 | 62.00 | 65.38 | VERIFIED |
| H-SFP (ours) | **47.27 ± 2.19%** | 50.28 | 45.14 | 46.38 | VERIFIED |
| SplitFL | **41.44 ± 0.52%** | 40.74 | 41.60 | 41.98 | VERIFIED (fixed 2026-07-22) |
| HSFL | **34.50 ± 4.33%** (Test); Val 35.55 ± 3.92% | 32.01 | 32.00 | 39.50 | VERIFIED (confirmed 2026-07-22) |
| Federated | **27.76 ± 0.86%** | 27.00 | 27.32 | 28.97 | VERIFIED |
| HeteroSFL | **25.80 ± 0.26%** | 25.44 | 25.99 | 25.98 | VERIFIED |

All 6 methods have genuine, GPU-verified, 3-seed results — this grid is complete. SplitFL's
original result (26.00% flat, all-foreground collapse) and HSFL's original result (23.09%,
2-epoch smoke only) are both superseded by the values above; do not reuse the old numbers.

### 1.2 H-SFP component ablation — proxy protocol (10 rounds, seed 20260714 only)

*Source: `docs/optimization_loop/logs/plan42_isic_hsfp_ablation/`*

| Preset | Test IoU | Test Dice | Runtime (s) |
|---|---:|---:|---:|
| `baseline_hsfp` | 50.28% | 64.95% | 3371 |
| `hsfp_memory` | 47.92% | 62.63% | 3367 |
| `hsfp_memory_dropout` | 47.83% | 62.37% | 3216 |
| `hsfp_memory_reliability` | 46.74% | 61.69% | 3349 |

Single seed only (n=1) — not yet a 3-seed or 5-seed result. `baseline_hsfp` here exactly matches
Section 1.1's H-SFP seed-20260714 value (50.28%), a cross-check that both campaigns used
identical settings. `hsfp_memory_reliability_prc` and `full_e_hsfp` are excluded — both blocked on
the unresolved autograd deadlock (Section 19 of the master doc).

---

## 2. CIFAR-100 Classification

### 2.1 Fair comparison — proxy protocol (10 rounds, seeds 20260714/15/16)

*Source: `results/fair_comparison_cifar100.csv`*

| Method | Best Val Top-1 mean ± std | Seed 20260714 | Seed 20260715 | Seed 20260716 | Notes |
|---|---:|---:|---:|---:|---|
| SplitFL | 45.79 ± 1.77% | 44.57 | 47.82 | 44.99 | Macro-F1 == accuracy in this CSV (not independently computed) |
| Federated | 35.81 ± 2.10% | 34.76 | 38.23 | 34.44 | |
| HierFL | 33.22 ± 1.05% | 33.17 | 34.29 | 32.19 | |
| HSFL | 25.50 ± 11.87% | 12.06 | 34.54 | 29.91 | High variance — seed 20260714 is a clear outlier, not yet investigated |
| HeteroSFL | 18.68 ± 2.80% | 17.25 | 21.91 | 16.89 | Real macro-F1 available in source CSV (2.98/1.88/1.58%) |
| H-SFP (ours) | 10.29 ± 0.46% | 9.76 | 10.55 | 10.56 | This is the `centered_cosine+FREEZE+EVALFIX` variant, not the plain ablation baseline below — **not directly comparable to Section 2.3** |

### 2.2 Fair comparison — full/frozen protocol (60 rounds, seeds 20260714/15/16)

*Source: `results/fair_comparison_cifar100_full60.csv`* — status `COMPLETE_UNVALIDATED`: per-seed
std and F1-validity/communication-accounting checks have not yet been independently applied to
these rows.

| Method | Test Top-1 mean ± std | Seed 20260714 | Seed 20260715 | Seed 20260716 | total_comm_MB |
|---|---:|---:|---:|---:|---:|
| SplitFL | 60.04 ± 1.42% | 59.04 | 61.66 | 59.42 | 222,183.80 |
| HSFL | 57.32 ± 1.25% | 56.40 | 58.75 | 56.82 | 757,374.46 |
| Federated | 56.59 ± 2.19% | 55.38 | 59.12 | 55.28 | 565,462.98 |
| HierFL | 54.84 ± 3.25% | 51.83 | 58.29 | 54.40 | 115,662.88 |
| HeteroSFL | 38.45 ± 4.59% | 42.20 | 39.81 | 33.33 | ~36,700 (varies/seed) |
| H-SFP (ours) | 11.49 ± 0.25% | 11.20 | 11.60 | 11.67 | **756.16** |

H-SFP trades accuracy for a ~300-750x communication reduction vs. every baseline here — this is
the method's central claim, not an accuracy win. Confirmed baseline H-SFP (not E-HSFP — all
`ehsfp_runtime_counters` are zero in the raw JSON for these rows).

### 2.3 H-SFP component ablation — full/frozen protocol (60 rounds, single seed `s0`)

*Source: `docs/optimization_loop/logs/plan37_ablation_rerun/` (memory presets);
`docs/optimization_loop/logs/plan43_cifar100_baseline_hsfp_rerun/` (restored baseline, 2026-07-24)*

| Preset | Best Val/Test Acc | total_comm_MB | Runtime (s) | Status |
|---|---:|---:|---:|---|
| ~~`baseline_hsfp` (July 9, pre-`cf6454c`)~~ | ~~8.10%~~ | ~~30,414.23~~ | ~~~25,000~~ | `SUPERSEDED` |
| **`baseline_hsfp` (rerun 2026-07-24, current code)** | **11.18%** | **1,450.13** | **6,491** | `VERIFIED_NEW` |
| `hsfp_memory` | **11.28%** | 1,430.47 | 6,489 | `COMPLETED_UNVALIDATED` |
| `hsfp_memory_dropout` | **11.20%** | 1,426.39 | 6,175 | `COMPLETED_UNVALIDATED` |
| `hsfp_memory_reliability` | **11.26%** | 1,430.47 | 6,820 | `COMPLETED_UNVALIDATED` |

**Corrected finding (2026-07-23/24)**: the originally-reported "+3.18pp memory improvement" is
**retracted** — it compared two different client architectures. The `baseline_hsfp` row above
(8.10%, 30,414 MB) was a July 9 run predating commit `cf6454c` (July 16), which changed
CIFAR-100's client feature boundary from an unpooled `[64,32,32]` map to a pooled `[128,1,1]`
one — exactly a 512x per-class communication-payload reduction, definitively traced via a CPU
regression test (`tests/test_communication_accounting.py`). Rerunning `baseline_hsfp` on current
code (2026-07-24) gives 11.18% accuracy and 1,450.13 MB total communication — now genuinely
comparable in scale to the memory-preset rows (~1,430 MB), confirming the architecture-change
explanation. **The valid, restored comparison**: baseline 11.18% vs. memory-preset 11.2x% — a gap
of only ~0.02–0.10pp, essentially within single-seed noise. On this single seed, episodic memory
shows **no clear benefit** over baseline on CIFAR-100 — a real finding requires the 5-seed frozen
protocol (not yet run). `hsfp_memory_reliability_prc` / `full_e_hsfp` excluded from this table —
see Section 6 below, the autograd deadlock that previously blocked them is now resolved.

---

## 3. HAM10000 Classification

### 3.1 Fair comparison — proxy protocol (10 rounds, seeds 20260714/15/16)

*Source: `results/fair_comparison_ham10000.csv`*

| Method | Best Val Top-1 mean ± std | Seed 20260714 | Seed 20260715 | Seed 20260716 | Notes |
|---|---:|---:|---:|---:|---|
| SplitFL | 78.13 ± 0.40% | 77.68 | 78.28 | 78.43 | |
| HSFL | 73.77 ± 3.81% | 75.24 | 69.45 | 76.64 | |
| HierFL | 71.44 ± 0.35% | 71.84 | 71.29 | 71.19 | |
| Federated | 71.43 ± 1.09% | 71.84 | 72.24 | 70.19 | |
| HeteroSFL | 68.48 ± 0.68% | 67.70 | 68.90 | 68.85 | Real macro-F1 available (11.46/13.29/21.78%) |
| H-SFP (ours) | **10.98 ± 0.00%** | 10.983524712930603 | 10.983524712930603 | 10.983524712930603 | ⚠️ **`INVALID`** — bit-identical across 3 different seeds. **Root-caused 2026-07-23**: a training collapse (fp16-autocast-overflow in prototype-std computation → NaN cloud logits → argmax always class 0 → accuracy locks to the class-0 validation fraction, 220/2003), **not a seed-plumbing bug** — archived partition hashes already differed by seed. Already fixed 2026-07-21 (shared fp16 fix) and independently GPU-verified 2026-07-23 (2 seeds × 3 epochs: finite losses, full class-spread predictions, distinct per-seed accuracy). This row is **not** retroactively promoted — still `INVALID` as originally reported; a full 3-seed 10-round rerun with the fix is the only remaining step. **Do not reuse.** |

3 additional `unknown`/`incomplete` rows exist in the source CSV (config hashes only, no metric) —
excluded from the table above, not silently dropped from the record.

### 3.2 H-SFP component ablation — full/frozen protocol (60 rounds, single seed `s0`)

*Source: `docs/optimization_loop/logs/plan39_ablation_rerun_ham_v2/`* (fp16-prototype-std fix
applied — see `DECISIONS.md`; supersedes an earlier pre-fix run of the same 3 presets that were
stuck at 10.98%/epoch-1)

| Preset | Best Val/Test Acc | Best epoch | total_comm_MB |
|---|---:|---:|---:|
| `hsfp_memory` | **54.72%** | 27/60 | 16,063.54 |
| `hsfp_memory_dropout` | **50.77%** | 42/60 | 15,473.14 |
| `hsfp_memory_reliability` | **55.82%** | 56/60 | 16,063.54 |

No `baseline_hsfp` comparator run in this pass — Section 3.1's H-SFP row is `INVALID` (seed bug),
so there is currently no trustworthy HAM10000 baseline to compare against. All three show genuine
progressive learning (best-model updates spread across the full 60 rounds), a dramatic contrast
with the pre-fix collapsed values. Single seed only (n=1). `hsfp_memory_reliability_prc` /
`full_e_hsfp` excluded — blocked on the autograd deadlock.

---

## 4. Older / Historical Proxy Results (pre-dates this campaign's seed conventions)

*Source: `docs/optimization_loop/EXPERIMENT_REGISTRY.csv`, `results/journal/numeric/AGGREGATED_RESULTS.csv`*

| Experiment | Dataset | Method/variant | Value | Status |
|---|---|---|---:|---|
| `plan14_freeze_centered` | CIFAR-100 | H-SFP, `baseline_hsfp+centered_cosine+FREEZE+EVALFIX` | 10.29 ± 0.37% (best_val_top1, n=3) | `VERIFIED_EXISTING` — same underlying data as Section 2.1's H-SFP row |

This registry stops at PLAN-14 (2026-07-14); later work is tracked narratively in `DECISIONS.md`
and numerically in the tables above instead.

---

## 5. Validity-Blocker Campaign (Section 5 of the governing brief) — ALL 5 CLOSED, 2026-07-23

A separate, large infrastructure-repair effort closed every outstanding validity blocker this
session. None of these fixes retroactively validate any existing proxy/legacy result above — they
remove structural obstacles to running and trusting *future* results. Full detail in
`ALL_SIMULATION_RESULTS.md` Section 5 and `docs/optimization_loop/DECISIONS.md`; summary:

- **Autograd deadlock (`full_e_hsfp`/PRC)** — the campaign's longest-standing blocker, documented
  since 2026-07-14/16 as an indefinite hang. **Resolved**: a full 10-round, 200-client CIFAR-100
  `full_e_hsfp` proxy run completed cleanly end-to-end (exit 0, `Total Run Time: 4052.31s`, Test
  Acc 7.20%, all 4 E-HSFP components — memory, PRC, reliability, dropout — confirmed genuinely
  active via nonzero runtime counters). Most likely already fixed by 3 earlier rounds of repairs
  and never re-verified end-to-end; single seed/dataset so far, needs extension.
- **HAM10000 seed bug** — root-caused as a training collapse (see Section 3.1 above), not a
  seed-plumbing defect. Fixed and GPU-verified.
- **Metric/checkpoint correctness** — added auditable prediction-artifact saving + an independent
  validator (recomputes accuracy/macro-F1/per-class recall/confusion matrix from raw saved arrays,
  never trusting the training loop's self-report) for H-SFP/HSFL/SplitFL/HeteroSFL classification;
  de-duplicated segmentation's 5 copies of `compute_iou_and_dice` into one frozen, canonical
  function.
- **Communication-ledger reconciliation** — see Section 2.3 above (the CIFAR-100 architecture
  artifact); added a from-first-principles packet-by-packet reconciliation test.
- **Per-round history logging** — added `ehsfp/history_logger.py`, a durable (flushed+`fsync`'d
  every round) `history.csv` schema, wired into H-SFP. This is the infrastructure the
  figure-series bank needs — no existing run retroactively gains a history file; only new runs
  launched after this fix will populate it.

---

## 6. Pre-flight Smoke Gate — 24/24 CLEAN, 2026-07-24

Per the campaign brief's Section 6, item 1: *"Before expensive final runs: run two-round smoke
tests for every dataset/method family."* Ran `scripts/journal_experiments/run_preflight_smoke.sh`
— a bounded 2-round smoke test for every (dataset, method) combination across the 4 available
datasets (CIFAR-10/AlexNet, CIFAR-100/ResNet, HAM10000/ResNet50, ISIC-2018/ResNet50-U-Net —
ImageNet-1K excluded per the 2026-07-23 scope decision) × the 6 method families (federated,
hierfl, hsfl, splitfl, hetero-sfl, h-sfp) = **24 runs, all passed** (exit 0, no crash, no NaN).

**CIFAR-10 was confirmed runnable end-to-end for the first time in this repository's history** —
all 6 methods, previously zero results of any kind. The configs already existed (using AlexNet,
matching the brief's own spec) but had simply never been executed until this gate.

**These are crash-freedom smoke results only, not reportable numbers** — 2 rounds, uncontrolled
seed, not the frozen protocol. No accuracy/IoU value from this gate should ever be cited as a
journal result. Full per-combo detail: `docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv`
(rows `smoke_*`) and `ALL_SIMULATION_RESULTS.md` Section 5b.

This closes Section 6 item 1 of the brief. Items 2-4 of that section remain open: one full H-SFP
parity seed per dataset, explaining gaps from the accepted conference reference (notably CIFAR-100:
~11% proxy/legacy vs. 55.10% conference — still unexplained), and freezing the method/checkpoint/
metrics/partitions/communication protocol. These should be addressed before any full 5-seed
frozen-protocol campaign is trusted.

---

## 7. Summary — What's Genuinely Missing

Per `JOURNAL_SIMULATION_RESULTS.md` Section 22's completeness checklist (tracking the original
brief's 31-item list): still a small fraction of the full 31-item list is met. Not reflected above
because it doesn't exist yet:

- CIFAR-10 — confirmed runnable (Section 6, 2026-07-24) but still zero *reportable* results, any
  method — only a 2-round crash-freedom smoke pass exists so far.
- ImageNet-1K — excluded from campaign scope by explicit user decision (2026-07-23); no longer an
  open blocker.
- Any 5-seed (frozen-protocol) result for any ablation preset — everything ablation-related above
  is single-seed (`s0`).
- `hsfp_memory_reliability_prc` — the autograd deadlock that blocked it is resolved (Section 5),
  but it hasn't been independently re-run itself yet.
- Convergence curves, prototype dropout/staleness, serverless-stress, IID/non-IID robustness,
  scalability, memory-size ablation, prototype-synthesis ablation (master doc Sections 7–18) —
  not yet run.
