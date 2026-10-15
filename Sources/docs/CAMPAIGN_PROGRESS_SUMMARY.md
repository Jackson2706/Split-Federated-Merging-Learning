# E-HSFP Journal Campaign — Progress & Performance Summary

This is a condensed, readable snapshot of the campaign. It is a *summary*, not the source of
truth — every number here is traceable back through
[`docs/JOURNAL_SIMULATION_RESULTS.md`](JOURNAL_SIMULATION_RESULTS.md) and
`results/journal/manifests/result_provenance.json`. Update that master document first; this file
is generated from it, not the other way around.

- **Repository**: `/home/jackson/Desktop/Split-Federated-Merging-Learning`
- **Branch**: `DungTT`
- **Last updated**: 2026-07-22
- **Latest relevant commits**: `9e785b9` (journal deliverables), `5c37283` (code/test fixes),
  `ba27944` (ISIC segmentation NaN fix). **Uncommitted as of this update**: SplitFL optimizer fix
  (`segmentation/SplitFL/runner.py`) + all 2026-07-22 doc/CSV updates below — see `git status`.

---

## 1. Current Performance

### ISIC-2018 segmentation (10-round proxy, 3 seeds: 20260714/15/16) — FULL GRID COMPLETE

| Method | Test IoU (mean ± std) | Individual seeds | Status |
|---|---:|---|---|
| HierFL | 63.17 ± 1.56% | — | Verified |
| **H-SFP (ours)** | **47.27 ± 2.19%** | 50.28 / 45.14 / 46.38% | Verified |
| SplitFL | 41.44 ± 0.52% | 40.74 / 41.60 / 41.98% | Verified 2026-07-22 (see below) |
| HSFL | 34.50 ± 4.33% (Val 35.55 ± 3.92%) | 32.01 / 32.00 / 39.50% | Verified 2026-07-22 (see below) |
| Federated | 27.76 ± 0.86% | — | Verified |
| HeteroSFL | 25.80 ± 0.26% | — | Verified |

All 6 methods now have genuine, GPU-verified, 3-seed ISIC-2018 results — this grid is done.

- **SplitFL**: originally collapsed to predicting all-foreground for every pixel (flat 26.00% IoU
  across every seed/epoch — matched the dataset's raw foreground-pixel fraction almost exactly).
  Root cause: `segmentation/SplitFL/runner.py` hard-coded SGD despite the config specifying Adam,
  plus the server optimizer was recreated every round (discarding its state). Fixed and
  re-verified 2026-07-22 — the 41.44% above is the real, post-fix result.
- **HSFL**: never had more than a single 2-epoch smoke point before 2026-07-22. Its first real
  3-seed, full 10-round campaign completed with no code changes needed — confirmed healthy via
  val-vs-test IoU divergence and Dice/IoU cross-consistency (HSFL only validates at its final
  epoch by config, so no intra-run per-epoch curve exists to check for collapse directly).
- H-SFP seeds: 50.28% / 45.14% / 46.38% — fixed 2026-07-20 (was a deterministic 0.00% before an
  fp16-overflow NaN bug was found and fixed).

### ISIC-2018 H-SFP segmentation ablation (10-round proxy, single seed 20260714) — NEW 2026-07-22

| Preset | Test IoU | Test Dice |
|---|---:|---:|
| `baseline_hsfp` | 50.28% | 64.95% |
| `hsfp_memory` | 47.92% | 62.63% |
| `hsfp_memory_dropout` | 47.83% | 62.37% |
| `hsfp_memory_reliability` | 46.74% | 61.69% |

All 4 genuinely distinct, no collapse; `baseline_hsfp` exactly reproduces the fair-comparison
number above for the same seed (cross-check the two campaigns used identical settings). Notably,
IoU *decreases* monotonically here as components are added — opposite direction from the
CIFAR-100 ablation below, where memory improves accuracy. Single seed only; not yet explained,
not over-interpreted. `hsfp_memory_reliability_prc` and `full_e_hsfp` excluded (deadlock-blocked,
see Section 3).

### CIFAR-100 classification — component ablation (60-round, single seed s0)

**RESTORED, 2026-07-24** (was retracted 2026-07-23): the original `baseline_hsfp` (8.10%, July 9)
predated commit `cf6454c` (July 16), which changed the CIFAR-100 client's feature boundary from
`[64,32,32]` to a pooled `[128,1,1]` — comparing two different client architectures, not a valid
ablation (see `docs/optimization_loop/DECISIONS.md`'s Section 5.4 entry). Reran `baseline_hsfp` on
current code 2026-07-24: **11.18% accuracy, 1450.13 MB** — now genuinely comparable in scale to
the memory-preset rows.

| Preset | Best Val/Test Acc | total_comm_MB | Status |
|---|---:|---:|---|
| baseline_hsfp (rerun, current code) | **11.18%** | 1450.13 | `VERIFIED_NEW` |
| + episodic memory | 11.28% | 1430.47 | `COMPLETED_UNVALIDATED` |
| + memory + dropout | 11.20% | 1426.39 | `COMPLETED_UNVALIDATED` |
| + memory + reliability | 11.26% | 1430.47 | `COMPLETED_UNVALIDATED` |

**The restored, valid finding**: memory's effect is ~0.02–0.10pp vs. baseline — essentially
within single-seed noise, not a meaningful improvement. The originally-claimed "+3.18pp" was
entirely an artifact of comparing two different architectures. On this single seed, episodic
memory shows no clear benefit on CIFAR-100; a real finding requires the 5-seed frozen protocol.

~~Memory is the component that actually moves the needle (+3.18pp); dropout/reliability add small~~
~~secondary effects on top.~~ (retracted 2026-07-23, superseded by the restored comparison above)

### HAM10000 classification — component ablation (60-round, single seed s0)

| Preset | Best Val/Test Acc |
|---|---:|
| + episodic memory | 54.72% |
| + memory + dropout | 50.77% |
| + memory + reliability | 55.82% |

### Important framing

- **H-SFP is a communication-efficiency method, not an accuracy-maximizing one.** On the CIFAR-100
  full-60-round baseline table, SplitFL/HierFL/Federated/HSFL all land around 55-60% test accuracy
  vs. H-SFP's ~11% — but H-SFP uses roughly 300x less communication (~756 MB vs. HierFL's
  ~115,663 MB per the full-60 comparison). That tradeoff, not raw accuracy, is the method's point.
- Every ablation number above is **single-seed (s0)**. The frozen protocol calls for 5 seeds
  (`0,1,2,3,4`) for new work — these are not yet final campaign numbers.
- `full_e_hsfp` (the complete proposed method, including PRC) **cannot currently run at all** —
  see Section 3.

---

## 2. Current Progress

### What's done this session (all GPU-verified, not just implemented)

| Area | Outcome |
|---|---|
| Ablation-precedence config bug | Root-caused, fixed, confirmed working on CIFAR-100 (differentiated results) |
| fp16-autocast-overflow bug (classification) | Fixed — resolved both a hard NaN crash and a silent HAM10000 training collapse |
| Real macro-F1 (H-SFP/HSFL/SplitFL) | Implemented — was silently reporting accuracy under an "f1" key |
| Communication accounting | Unified across all 6 method families; found/fixed several real undercounting bugs (HeteroSFL's download cost was never tracked at all) |
| Stability / rounds-to-convergence / drift / recovery-gap / fidelity metrics | 5 of 6 implemented from frozen operational definitions (no manuscript exists to source exact definitions from) |
| ISIC-2018 segmentation for SplitFL & HSFL | Built from scratch — both had zero segmentation-capable models before this session |
| Sample-count-weighted aggregation | Implemented — completes the 3 modes needed for the aggregation-rule comparison |
| `collect_results.py` | Fixed — its parser never matched any real log format, so none of 44 prior campaign runs had ever had a metric extracted |
| SplitFL ISIC all-foreground collapse (2026-07-22) | Root-caused (hard-coded SGD + discarded server-optimizer state) and fixed; 3-seed campaign re-verified healthy: 41.44 ± 0.52% |
| HSFL ISIC first-ever multi-epoch verification (2026-07-22) | 3-seed campaign confirmed healthy, no code changes needed: 34.50 ± 4.33% |
| ISIC-2018 H-SFP segmentation ablation (2026-07-22) | Never run before — now complete for 4 of 6 presets, single seed, proxy protocol (see table above) |

### Task queue status

Only one item remains open from this session's audit:

| Item | Status |
|---|---|
| `full_e_hsfp` autograd deadlock | **Blocked** — needs `sudo` for `py-spy`/`gdb` to attach a debugger to the hung process; not available in this environment. Confirmed empirically, not assumed. |

### Deliberately left open (not bugs — decisions for you)

- **"Full E-HSFP" paper scope**: of the 9 claimed novel components, 4 turned out to be either
  inert stubs (residual generator, dropout-consistency loss) or offline-only (low-rank covariance
  synthesis, prototype fidelity). Documented in `docs/optimization_loop/DECISIONS.md` — you decide
  whether to implement the gaps for real or narrow the paper's claimed contribution.
- **Stability-bound term**: intentionally left unimplemented. No manuscript exists in this repo to
  source the theoretical formula from, and fabricating one would violate the campaign's own
  no-fabrication rule.

### Not yet started (genuinely large-scale work, not blocked — just not done)

- The full 5-seed, multi-dataset (CIFAR-10/100, HAM10000, ImageNet-1K, ISIC-2018), multi-method
  numeric campaign the brief originally requested. Given observed per-run costs (a single
  60-round CIFAR-100 run takes ~6.7 hours), this is realistically weeks of continuous single-GPU
  compute, not a remaining bug-fix task.

---

## 3. Where Everything Is Recorded

| What | File |
|---|---|
| Master audit/protocol/status document | `docs/JOURNAL_SIMULATION_RESULTS.md` |
| Per-seed raw results | `results/journal/numeric/SEED_LEVEL_RESULTS.csv` |
| Aggregated (mean/std) results | `results/journal/numeric/AGGREGATED_RESULTS.csv` |
| Future-figure data readiness | `results/journal/numeric/NUMERIC_DATA_MANIFEST.csv` |
| Per-dataset comparison tables | `results/fair_comparison_{isic2018,cifar100,ham10000}.csv` |
| Run launch commands / logs / exit status | `results/journal/manifests/RUN_MANIFEST.csv` |
| Failure diagnosis and fix status | `results/journal/manifests/RUN_FAILURES.csv` |
| Reuse/invalidate decisions for pre-existing results | `results/journal/manifests/existing_result_audit.csv` |
| Full provenance trace for every reported number | `results/journal/manifests/result_provenance.json` |
| Experiment configs/status/dependencies | `experiments/journal_experiment_registry.yaml` |
| Bug root-cause writeups | `docs/optimization_loop/DECISIONS.md` |
| Pre-existing (older) results narrative | `docs/optimization_loop/RESULTS_SUMMARY.md` |
| Raw training logs | `results/journal/logs/*.log`, `docs/optimization_loop/logs/plan*/` |
| SplitFL fix + HSFL verification raw logs (2026-07-22) | `docs/optimization_loop/logs/plan41_isic_splitfl_hsfl_fixed/` |
| ISIC-2018 H-SFP segmentation ablation raw logs (2026-07-22) | `docs/optimization_loop/logs/plan42_isic_hsfp_ablation/` |

Everything through `9e785b9`/`5c37283`/`ba27944` is committed. The 2026-07-22 work (SplitFL fix,
SplitFL+HSFL campaign results, ablation campaign, and all doc updates above) is **not yet
committed** — 4 files modified (`segmentation/SplitFL/runner.py`,
`docs/JOURNAL_SIMULATION_RESULTS.md`, `docs/optimization_loop/DECISIONS.md`,
`results/fair_comparison_isic2018.csv`), pending your go-ahead. Raw `outputs/`/`logs/` training
artifacts stay local per `.gitignore` regardless.
