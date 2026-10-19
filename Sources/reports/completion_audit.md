# E-HSFP journal campaign — completion audit

Audit date: 2026-10-02. Scope: `E_HSFP_Claude_Code_Complete_Pending_Runs.md`.
Machine: 1× RTX 3080 Ti (12 GB), 16 cores, 30 GB RAM; root disk 468 GB at 99% (8 GB free);
data disk `/media/jackson/Data` (1.8 TB, 1.2 TB free, fstab `nofail`), which also holds every
dataset. No scheduler or cluster: work runs in tmux lanes on this host.
Code: `9e785b9` plus the working-tree changes listed in §4. Live per-run state is in
`manifests/run_status.csv`; per-cell coverage is in `artifacts/coverage_report.csv`.

## 1. Initial inventory (read-only, before any launch)

| Item | Finding |
|---|---|
| Active experiment processes | **none**. The GPU was idle and the only tmux session was the user's own. |
| Launcher | `journal_run.sh` → `results/journal_v1/` (manifest, markers, logs) and `outputs/journal_v1/` (artifacts) |
| Completed runs | **140 PASS** (manifest + `.done` markers agree): `s1_parity` 4, `s2_main` 136 |
| Completed GPU time | 499.3 h of training wall time, 2026-08-08 → 2026-08-29 |
| Interrupted | `main_isic2018_ehsfp_s1`: log stops at epoch **50/60** on 2026-08-29 23:34, with no traceback, no `.failed` marker and no manifest row |
| Never started | `main_isic2018_ehsfp_s2..s4`, and every stage after `s2_main` (s2b–s13) |
| Failed (`.failed`) | 0 |
| Prior campaigns | `results/journal/` (seed-0, pre-fix code) and `results/camera_ready/` (ECCV protocol, 30 rounds). Neither is reusable for journal cells (see `journal_run.sh` header); both are kept untouched. |

Status of the 140 completed runs, after validation (§3):
- **129 completed-valid.**
- **11 completed-suspect** — kept, flagged, never silently dropped:
  - CIFAR-10 HeteroSFL s0, s2, s3: final accuracy 10.0–14.7%, at chance.
  - ISIC HeteroSFL s0–s4: IoU 26.02–26.05% on every seed, i.e. a constant predictor.
  - ISIC FedAvg s0 and s4: best round = 1, so nothing was learned after initialisation.
  - HAM10000 H-SFP s1: final-round accuracy ≤ 1.5× uniform chance.

## 2. Why results were pending (root causes)

1. **Host went down mid-run, and nothing restarted the queue.** `journalctl --list-boots`
   shows the next boot at 2026-08-31 07:33; the boot covering Aug 29 is no longer in the
   journal. The ISIC E-HSFP s1 log ends mid-round with no error. The campaign was a foreground
   `journal_run.sh` call with no durable session, no auto-resume and no run locks, so every
   later stage stayed unlaunched for 34 days.
2. **The launcher could not run any more.** `check_env` requires `MIN_FREE_GB=15` on the repo
   disk, which had 8 GB free. Any relaunch would have exited immediately.
3. **No exporter for the journal_v1 campaign.** `collect_results.py`, `build_matrix.py` and
   `assemble_md.py` read the July `results/journal/` campaign. None of the 140 journal_v1 runs
   had been aggregated into any table (an export-only gap).
4. **Stress axes were not implemented, only labelled.** The planned stages s5–s8 would have
   produced mislabelled or byte-identical arms:
   - `max_prototype_age` is a memory-eviction bound. It never delays a packet, so it is not
     Table IV's staleness τ (Def. 3).
   - Cold start only changes *simulated* latency. Accuracy is identical by construction.
   - Partial edge execution did not exist.
   - Client and edge timeouts shared one probability.
   - Memory had a single global 500-record cap, not Table IX's per-class capacity M ∈ {0,1,5,10,20}.
   - Prototype drift (Def. 6), reliability-weight logging (Fig. 8) and tier memory (Fig. 11)
     were not recorded anywhere.
5. **HAM10000 E-HSFP was ~17× slower than H-SFP** (≈190,000 s vs 11,300 s per seed). Per-round
   time climbs from 835 s to ~3,400 s as episodic memory fills to its cap. The cause: the PRC
   loss rebuilt `memory.to_proto_dist_dicts()` and `aggregate_proto_dicts()` from scratch on
   **every minibatch**. For HAM's 256×56×56 prototypes that meant re-stacking ~1.6 GB of CPU
   tensors about 4,350 times per round. Those inputs never change within a phase.
6. **Scientific blocker (not a scheduling issue):** the parity gate failed. Journal-protocol
   H-SFP gives CIFAR-100 10.8%, CIFAR-10 27.9%, HAM10000 33.6% (final round) vs the conference
   55.10 / 67.44 / 80.86. `s1_parity` itself says not to start `s2_main` until this is
   explained; s2 was run anyway. Nothing here fixes it. It is reported, not tuned away
   (§6 of the brief).

## 3. Validity findings in completed runs (fixed at export time, no retraining)

- **Test-set model selection.** `classification/H-SFP/data/get_data.py:146` returns
  `test_dataset` as the validation set. H-SFP's logged "Best Validation / Test Acc" is therefore
  a max-over-rounds pick **on the test split**. Meanwhile FedAvg/HierFL log a final-round test
  accuracy (FedAvg mislabels it "Avg Train Acc"). The logged numbers mixed two rules.
  **Fix:** every classification cell now uses the **final-round (60) test metric**. Sources are
  `last_round_predictions.npz` (accuracy and macro-F1 recomputed with sklearn) or the per-round
  test-accuracy history. The test-selected best round is exported only as a secondary,
  explicitly labelled table. Effect is large for HAM10000 H-SFP: 33.6 ± 12.3 final vs ~60
  best-round.
- **Macro-F1** cannot be recovered for FedAvg/FedProx/FedNova/HierFL: they never saved
  predictions or F1, and FedAvg's printed "F1" is accuracy. Those cells are blank, not imputed.
- **Segmentation selection rules differ by runner** (best-round on val=test vs final-round
  test). Table II records each row's rule; nothing is harmonised by guesswork.
- **Overwritten artifacts.** Segmentation baselines write one fixed JSON name per method, so
  seeds overwrote each other. Per-seed values are parsed from the per-run logs instead, which
  are intact.
- **HierFL test-set leakage during training.** `management: true` in both HierFL configs runs
  `manage_models_top_down()` every round. It evaluates the cloud and edge candidate weights **on
  the test split** and propagates the better one (`segmentation/HierFL/hierarchy.py:206-236`,
  `classification/HierFL/hierarchy.py:~227`). HierFL rows therefore benefit from test-set
  feedback and are not clean comparisons (ISIC 71.5% IoU vs FedAvg's collapse). The baseline
  was not altered mid-campaign; it is flagged in Tables I/II and in this audit.
- **ISIC FedAvg pathology.** All 5 seeds peak at round 1–2 (≈26–28% IoU) and decay towards 0%
  IoU by round 60. Clients, loss, model and aggregator are byte-identical to HierFL's; the only
  material difference is HierFL's test-set model management above. Not root-caused in this
  session; the runs are kept as completed-suspect.
- **ISIC HeteroSFL** is constant across seeds (26.02–26.05% IoU), a constant predictor.
  **CIFAR-10 HeteroSFL** collapses to chance in 3 of 5 seeds (10.0–14.7%). Both are kept and
  flagged as completed-suspect, and are not root-caused yet.
- **Fixed RNG seeds.** `PrototypeDropout` and `ServerlessMetricsTracker` use a fixed RNG seed
  (42), so the drop and timeout pattern is the same for all five seeds. This was left unchanged
  to stay comparable with completed runs. Seed variance therefore understates the variance of
  the stress pattern.

## 4. Fixes applied (all verified)

| Fix | Files | Verification |
|---|---|---|
| Loop-invariant PRC inputs cached once per phase (first batch, same autocast context) | `classification/H-SFP/hierarchy.py`, `segmentation/H-SFP/hierarchy.py` | Old vs new code, CIFAR-10 `full_e_hsfp`, seed 0, 2 rounds, deterministic cuDNN: `train_loss` identical to every digit, all 19 `metrics.json` keys equal, `history.csv` identical except wall time. 13% faster even in this early-memory case. |
| Episodic stress protocol: `staleness_tau` (Def. 3 delay queue: packets delivered at s+d, d~U{0..τ}; memory receives them with true age), `memory_per_class` (eq. 7 per-class FIFO; M=0 disables), `partial_edge_probability/fraction`, `client_/edge_timeout_probability`, `cold_start_defers_packet` | `ehsfp/episodic_stress.py` (new), `ehsfp/memory.py`, `ehsfp/serverless_sim.py`, `ehsfp/aggregation.py`, `classification/H-SFP/hierarchy.py` | Each mechanism has its own seeded generator and is an exact no-op at its neutral value (unit-tested, including "no RNG draw"). The default path, re-run after wiring, is again bit-identical (losses 2.3029213874570784 / 2.3027642773043726, acc 0.1000 / 0.1355). 11 new tests; full suite 73/73. A GPU stress smoke found and fixed one bug (record eviction via `list.remove` compared tensors), now regression-tested. |
| Oracle diagnostics: Def. 6 drift (W2² of the synthesized vs complete-fresh diagonal Gaussians, edge and cloud), reliability weights with their four factors, per-tier CUDA peaks | same | `return_stats` path leaves sampled features bit-identical (unit-tested). Diagnostics never feed training. |
| Segmentation H-SFP honours `output_dir` (was hard-coded) | `segmentation/H-SFP/runner.py` | Lane A writes to the data disk. |
| Launcher: `OUT_ROOT` override with the disk check on that filesystem; atomic per-run locks (stale-PID takeover); `EMIT_QUEUE=1`; earlier attempt logs preserved, never overwritten | `journal_run.sh` | `EMIT_QUEUE` reproduced exactly the 4 missing s2 runs. The Aug 29 log was preserved as `main_isic2018_ehsfp_s1.attempt-20260829T233409.log`. |
| Durable execution: `queue_lane.sh` (drains a priority queue, re-read after every run), `start_lanes.sh` (idempotent tmux lanes), cron `@reboot` restart guarded on the data-disk mount | `scripts/journal_experiments/` and the user crontab (one appended line) | Lanes A and B launched and verified writing to the data disk. |
| Manuscript-mapped completion stages `c_*`, one per table axis; cells identical to completed runs are reused, not re-run | `journal_run.sh` | 286 new runs and 4 resumed s2 runs. |
| Exporter | `scripts/journal_experiments/collect_journal_v1.py` | Reproduces logged values. Writes the ledger, coverage, tables (CSV + LaTeX) and figures. |
| Test runner | `pip install pytest` into the `AAAI-2026` env | The suite had never been runnable in this env (s0 called pytest silently). |

Disk: all 140 runs' artifacts (4.4 GB) were copied to `/media/jackson/Data/ehsfp_campaign/`
with an md5 match on all 462 files. New runs write there. The originals under
`outputs/journal_v1/` and `segmentation/H-SFP/Figure/data/runs/` were left in place (deleting
them was not authorised); they can be removed manually to free 4.4 GB.

## 5. Resume semantics

No mid-run checkpoint exists. `checkpoint.pt` holds best-model weights only, with no optimizer,
scheduler, RNG, memory or queue state. An exact resume of `main_isic2018_ehsfp_s1` was
therefore impossible, and a warm start would break the frozen protocol. It was restarted from
seed and initialisation as a fresh attempt; the interrupted attempt's log is preserved.

## 6. Genuine blockers (not runnable as specified)

| Cell | Reason | What would unblock it |
|---|---|---|
| Table I ImageNet-1K column | `data/ImageNet → /media/jackson/Data/HSF-P/ImageNet` is empty (4 KB) | ImageNet-1K train/val on disk (licence-gated download) |
| Table I FedProto / FedGen / FedDF rows | no implementation in the repo | faithful implementations of the three baselines |
| Table VIII "+ residual generator", Table XI residual rows | `self.residual_generator = None` in both hierarchies; manuscript eq. 14 does not specify G's training objective; `dropout_consistency_loss` (eq. 13) is never invoked | author decision on G's objective and on the L_drop placement |
| Table XI low-rank covariance | not in the pipeline; needs a new packet format and rank r | author decision on r (communication changes) |
| Table IV "stability bound term", Fig. 4 | Theorem 1 constants (L_z, C_proto, C_τ, …) are unspecified | constants; until then only empirical drift is plotted |
| Parity with the conference H-SFP numbers | H-SFP far below the published reference under this repo's protocol | scientific diagnosis (see `docs/optimization_loop/`) |
| ISIC non-IID | no Dirichlet partitioner for segmentation | new code; not required by Table VI (classification only) |
