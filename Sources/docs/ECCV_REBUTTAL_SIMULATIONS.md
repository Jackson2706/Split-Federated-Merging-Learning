# ECCV 2026 Rebuttal / Camera-Ready Simulations — Completed Record

Everything actually executed for the ECCV 2026 H-SFP rebuttal and camera-ready
response, recorded so the journal-extension campaign does not redo it.

**Generated**: 2026-08-02, from `results/camera_ready/` (65 run logs, 65 resume
markers, 7 experiment families). Every number below is read programmatically
from the committed CSV/JSON artifacts — none is hand-transcribed or estimated.

- **Campaign window**: 2026-06-22 → 2026-06-26
- **Executed**: 65 training runs — **63 completed, 2 failed** (CUDA OOM)
- **Cost**: ~47.9 GPU-hours accumulated wall time, 1x RTX 3080 Ti (12 GB)
- **Driver**: `scripts/camera_ready/run_all_camera_ready.sh`
- **Artifacts**: `results/camera_ready/`, markers in `results/camera_ready/.markers/`

---

## 1. READ THIS FIRST — three caveats that decide reusability

**(a) The executed protocol is NOT the protocol the fairness manifest claims.**
`results/camera_ready/fairness/baseline_fairness_summary.md` documents the
*intended* common protocol: 200 global rounds, seeds 0–4, `local_epochs=5`,
`ssl_epochs_client/edge=10`, `syn_epochs_cloud=10`, `t1=5, t2=10`. What the logs
show was actually run is a **reduced protocol**:

| Parameter | Fairness manifest (intended) | Actually executed |
|---|---:|---:|
| Global rounds (`epochs`) | 200 | **30** |
| Seeds | 0, 1, 2, 3, 4 | **0 only** |
| `local_ep` | 5 | **3** |
| `ssl_epochs_client` / `_edge` | 10 / 10 | **3 / 3** |
| `syn_epochs_cloud` | 10 | **3** |
| `t1` / `t2` | 5 / 10 | **3 / 6** |
| `num_users` / `num_edges` | 200 / 5 | 200 / 5 (matches) |
| Dataset | CIFAR-100 | CIFAR-100 (matches) |

Verified from the `[Runner] Overrides:` line at the head of every log, e.g.
`results/camera_ready/logs/hetero_h-sfp_ae1.0_ac1.0_s0.log`. Treat the manifest
as a statement of intent, not a description of these runs.

**(b) The `test_f1` column is not F1 for three of the five methods.** At the time
these ran, `classification/H-SFP`, `HSFL` and `SplitFL` did not compute a real
macro-F1 — the collector parsed their printed `Test Acc` into a column named
`test_f1`. Cross-checked: `hetero_h-sfp_ae1.0_ac1.0_s0.log` ends with
`Test Acc: 5.74%`, and `summary_hetero.csv` records `test_f1_mean = 0.0574` for
that cell. Real macro-F1 landed 2026-07-21, after this campaign. So the tables
below are **top-1 accuracy** for `h-sfp`/`hsfl`/`splitfl` and genuine macro-F1
only for `hierfl`/`federated` — the two are not comparable within a row.

**(c) H-SFP is collapsed in every one of these runs** (3.5–6.5%, vs. 47% for
FedAvg and the 55.10% ECCV reference). This is the same unexplained parity gap
tracked in `ALL_SIMULATION_RESULTS.md` Section 6, visible here two weeks
earlier. These runs predate all five validity fixes of 2026-07-21..24 (fp16
prototype-std NaN overflow, ablation config precedence, unified communication
accounting, prediction artifacts, per-round history logging).

**Consequence: none of the numbers below may be pooled with, or reused as, a
journal `frozen_journal_v1` result.** They are a completed, self-consistent
record of what the rebuttal answered — not a head start on the journal campaign.

---

## 2. Reviewer concern → experiment map (all 7 delivered)

| # | Reviewer concern | Experiment | Runs | Status | Artifacts |
|---|---|---|---:|---|---|
| 1 | Hierarchical heterogeneity | Two-level Dirichlet (αe / αc) | 20 | 20/20 done | `hetero/` |
| 2 | Partial participation | frac × distribution sweep | 45 | **43/45 done, 2 OOM** | `partial/` |
| 3 | Diagonal-Gaussian assumption | Covariance ablation + MMD | 5 modes | done | `covariance/` |
| 4 | Lstat estimation | Empirical Lipschitz sensitivity | 3 ρ, 371 records | done | `lstat/` |
| 5 | Packing/unpacking overhead | CUDA-synced phase profiling | 2 methods | done | `profiling/` |
| 6 | Privacy / feature inversion | Inversion exposure comparison | 5 samples × 3 targets | done | `inversion/` |
| 7 | Baseline fairness | Shared-protocol manifest | n/a (doc) | done | `fairness/` |

Note `results/camera_ready/README.md` still says *"the full (and most smoke) runs
have not been executed yet"*. That line is stale — it predates the 2026-06-24/25
execution and should be ignored.

---

## 3. Experiment 1 — Hierarchical heterogeneity (two-level Dirichlet)

20 runs: 5 methods × (αe, αc) ∈ {1.0, 0.1}², CIFAR-100, seed 0, 30 rounds,
200 clients / 5 edges. The partitioner also fixes the client→edge mapping, which
the hierarchy honors via `_client_to_edge`.

Metric: **top-1 accuracy %** for `h-sfp`/`hsfl`/`splitfl`, **macro-F1 %** for
`hierfl`/`federated` (see caveat b).

| Method | αe=1.0, αc=1.0 | αe=1.0, αc=0.1 | αe=0.1, αc=1.0 | αe=0.1, αc=0.1 |
|---|---:|---:|---:|---:|
| `h-sfp` | 5.74 | 4.24 | 3.68 | 3.49 |
| `hsfl` | 13.55 | 7.57 | 7.29 | 1.96 |
| `splitfl` | 15.66 | 6.42 | 10.11 | 3.45 |
| `hierfl` | 43.40 | 36.76 | 16.02 | 13.27 |
| `federated` | 47.47 | 41.70 | 42.63 | 29.47 |

Source: `hetero/summary_hetero.csv` (n_seeds=1, so every `*_std` is 0.0 — a
single-seed artifact, not a variance measurement).

Also produced: partition diagnostics for all four (αe, αc) settings —
per-edge and per-client class histograms plus summaries, as `.json` + `.png` +
`.pdf`, in `hetero/partition_diagnostics/` (24 files). These are dataset
statistics, not training runs, and remain valid regardless of the code fixes.

---

## 4. Experiment 2 — Partial participation

45 runs: 5 methods × frac ∈ {0.1, 0.5, 1.0} × distribution ∈ {IID, Dir(0.1),
Dir(0.05)}, CIFAR-100, seed 0, 30 rounds, 200 clients. Same metric caveat.

| Method | frac | IID | Dir(0.1) | Dir(0.05) |
|---|---:|---:|---:|---:|
| `h-sfp` | 0.1 | 6.45 | 4.78 | 5.15 |
| `h-sfp` | 0.5 | 5.53 | 5.12 | 3.63 |
| `h-sfp` | 1.0 | 5.79 | 5.28 | 3.66 |
| `hsfl` | 0.1 | 16.42 | 8.61 | 4.80 |
| `hsfl` | 0.5 | 18.15 | 10.72 | 5.99 |
| `hsfl` | 1.0 | 19.90 | 11.44 | 7.19 |
| `splitfl` | 0.1 | 24.47 | 8.73 | 6.31 |
| `splitfl` | 0.5 | 36.69 | 22.32 | 16.96 |
| `splitfl` | 1.0 | 37.05 | 29.61 | 25.22 |
| `hierfl` | 0.1 | 41.94 | 18.11 | 9.27 |
| `hierfl` | 0.5 | 48.84 | 43.96 | 39.11 |
| `hierfl` | 1.0 | **FAILED** | 45.26 | 41.01 |
| `federated` | 0.1 | 48.55 | 43.56 | 39.21 |
| `federated` | 0.5 | 49.52 | 45.62 | 41.67 |
| `federated` | 1.0 | **FAILED** | 45.89 | 42.27 |

Source: `partial/summary_partial.csv`.

### The 2 failed runs

`partial_federated_frac1.0_iid_s0` and `partial_hierfl_frac1.0_iid_s0` both died
with `torch.OutOfMemoryError` inside `Adam.step()` — the 12 GB card ran out with
all 200 clients active (`frac=1.0`) under an IID partition, which gives every
client a full-size shard. Both marked `.failed`; the other seven `frac=1.0`
cells (non-IID, smaller effective shards) completed.

These are the only genuinely incomplete cells in the ECCV campaign. Recovery
needs a memory reduction, not a code fix — see `s12_eccv_recover` in
`journal_run.sh`, which reruns exactly these two with `local_bs=8`.

---

## 5. Experiment 3 — Covariance ablation (diagonal-Gaussian assumption)

Measured at the pooled edge→cloud synthesis boundary with a frozen client+edge
extractor, CIFAR-100, D=512, 100 classes. Communication is analytic per-class
payload; MMD is RBF-MMD(real, synthetic).

| Mode | Transmitted | Asymptotic | Comm (MB) | Accuracy | MMD | Runtime (s) |
|---|---|---|---:|---:|---:|---:|
| `mu_only` | μ | O(D) | 0.205 | 0.0194 | 0.0000 | 0.14 |
| `diag_sigma` (default) | μ, σ | O(D) | 0.410 | 0.0206 | 0.1589 | 1.08 |
| `full_covariance` | μ, Σ | O(D²) | 52.736 | 0.0164 | 0.2375 | 0.49 |
| `low_rank_covariance` | μ, U_r, d | O(8D) | 2.048 | 0.0156 | 0.2388 | 1.08 |
| `mixture_gaussian` | {π_k, μ_k, σ_k} | O(3D) | 1.230 | 0.0156 | 0.2233 | 9.62 |

Source: `covariance/covariance_cifar100.csv` (+ `.json`, `.tex`).

The headline for the rebuttal: `diag_sigma` is the best accuracy in this sweep
at 0.41 MB, while `full_covariance` costs **129× the communication** for *lower*
accuracy — i.e. the diagonal assumption is not the bottleneck. Absolute
accuracies are ~2% because the extractor is deliberately frozen at random init
to isolate the synthesis question; they are not end-to-end model accuracies and
must never be quoted as such.

---

## 6. Experiment 4 — Lstat (empirical Lipschitz sensitivity)

Sensitivity of (μ, σ) to input Gaussian perturbations, CIFAR-100, 371 records
per ρ.

| ρ | Δμ (mean ± std) | Δσ (mean ± std) | Δx (mean) | **Lstat (mean ± std)** |
|---:|---|---|---:|---|
| 0.01 | 2.354 ± 0.400 | 2.506 ± 0.287 | 0.554 | **8.767 ± 1.180** |
| 0.10 | 29.569 ± 11.346 | 29.909 ± 9.087 | 5.539 | **10.737 ± 3.555** |
| 0.20 | 68.431 ± 29.820 | 67.010 ± 23.412 | 11.057 | **12.248 ± 4.600** |

Source: `lstat/lstat_cifar100_agg.csv` (+ `_raw.csv`, `.json`, `.tex`).

Supports **bounded empirical sensitivity only** — explicitly not a proof of
representation preservation. Lstat grows sub-linearly as the perturbation grows
20×, which is the property the rebuttal claimed.

---

## 7. Experiment 5 — Runtime profiling (packing/unpacking overhead)

CUDA-synchronized per-phase timings, H-SFP on CIFAR-100, 30 rounds.

| Phase | Samples | Mean (s/round) |
|---|---:|---:|
| `client_pack` | 30 | 28.10 |
| `edge_process` | 30 | 110.95 |
| `cloud_process` | 30 | 1.23 |
| `edge_aggregate` | 10 | 0.02 |
| `cloud_aggregate` | 5 | 0.01 |
| **`total_round`** | 30 | **140.65** |

Wall clock, same protocol: H-SFP **4228.4 s**, SplitFL **524.5 s**.

Source: `profiling/profile_hsfp_cifar100_summary.json`, `profiling/wallclock.csv`.

The rebuttal point: aggregation is free (0.02 s + 0.01 s ≈ 0.02% of a round) and
packing is 20% — the cost is `edge_process` (SSL training) at 79%, which is
compute, not protocol overhead.

---

## 8. Experiment 6 — Feature inversion / privacy exposure

Optimization-based inversion reconstructing a specific sample from (a) SplitFed
smashed activations, (b) H-SFP μ, (c) H-SFP μ+σ. 5 CIFAR-100 samples × 3 targets
= 15 reconstructions. Higher MSE / lower PSNR = harder to invert = better privacy.

| Target | MSE (mean) | PSNR dB (mean) |
|---|---:|---:|
| `splitfed_smashed` | 1.563 | 16.22 |
| `hsfp_mu` | 2.313 | 10.25 |
| `hsfp_mu_sigma` | 3.127 | 8.73 |

Per-sample values in `inversion/inversion_cifar100.csv` (+ `.json`, `.tex`,
and reconstruction grids `inversion_grid_cifar100.{png,pdf}`). SSIM/LPIPS columns
are empty — `scikit-image`/`lpips` were not installed, so those metrics were
skipped rather than fabricated.

Both H-SFP variants are harder to invert than SplitFed's smashed activations,
and adding σ makes it harder still. **Empirical exposure, not formal DP** — the
README is explicit about this and the paper text should stay equally explicit.

---

## 9. Experiment 7 — Baseline fairness manifest

`fairness/baseline_fairness_summary.md` + `.tex`, generated from
`configs/camera_ready/baseline_fairness.yaml`. A documentation artifact (no
training), recording the common protocol, split points, communication-accounting
formulas, LR search space, and determinism settings applied across
`h-sfp, hsfl, splitfl, hierfl, federated` and the four federated strategies.

Per caveat (a), this manifest describes the *intended* protocol; the executed
runs used the reduced settings.

---

## 10. What this means for the journal campaign

**Reusable as-is (do not rerun):**
- All 7 analysis experiments — covariance, Lstat, profiling, inversion,
  fairness manifest, and the partition diagnostics. These answer ECCV reviewer
  questions, are not part of the journal's numeric matrix, and have no stage in
  `journal_run.sh`. Nothing to skip, because nothing was scheduled.

**Not reusable — must be rerun at the frozen protocol:**
- The 63 completed hetero + partial training runs. Wrong round budget (30 vs
  60/200), one seed instead of five, reduced SSL/local epochs, pre-fix code, a
  mislabeled metric column, and a collapsed H-SFP arm. Any of those alone would
  disqualify them; together they make pooling indefensible.

**Genuinely still missing from the ECCV campaign itself:** the 2 OOM runs
(§4). `journal_run.sh s12_eccv_recover` reruns exactly those two at the ECCV
protocol with a halved batch size, and nothing else.

**Dimension the ECCV set covers that the journal campaign does not:** partial
participation (frac sweep). The journal brief's Sections 8.1–8.9 do not request
it, and the ECCV grid already answers it, so `journal_run.sh` deliberately has
no partial-participation stage. If a journal reviewer asks, the recovery is a
5-seed rerun of §4's grid — roughly 45 runs × 5 seeds at the frozen budget.

**On the alphas:** the ECCV runs used α ∈ {1.0, 0.1} (two-level) and
{0.1, 0.05} (flat). The journal's frozen protocol calls for **0.7 (mild) and
0.3 (severe)**, which have never been run in this repository at any protocol
(`JOURNAL_SIMULATION_RESULTS.md` §21 item 3). `journal_run.sh s4_noniid`
therefore runs new alphas — it is not duplicating the ECCV hetero grid.

---

## 11. Provenance

| Artifact | Path |
|---|---|
| Run logs (65) | `results/camera_ready/logs/*.log` |
| Resume markers (65) | `results/camera_ready/.markers/*.{done,failed}` |
| Submission driver logs | `results/camera_ready/submission_20260623_{222428,234934}.log` |
| Launchers | `scripts/camera_ready/run_{hetero,partial,profiling}.sh`, `run_all_camera_ready.sh` |
| Analysis entry points | `scripts/camera_ready/run_{covariance,lstat,inversion,partition_diagnostics}.py` |
| Aggregation | `scripts/camera_ready/collect_camera_ready.py`, `make_tables.py` |
| Library | `camera_ready/{partition,synthesis,lstat,inversion,profiling,feature_distance}.py` |
| Stale status line | `results/camera_ready/README.md` (says "not executed yet" — ignore) |
