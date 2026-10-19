# H-SFP Camera-Ready Experiments

Additional experiments for the camera-ready version of **H-SFP: Hierarchical
Federated Learning with Decoupled Split-Model Prototyping**, addressing reviewer
concerns. All new behavior is **additive and flag-gated**; default H-SFP behavior
is unchanged when the new config keys are absent.

> Status: code is implemented and smoke-/syntax-validated, but the full (and most
> smoke) runs have **not** been executed yet. Run the commands below to generate
> results. No numbers in the paper should be taken from here until produced by code.

## Reviewer concern → experiment map

| # | Reviewer concern | Experiment | Launcher | Output dir |
|---|---|---|---|---|
| 1 | Hierarchical heterogeneity | Two-level Dirichlet (α_edge / α_client) | `run_hetero.sh` + `run_partition_diagnostics.py` | `hetero/` |
| 2 | Partial participation | Active-ratio × distribution sweep | `run_partial.sh` | `partial/` |
| 3 | Diagonal Gaussian assumption | Covariance ablation + MMD | `run_covariance.py` | `covariance/` |
| 4 | Lstat estimation | Empirical Lipschitz sensitivity | `run_lstat.py` | `lstat/` |
| 5 | Packing/unpacking overhead | CUDA-synced phase profiling | `run_profiling.sh` | `profiling/` |
| 6 | Privacy / feature inversion | Inversion exposure comparison | `run_inversion.py` | `inversion/` |
| 7 | Baseline fairness / reproducibility | Shared-protocol manifest | `gen_fairness.py` | `fairness/` |

## Quick start

```bash
# Fast end-to-end sanity over ALL experiments (tiny configs, 1 seed):
SMOKE=1 ./scripts/camera_ready/run_all_camera_ready.sh

# See exactly what would run, without executing:
SMOKE=1 DRY_RUN=1 ./scripts/camera_ready/run_all_camera_ready.sh

# Full run (200 clients, 200 rounds, 5 seeds): expensive.
./scripts/camera_ready/run_all_camera_ready.sh
```

Common env vars (see `scripts/camera_ready/common.sh`): `SMOKE`, `SEEDS`,
`METHODS`, `DRY_RUN`, `RESET`, `WANDB_ARGS`. Resume markers live in
`results/camera_ready/.markers/` (a completed run is skipped unless `RESET=1`).

The mechanism: `main.py` gained a generic, additive `--set KEY=VALUE` override (and
the launchers use it) so experiments vary `partition`, `alpha_edge`, `alpha_client`,
`frac`, `profile`, etc. without new per-run config files.

---

## 1. Hierarchical heterogeneity

Two-level Dirichlet: edge-level distribution ~ Dir(α_edge), then within-edge
client distribution ~ Dir(α_client). The partitioner also fixes the client→edge
mapping, which the hierarchy honors (`_client_to_edge`). Settings: (α_e, α_c) ∈
{1.0, 0.1}². Methods: H-SFP, HSFL, SplitFed, FedAvg, HierFL.

```bash
# Smoke (2 edges, 8 clients, 2 rounds, 1 seed):
SMOKE=1 ./scripts/camera_ready/run_hetero.sh
# Full:
./scripts/camera_ready/run_hetero.sh
# Partition diagnostics (histograms per edge/client; no training):
python scripts/camera_ready/run_partition_diagnostics.py \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml --tag cifar100
# Aggregate + LaTeX:
python scripts/camera_ready/collect_camera_ready.py
python scripts/camera_ready/make_tables.py --only hetero
```
Outputs: `hetero/summary_hetero.csv`, `hetero/table_hetero.tex`,
`hetero/partition_diagnostics/diag_*.{json,png,pdf}`.

## 2. Partial participation

Active-client ratio `frac` ∈ {0.1, 0.5, 1.0} × distribution ∈ {IID, Dir(0.05),
Dir(0.1)}, 200 clients. Methods: H-SFP, FedAvg, HierFL, SplitFed, HSFL.

```bash
SMOKE=1 ./scripts/camera_ready/run_partial.sh          # smoke
./scripts/camera_ready/run_partial.sh                  # full (5 seeds)
# Run a reduced grid:
FRACS="0.1 0.5" DISTS="iid dir:0.1" ./scripts/camera_ready/run_partial.sh
python scripts/camera_ready/collect_camera_ready.py
python scripts/camera_ready/make_tables.py --only partial
```
Outputs: `partial/summary_partial.csv`, `partial/table_partial.tex`.

## 3. Covariance ablation

Measured at the pooled edge→cloud synthesis boundary with a frozen client+edge
extractor (random init unless `--ckpt` given). Modes: `mu_only`, `diag_sigma`
(default), `full_covariance`, `low_rank_covariance`, `mixture_gaussian`. Tracks
accuracy, communication (analytic), runtime, and RBF-MMD(real, synth).

```bash
# Smoke:
python scripts/camera_ready/run_covariance.py \
    --cfg configs/camera_ready/smoke/hsfp_smoke.yaml \
    --max-classes 10 --probe-per-class 30 --syn-per-class 30 --clf-epochs 3 --tag smoke
# Full (optionally load a trained extractor for higher absolute accuracy):
python scripts/camera_ready/run_covariance.py \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ckpt classification/H-SFP/checkpoint_hfl.pt --tag cifar100
```
Outputs: `covariance/covariance_<tag>.{csv,json,tex}`.

## 4. Lstat estimation

Empirical Lipschitz sensitivity of (μ, σ) to input Gaussian perturbations
ρ ∈ {0.01, 0.10, 0.20}. **Supports bounded empirical sensitivity only — not a
proof of representation preservation.**

```bash
python scripts/camera_ready/run_lstat.py \
    --cfg configs/camera_ready/smoke/hsfp_smoke.yaml --num-clients 2 --batch 64 --tag smoke
python scripts/camera_ready/run_lstat.py \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ckpt classification/H-SFP/checkpoint_hfl.pt --num-clients 10 --tag cifar100
```
Outputs: `lstat/lstat_<tag>_{raw,agg}.csv`, `lstat/lstat_<tag>.tex`.

## 5. Runtime profiling

H-SFP with `profile=true` records CUDA-synchronized per-phase timings
(`client_pack`, `edge_process`, `cloud_process`, `edge_aggregate`,
`cloud_aggregate`, `total_round`); SplitFed total time is captured by wall clock.

```bash
SMOKE=1 ./scripts/camera_ready/run_profiling.sh        # smoke
./scripts/camera_ready/run_profiling.sh                # full
python scripts/camera_ready/make_tables.py --only profiling
```
Outputs: `profiling/profile_hsfp_<dataset>_{records.csv,summary.json}`,
`profiling/wallclock.csv`, `profiling/table_profiling.tex`. Column mapping:
pack/client/rnd = `client_pack`/#clients; aggregate/edge/rnd = `edge_aggregate`;
synth+train/edge/rnd = `edge_process`.

## 6. Inversion / exposure

Optimization-based feature inversion reconstructing a specific sample from
(a) SplitFed smashed activation, (b) H-SFP μ, (c) H-SFP μ+σ. Metrics: MSE/PSNR
(+ SSIM/LPIPS if `scikit-image`/`lpips` installed). **Empirical exposure, not
formal differential privacy.**

```bash
python scripts/camera_ready/run_inversion.py \
    --cfg configs/camera_ready/smoke/hsfp_smoke.yaml --num-samples 3 --iters 100 --tag smoke
python scripts/camera_ready/run_inversion.py \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ckpt classification/H-SFP/checkpoint_hfl.pt --tag cifar100
```
Outputs: `inversion/inversion_<tag>.{csv,json,tex}`,
`inversion/inversion_grid_<tag>.{png,pdf}`.

## 7. Baseline fairness & reproducibility

```bash
python scripts/camera_ready/gen_fairness.py
```
Outputs: `fairness/baseline_fairness_summary.md`, `fairness/baseline_fairness.tex`.
Source manifest: `configs/camera_ready/baseline_fairness.yaml`.

### Sanity-check commands (documented)
```bash
# Tuned FedAvg (sweep LR), CIFAR-100, IID:
for lr in 1e-3 5e-4 1e-4 5e-5; do
  python main.py --task classification --method federated \
    --cfg configs/classification/federated/cifar_fedavg_resnet50.yaml \
    --seed 0 --set lr=$lr --set iid=true
done
# Centralized oracle (upper bound): train FedAvg with 1 client = centralized:
python main.py --task classification --method federated \
  --cfg configs/classification/federated/cifar_fedavg_resnet50.yaml \
  --seed 0 --set num_users=1 --set frac=1.0 --set iid=true
```

---

## Notes
- Metric reported for tables 1–2 is **test F1 (macro, %)** as printed by every
  method runner ("Test F1: …%").
- Covariance/Lstat/inversion use a **frozen** extractor by design (isolates the
  question under study); pass `--ckpt classification/H-SFP/checkpoint_hfl.pt` to
  use a trained extractor.
- Reusable for CIFAR-10/HAM10000/ImageNet by changing `--set dataset=...` and the
  matching base config; the partitioner is dataset-agnostic.
- Changed/added files are listed in the implementation summary accompanying this work.
