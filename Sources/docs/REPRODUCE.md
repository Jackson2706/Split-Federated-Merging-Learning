# Reproduction Guide

This is the single index for reproducing every result. It assumes you have:

1. Installed dependencies — see the [README](../README.md#installation).
2. Set up datasets — see [docs/datasets.md](datasets.md).

All commands are run from the repository root. Every experiment has a **smoke**
form (tiny, fast, 1 seed — for sanity) and a **full** form (paper settings).

---

## 0. Sanity checks (run these first)

```bash
python main.py --list                      # registry of tasks/methods
python -m camera_ready.partition           # partition self-test (no data needed)
make smoke                                  # one tiny end-to-end H-SFP run
```

---

## 1. Single runs

```bash
# H-SFP (CIFAR-100, ResNet50, edge/cloud intervals t1=5/t2=10)
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml --seed 0

# E-HSFP (all journal extensions)
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ablation full_e_hsfp --seed 0

# Baselines
python main.py --task classification --method federated \
    --cfg configs/classification/federated/cifar_fedavg_resnet50.yaml --seed 0
python main.py --task classification --method hierfl  --cfg configs/classification/hierfl/cifar_hierfl_resnet50.yaml --seed 0
python main.py --task classification --method splitfl --cfg configs/classification/splitfl/cifar_splitfed_resnet50.yaml --seed 0
python main.py --task classification --method hsfl    --cfg configs/classification/hsfl/cifar_our_resnet50_5_10.yaml --seed 0

# Segmentation (ISIC-2018)
python main.py --task segmentation --method h-sfp \
    --cfg configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml --seed 0
```

Outputs: per-run metrics are printed (`Test F1: …%` / IoU+Dice) and saved under each
method's `Figure/data/*.json`; communication and timing are reported at the end.

---

## 2. Journal experiments (E-HSFP)

Convergence, component ablation, dropout/staleness robustness, aggregation intervals.

```bash
# Smoke
DRY_RUN=0 bash scripts/journal_experiments/run_smoke.sh
# Full suite (5 seeds)
bash scripts/journal_experiments/run_all_journal.sh
# Aggregate to summary CSVs
python scripts/journal_experiments/collect_results.py
```

Details and per-group scripts: [scripts/journal_experiments/README.md](../scripts/journal_experiments/README.md).

---

## 3. Camera-ready experiments (H-SFP)

Each addresses a specific reviewer concern. Full smoke + full commands are in
[results/camera_ready/README.md](../results/camera_ready/README.md). Quick index:

| # | Experiment | Smoke | Full launcher |
|---|------------|-------|---------------|
| 1 | Hierarchical heterogeneity (two-level Dirichlet) | `SMOKE=1 ./scripts/camera_ready/run_hetero.sh` | `./scripts/camera_ready/run_hetero.sh` |
| 2 | Partial participation | `SMOKE=1 ./scripts/camera_ready/run_partial.sh` | `./scripts/camera_ready/run_partial.sh` |
| 3 | Covariance ablation | `python scripts/camera_ready/run_covariance.py --cfg configs/camera_ready/smoke/hsfp_smoke.yaml --max-classes 10 --probe-per-class 30 --syn-per-class 30 --clf-epochs 3 --tag smoke` | `python scripts/camera_ready/run_covariance.py --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml --ckpt checkpoint_hfl.pt --tag cifar100` |
| 4 | Lstat sensitivity | `python scripts/camera_ready/run_lstat.py --cfg configs/camera_ready/smoke/hsfp_smoke.yaml --num-clients 2 --batch 64 --tag smoke` | `python scripts/camera_ready/run_lstat.py --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml --num-clients 10 --tag cifar100` |
| 5 | Runtime profiling | `SMOKE=1 ./scripts/camera_ready/run_profiling.sh` | `./scripts/camera_ready/run_profiling.sh` |
| 6 | Feature-inversion exposure | `python scripts/camera_ready/run_inversion.py --cfg configs/camera_ready/smoke/hsfp_smoke.yaml --num-samples 3 --iters 100 --tag smoke` | `python scripts/camera_ready/run_inversion.py --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml --tag cifar100` |
| 7 | Baseline fairness summary | `python scripts/camera_ready/gen_fairness.py` | (same) |

Aggregate + build LaTeX tables:

```bash
python scripts/camera_ready/collect_camera_ready.py
python scripts/camera_ready/make_tables.py
```

Run everything (smoke) in one shot:

```bash
SMOKE=1 ./scripts/camera_ready/run_all_camera_ready.sh
```

---

## 4. Figures

```bash
python tools/plot_communication.py communication_overhead.png
```

---

## Reproducibility notes

- Use `--seed S` for deterministic runs; launchers sweep `SEEDS="0 1 2 3 4"` by default.
- Re-running a launcher skips completed runs (resume markers under
  `results/**/.markers/`); pass `RESET=1` to force re-run.
- The shared protocol for all methods is recorded in
  [configs/camera_ready/baseline_fairness.yaml](../configs/camera_ready/baseline_fairness.yaml).
- Override any config key without editing files: `--set key=value` (repeatable).
