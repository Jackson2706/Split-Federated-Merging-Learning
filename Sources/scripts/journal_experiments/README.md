# E-HSFP Journal Experiment Scripts

Scripts for running all experiments in the E-HSFP journal paper. H-SFP is treated as the **reference baseline**; E-HSFP is the proposed extension. They do not overlap.

## Quick Start

```bash
cd scripts/journal_experiments

# 1. Smoke test (verify everything works, ~minutes)
./run_smoke.sh

# 2. Run all journal experiments (hours/days depending on hardware)
./run_all_journal.sh

# 3. Collect results into summary CSVs
python collect_results.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `common.sh` | Shared functions (logging, markers, resume, seed dispatch) |
| `run_smoke.sh` | Quick sanity check — 1 seed, 1 dataset per task |
| `run_convergence.sh` | Convergence curves: H-SFP vs E-HSFP vs baselines, 5 seeds |
| `run_ablation.sh` | Component ablation: 6 presets x datasets x 5 seeds |
| `run_dropout_staleness.sh` | Robustness: dropout rate sweep + staleness sweep |
| `run_intervals.sh` | Aggregation intervals: (5,10), (10,20), (25,50) |
| `run_all_journal.sh` | Master launcher — runs all groups sequentially |
| `collect_results.py` | Parse logs, produce summary CSVs |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_ID` | `0` | CUDA device (sets `CUDA_VISIBLE_DEVICES`) |
| `SEEDS` | `0 1 2 3 4` | Space-separated seed list |
| `WANDB_ARGS` | (empty) | E.g. `--wandb --wandb-project E-HSFP` |
| `DRY_RUN` | `0` | Set to `1` to print commands without running |

## Resume Support

Every experiment writes a `.done` marker on success or `.failed` on failure under `results/journal/.markers/`. Re-running any script automatically skips completed experiments.

```bash
# Re-run (only failed/pending experiments execute)
./run_all_journal.sh

# Force re-run everything
./run_all_journal.sh --reset

# Run specific groups
./run_all_journal.sh convergence ablation
```

## Output Structure

```
results/journal/
  logs/                    # Per-experiment log files
  .markers/                # .done / .failed resume markers
  summary_convergence.csv  # Aggregated results (after collect_results.py)
  summary_ablation.csv
  summary_dropout.csv
  summary_staleness.csv
  summary_intervals.csv
  summary_all.csv
```

## Experiment Groups

### Convergence (`run_convergence.sh`)
- **H-SFP** (reference): CIFAR/HAM10000/ISIC, ResNet50+AlexNet/VGG backbones
- **E-HSFP** (full): same configs with `--ablation full_e_hsfp`
- **Baselines**: FedAvg, FedProx, FedNova, FedSGD, HierFL, SplitFL, HeteroSFL, HSFL

### Ablation (`run_ablation.sh`)
Progressive component addition:
1. `baseline_hsfp` — vanilla H-SFP (no E-HSFP features)
2. `hsfp_memory` — + episodic memory
3. `hsfp_memory_dropout` — + prototype dropout
4. `hsfp_memory_reliability` — + reliability aggregation
5. `hsfp_memory_reliability_prc` — + PRC loss
6. `full_e_hsfp` — all features

### Dropout & Staleness (`run_dropout_staleness.sh`)
- Dropout rates: 0.0, 0.1, 0.3, 0.5, 0.7
- Staleness (max_prototype_age): 0, 1, 3, 5, 10

### Intervals (`run_intervals.sh`)
- (Ic=5, Ie=10), (Ic=10, Ie=20), (Ic=25, Ie=50)
- Both H-SFP and E-HSFP across CIFAR, HAM10000, ISIC

## Notes

- **FedProto, FedGen, FedDF** are not yet implemented — convergence script warns and skips them.
- The `--seed` flag in `main.py` sets `random`, `numpy`, `torch`, and `PYTHONHASHSEED` for full reproducibility.
- Dropout/staleness scripts create temporary YAML configs with overridden parameters; these are cleaned up after each run.
