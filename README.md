# H-SFP & E-HSFP

**Hierarchical Federated Learning with Decoupled Split-Model Prototyping**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains two closely related methods built on a single, unified codebase:

- **H-SFP** *(ECCV)* — **H**ierarchical Federated Learning with Decoupled **S**plit-Model
  **P**rototyping. A communication-efficient 3-tier (client → edge → cloud) split-federated
  method that transmits only **class-wise prototype statistics** `(μ, σ)` between tiers
  instead of raw activations, gradients, or per-round full model weights.
- **E-HSFP** *(journal extension)* — **E**pisodic H-SFP. Adds episodic prototype memory,
  learnable reliability-aware aggregation, a prototype replay consistency (PRC) loss,
  serverless prototype dropout, and a serverless episode simulator for stateless cloud-edge
  deployment.

> All E-HSFP features are **config-driven and disabled by default**, so the default pipeline
> is exactly H-SFP. This keeps the base method reproducible and the extension opt-in.

```
Client (SSL)  ──►  Edge (SSL)  ──►  Cloud (Supervised)
   (μ, σ)            (μ, σ)            classifier / decoder
```

---

## Table of contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Methods & baselines](#methods--baselines)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [E-HSFP components & ablations](#e-hsfp-components--ablations)
- [Experiments](#experiments)
- [Project structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **Communication-efficient.** Only prototype statistics flow between tiers — orders of
  magnitude smaller than smashed activations or model weights (see
  [`tools/plot_communication.py`](tools/plot_communication.py)).
- **Unified entry point.** One `main.py` dispatches every task/method via a registry.
- **Strong baselines included.** FedAvg/FedProx/FedNova/FedSGD, HierFL, SplitFL, HSFL, HeteroSFL.
- **Two tasks.** Image classification (CIFAR-10/100, HAM10000, ImageNet) and medical image
  segmentation (ISIC-2018).
- **Reproducible experiments.** Journal experiment suite plus a camera-ready experiment
  suite (heterogeneity, partial participation, covariance ablation, Lstat, profiling,
  inversion, fairness) with one-command launchers.

---

## Installation

Requires Python ≥ 3.9. Tested with PyTorch 2.7 + CUDA 12.8 on an NVIDIA RTX 3080 Ti.

```bash
git clone https://github.com/Jackson2706/Split-Federated-Merging-Learning.git
cd Split-Federated-Merging-Learning

python -m venv .venv && source .venv/bin/activate

# Install a CUDA build of torch first (match your CUDA version), e.g. CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e .          # optional: install ehsfp / camera_ready / serverless as packages

# or simply:
make install-cuda         # CUDA torch + requirements   (see `make help`)
```

Then set up datasets under `data/` (symlink or download) — see
[docs/datasets.md](docs/datasets.md). Verify with `make data-check`. CIFAR auto-downloads.

---

## Quick start

```bash
# List methods and configs
python main.py --list
python main.py --list-configs --task classification

# Run H-SFP on CIFAR-100 (ResNet50, t1=5 / t2=10)
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml

# Run E-HSFP (all journal extensions) via the ablation flag
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ablation full_e_hsfp

# Run a baseline
python main.py --task classification --method federated \
    --cfg configs/classification/federated/cifar_fedavg_resnet50.yaml

# Segmentation (ISIC-2018)
python main.py --task segmentation --method h-sfp \
    --cfg configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml

# Override any config key on the fly (repeatable), and set a seed
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --seed 0 --set frac=0.5 --set iid=false
```

`--ablation`, `--seed`, and `--set KEY=VALUE` are injected into a temporary copy of the YAML
config, so on-disk configs stay clean. Add `--wandb` for Weights & Biases logging.

---

## Methods & baselines

| Method | `--method` | Description | Role |
|--------|-----------|-------------|------|
| **H-SFP** | `h-sfp` | Hierarchical split-federated prototyping (ECCV) | Primary |
| **E-HSFP** | `h-sfp --ablation full_e_hsfp` | Episodic H-SFP (journal extension) | Primary (extended) |
| FedAvg / FedProx / FedNova / FedSGD | `federated` | Standard FL aggregation strategies | Baseline |
| HierFL | `hierfl` | Hierarchical federated learning | Baseline |
| SplitFL | `splitfl` | Split federated learning | Baseline |
| HSFL | `hsfl` | Hierarchical split FL | Baseline |
| HeteroSFL | `hetero-sfl` | Heterogeneous (wide/narrow) split FL | Baseline |

---

## Datasets

| Task | Datasets | Metric |
|------|----------|--------|
| Classification | CIFAR-10, CIFAR-100, HAM10000, ImageNet | F1 (macro) |
| Segmentation | ISIC-2018 | IoU, Dice |

Datasets are not bundled. All configs use a portable, repo-relative `data/` layout; populate
it by symlink or download (and override any path with `--set dataset_root=...`). Full
instructions: [docs/datasets.md](docs/datasets.md).

---

## Configuration

Configs are YAML with single-key `base:` inheritance resolved by
[`ConfigLoader`](classification/H-SFP/config/config_loader.py):

```yaml
base: default.yaml          # inherit + override
strategy: "hier_fedavg"
dataset: "cifar100"
dataset_root: "/path/to/cifar/"
model: "resnet50"
num_classes: 100
t1: 5                       # edge aggregation interval
t2: 10                      # cloud aggregation interval
```

Common keys: `num_users`, `mid_server` (edges per layer), `frac` (active-client ratio),
`local_bs`, `epochs`, `lr`, `optimizer`, `ssl_epochs_client/edge`, `syn_epochs_cloud`,
`syn_samples_per_class`. See [docs/usage.md](docs/usage.md) for the full CLI reference and
[docs/architecture.md](docs/architecture.md) for the design.

---

## E-HSFP components & ablations

| Component | Config flag |
|-----------|-------------|
| Episodic prototype memory + replay | `use_episodic_memory`, `memory_replay_ratio` |
| Learnable reliability-aware aggregation | `aggregation_mode: learnable_reliability` |
| Prototype replay consistency (PRC) loss | `use_prc_loss`, `lambda_prc` |
| Serverless prototype dropout | `use_prototype_dropout`, `prototype_dropout_rate` |
| Serverless episode simulator | `use_serverless_simulation` |
| Residual prototype generator | `use_residual_generator` |

Ablation presets (via `--ablation`):

| Mode | Memory | Reliability | PRC | Dropout |
|------|:------:|:-----------:|:---:|:-------:|
| `baseline_hsfp` | – | – | – | – |
| `hsfp_memory` | ✓ | – | – | – |
| `hsfp_memory_dropout` | ✓ | – | – | ✓ |
| `hsfp_memory_reliability` | ✓ | ✓ | – | – |
| `hsfp_memory_reliability_prc` | ✓ | ✓ | ✓ | – |
| `full_e_hsfp` | ✓ | ✓ | ✓ | ✓ |

---

## Experiments

> **One place to reproduce everything:** [docs/REPRODUCE.md](docs/REPRODUCE.md) maps each
> result to its exact smoke + full command. `make help` lists shortcut targets.

- **Batch runner.** `./run.sh [classification|segmentation] [--wandb]` runs the configured set.
- **Journal experiments** (E-HSFP): convergence, ablation, dropout/staleness, intervals —
  see [scripts/journal_experiments/README.md](scripts/journal_experiments/README.md).
- **Camera-ready experiments** (H-SFP): hierarchical heterogeneity (two-level Dirichlet),
  partial participation, covariance ablation, Lstat sensitivity, runtime profiling,
  feature-inversion exposure, and a baseline-fairness summary — with smoke + full commands in
  [results/camera_ready/README.md](results/camera_ready/README.md).

```bash
# Camera-ready smoke (fast end-to-end sanity over all experiments)
SMOKE=1 ./scripts/camera_ready/run_all_camera_ready.sh
```

---

## Project structure

```
.
├── main.py                  # Unified entry point (task/method dispatch)
├── run.sh                   # Batch runner
├── Makefile                 # Shortcut targets (make help)
├── requirements.txt
├── pyproject.toml           # Installs ehsfp / camera_ready / serverless
├── data/                    # Datasets (git-ignored; see docs/datasets.md)
├── configs/                 # YAML configs (classification/, segmentation/, camera_ready/)
├── classification/          # Per-method code: H-SFP + baselines
├── segmentation/            # Per-method code for ISIC-2018
├── ehsfp/                   # E-HSFP shared library
├── camera_ready/            # Camera-ready experiment utilities
├── serverless/              # Serverless backend interfaces
├── scripts/                 # Experiment launchers (journal_experiments/, camera_ready/)
├── tools/                   # Standalone plotting scripts
└── docs/                    # architecture, datasets, REPRODUCE, usage, conventions, serverless
```

See [docs/architecture.md](docs/architecture.md) for the full layout and the `main.py`
dispatch mechanism.

---

## Reproducibility

- Deterministic seeding (`random`/`numpy`/`torch`, cuDNN deterministic) via `--seed`.
- Experiment launchers use resume markers (`.markers/`) so re-runs skip completed work.
- The baseline-fairness manifest
  ([`configs/camera_ready/baseline_fairness.yaml`](configs/camera_ready/baseline_fairness.yaml))
  records the shared protocol (clients/edges, sampling ratio, schedule, optimizer, seeds,
  hardware, communication formulas) applied to all methods.

---

## Citation

If you use this code, please cite (see [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{hsfp,
  title     = {H-SFP: Hierarchical Federated Learning with Decoupled Split-Model Prototyping},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## License

Released under the [MIT License](LICENSE). Contributions welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).
