# E-HSFP: Episodic Hierarchical Split-Federated Prototyping

> **Journal extension** of **H-SFP** (Hierarchical Split-Federated Learning with Prototypes) — ECCV 2026.

E-HSFP extends H-SFP with episodic prototype memory, learnable reliability-aware aggregation, prototype replay consistency loss, serverless prototype dropout regularization, and a lightweight serverless episode simulator — enabling robust, stateless cloud-edge federated learning.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Methods](#methods)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [E-HSFP Components](#e-hsfp-components)
- [Ablation Modes](#ablation-modes)
- [Journal Experiments](#journal-experiments)
- [Baselines](#baselines)
- [Weights \& Biases Integration](#weights--biases-integration)
- [Serverless Roadmap](#serverless-roadmap)

---

## Overview

This repository implements a **3-tier federated learning hierarchy**:

```
Client (SSL) ──→ Edge (SSL) ──→ Cloud (Supervised)
```

- **Clients** learn local representations via self-supervised learning (SSL), extract per-class prototypes with uncertainty estimates (mu, sigma), and transmit them upstream.
- **Edge servers** aggregate client prototypes and refine representations via SSL.
- **Cloud server** performs final supervised training and periodically aggregates global model weights (FedAvg) at configurable intervals (t1, t2).

**H-SFP** (the base method) transmits only prototypes and distributions between tiers — not full model weights — making it communication-efficient and naturally suited for serverless deployment.

**E-HSFP** (the journal extension) adds:
- Episodic prototype memory with replay mixing
- Learnable reliability-aware prototype aggregation
- Prototype replay consistency (PRC) loss
- Serverless prototype dropout regularization
- Lightweight serverless episode simulator
- Optional residual prototype generator

All E-HSFP features are **config-driven and disabled by default**, preserving full backward compatibility with H-SFP.

---

## Architecture

### Training Phases per Round

| Phase | H-SFP | E-HSFP Addition |
|-------|-------|-----------------|
| 1 | Client SSL: learn features, extract (prototype, std) per class | **Store in episodic memory** |
| 2 | — | **Mix current prototypes with memory prototypes** (configurable replay ratio) |
| 3 | Edge SSL: aggregate client prototypes | **Reliability-weighted aggregation + PRC loss** |
| 4 | Cloud supervised: aggregate edge prototypes | **Reliability-weighted aggregation + PRC loss** |
| 5 | Aggregation & validation (FedAvg at t1/t2 intervals) | **Age memories, train reliability network, log E-HSFP metrics** |

### Tasks

| Task | Datasets | Primary Metric |
|------|----------|---------------|
| Classification | CIFAR-10/100, HAM10000 | F1 Score |
| Segmentation | ISIC-2018 | IoU, Dice |

---

## Methods

| Method | CLI Name | Description | Type |
|--------|----------|-------------|------|
| **H-SFP** | `h-sfp` | Hierarchical Split-Federated Prototyping (ECCV 2026) | Primary |
| **E-HSFP** | `h-sfp --ablation full_e_hsfp` | Episodic H-SFP with all journal extensions | Primary (extended) |
| FedAvg/FedProx/FedNova/FedSGD | `federated` | Standard federated learning strategies | Baseline |
| HierFL | `hierfl` | Hierarchical Federated Learning | Baseline |
| SplitFL | `splitfl` | Split Federated Learning | Baseline |
| HeteroSFL | `hetero-sfl` | Heterogeneous Split FL (wide/narrow) | Baseline |
| HSFL | `hsfl` | Hierarchical Split FL | Baseline |

---

## Project Structure

```
.
├── main.py                        # Single entry point for all experiments
├── run.sh                         # Batch runner (all methods, --wandb support)
├── requirement.txt                # Python dependencies (CUDA 12.8, PyTorch 2.7)
│
├── ehsfp/                         # Shared E-HSFP extension library
│   ├── __init__.py                #   Package exports (v0.1.0)
│   ├── config.py                  #   EHSFP_DEFAULTS, ABLATION_PRESETS, get_ehsfp_config()
│   ├── memory.py                  #   EpisodicPrototypeMemory, PrototypeRecord, mix_current_and_memory()
│   ├── reliability.py             #   PrototypeReliabilityNetwork, build_reliability_features()
│   ├── aggregation.py             #   reliability_weighted_aggregate(), train_reliability_bootstrap()
│   ├── losses.py                  #   prototype_replay_consistency_loss(), dropout_consistency_loss()
│   ├── dropout.py                 #   PrototypeDropout (simulates serverless failures)
│   ├── serverless_sim.py          #   ServerlessEpisode, ServerlessMetricsTracker
│   ├── generator.py               #   ResidualPrototypeGenerator (optional)
│   └── metrics_logger.py          #   EHSFPMetricsLogger (wandb integration)
│
├── configs/                       # All YAML configs (one centralized location)
│   ├── classification/
│   │   ├── h-sfp/                 #   Primary method configs (CIFAR, HAM10000, ImageNet)
│   │   ├── federated/             #   FedAvg, FedProx, FedNova, FedSGD
│   │   ├── hierfl/
│   │   ├── splitfl/
│   │   ├── hetero-sfl/
│   │   └── hsfl/
│   └── segmentation/
│       ├── h-sfp/                 #   ISIC-2018 configs (t1/t2 variants)
│       ├── federated/
│       ├── hierfl/
│       ├── splitfl/
│       ├── hetero-sfl/
│       └── hsfl/
│
├── classification/
│   ├── H-SFP/                    # Primary method (classification)
│   │   ├── runner.py              #   run(cfg_path) — called by main.py
│   │   ├── hierarchy.py           #   5-phase training pipeline + E-HSFP hooks
│   │   ├── core/                  #   prototype.py, ssl.py, aggregation.py, losses.py
│   │   ├── models/                #   3-tier model splits (AlexNet, ResNet50, VGG)
│   │   ├── data/                  #   Dataset loaders + IID/non-IID sampling
│   │   └── config/                #   ConfigLoader (YAML loader)
│   ├── Federated/                 #   FedAvg/FedProx/FedNova/FedSGD baseline
│   ├── HierFL/                    #   Hierarchical FL baseline
│   ├── SplitFL/                   #   Split FL baseline
│   ├── HeteroSFL/                 #   Heterogeneous Split FL baseline
│   └── HSFL/                      #   Hierarchical Split FL baseline
│
├── segmentation/
│   ├── H-SFP/                    # Primary method (segmentation)
│   │   ├── runner.py
│   │   ├── hierarchy.py           #   5-phase pipeline + decoder training for masks
│   │   ├── clients/               #   DiceFocalLoss, test (IoU/Dice metrics)
│   │   ├── models/                #   3-tier split + ISICCloudDecoder for inference
│   │   ├── data/                  #   ISIC-2018 loader
│   │   └── config/                #   ConfigLoader
│   ├── Federated/
│   ├── HierFL/
│   ├── SplitFL/
│   ├── HeteroSFL/
│   └── HSFL/
│
├── serverless/                    # Serverless deployment interfaces (future)
│   ├── interfaces/                #   CommunicationBackend, AggregatorBackend (abstract)
│   └── adapters/                  #   LocalCommunicationBackend (current in-process impl)
│
├── scripts/
│   └── journal_experiments/       # E-HSFP journal experiment scripts
│       ├── common.sh              #   Shared functions (logging, resume, seed dispatch)
│       ├── run_smoke.sh           #   Quick sanity check
│       ├── run_convergence.sh     #   H-SFP vs E-HSFP vs baselines (5 seeds)
│       ├── run_ablation.sh        #   6 component ablation presets
│       ├── run_dropout_staleness.sh #   Dropout rate + staleness sweeps
│       ├── run_intervals.sh       #   Aggregation interval experiments
│       ├── run_all_journal.sh     #   Master launcher
│       ├── collect_results.py     #   Parse logs → summary CSVs
│       └── README.md              #   Experiment documentation
│
└── Figure/                        # Plotting scripts and output figures
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Split-Federated-Merging-Learning

# Install dependencies (requires CUDA 12.8)
pip install -r requirement.txt

# Optional: enable experiment tracking
pip install wandb && wandb login
```

**Requirements:** Python 3.10+, PyTorch 2.7+, CUDA 12.8

---

## Quick Start

```bash
# List all available methods
python main.py --list

# List available configs
python main.py --list-configs --task classification --method h-sfp

# Run H-SFP (reference method)
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml

# Run E-HSFP (full journal extension)
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ablation full_e_hsfp

# Run with specific seed for reproducibility
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ablation full_e_hsfp --seed 42

# Run segmentation
python main.py --task segmentation --method h-sfp \
    --cfg configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml

# Run a baseline
python main.py --task classification --method federated \
    --cfg configs/classification/federated/cifar_fedavg_resnet50.yaml

# Batch run all methods
./run.sh                          # everything
./run.sh classification           # classification only
./run.sh segmentation             # segmentation only
./run.sh ablation                 # E-HSFP ablation study
```

### CLI Arguments

| Flag | Description |
|------|-------------|
| `--task` | `classification` or `segmentation` |
| `--method` | Method name (see [Methods](#methods)) |
| `--cfg` | Path to YAML config file |
| `--seed` | Random seed (sets `random`, `numpy`, `torch`, `PYTHONHASHSEED`) |
| `--ablation` | E-HSFP ablation preset (only with `--method h-sfp`) |
| `--wandb` | Enable Weights & Biases logging |
| `--wandb-project` | W&B project name (default: `H-SFP`) |
| `--wandb-entity` | W&B team/entity |

---

## Configuration

All configs live under `configs/<task>/<method>/`. Each experiment config inherits from a `default.yaml` in the same directory.

**Naming convention:**
- H-SFP/HSFL: `{dataset}_{strategy}_{model}_{t1}_{t2}.yaml`
- Baselines: `{dataset}_{strategy}_{model}.yaml`

### Shared Hyperparameters (Fair Comparison)

| Parameter | Classification | Segmentation |
|-----------|---------------|--------------|
| `num_users` | 200 | 50 |
| `epochs` | 200 | 200 |
| `frac` | 0.1 | 0.1 |
| `local_bs` | 16 | 16 |
| `local_ep` | 5 | 5 |
| `optimizer` | adam | adam |
| `lr` | 1e-4 | 1e-4 |
| `iid` | true | true |

---

## E-HSFP Components

All E-HSFP features live in the shared `ehsfp/` library and are enabled via config keys in `default.yaml`. They are disabled by default.

### 1. Episodic Prototype Memory (`ehsfp/memory.py`)
Stores historical prototype records `(class_id, mu, sigma, support_count, age, reliability)` and mixes them with current-round prototypes using a configurable replay ratio.

| Config Key | Default | Description |
|------------|---------|-------------|
| `use_episodic_memory` | `false` | Enable memory |
| `memory_size` | `1000` | Max records per memory |
| `memory_replay_ratio` | `0.3` | Alpha for mixing: Q = alpha * current + (1-alpha) * memory |
| `memory_top_k` | `5` | Top-k reliable prototypes per class for replay |
| `max_prototype_age` | `10` | Evict prototypes older than this |

### 2. Learnable Reliability Aggregation (`ehsfp/reliability.py`, `ehsfp/aggregation.py`)
A small neural network predicts per-prototype reliability weights from features like support count, variance, age, and loss. Replaces simple averaging with reliability-weighted aggregation.

| Config Key | Default | Description |
|------------|---------|-------------|
| `aggregation_mode` | `"simple"` | `"simple"` or `"reliability"` |
| `reliability_hidden_dim` | `64` | Hidden layer size |
| `reliability_lr` | `1e-3` | Learning rate for reliability network |
| `reliability_bootstrap_epochs` | `5` | Bootstrap training epochs per round |

### 3. Prototype Replay Consistency Loss (`ehsfp/losses.py`)
L2 consistency loss between current and memory prototype representations, encouraging stable feature learning across rounds.

| Config Key | Default | Description |
|------------|---------|-------------|
| `use_prc_loss` | `false` | Enable PRC loss |
| `lambda_prc` | `0.1` | PRC loss weight |
| `prc_num_samples` | `5` | Number of memory prototypes to sample |

### 4. Serverless Prototype Dropout (`ehsfp/dropout.py`)
Simulates serverless failure modes (function timeouts, cold starts) by randomly dropping prototype packets during aggregation, improving robustness.

| Config Key | Default | Description |
|------------|---------|-------------|
| `use_prototype_dropout` | `false` | Enable dropout |
| `prototype_dropout_rate` | `0.1` | Probability of dropping each prototype |
| `dropout_mode` | `"class"` | `"class"` (per-class) or `"source"` (per-client) |

### 5. Serverless Episode Simulator (`ehsfp/serverless_sim.py`)
Tracks simulated cold starts, timeouts, latency, and cost proxies. These are secondary metrics for paper analysis — they do not affect training.

| Config Key | Default | Description |
|------------|---------|-------------|
| `use_serverless_simulation` | `false` | Enable simulator |
| `cold_start_probability` | `0.1` | Simulated cold start rate |
| `function_timeout_probability` | `0.05` | Simulated timeout rate |

### 6. Residual Prototype Generator (`ehsfp/generator.py`)
Optional module: `z = mu + sigma * eps + scale * G(cat(mu, sigma, eps))` for generating augmented prototype samples.

| Config Key | Default | Description |
|------------|---------|-------------|
| `use_residual_generator` | `false` | Enable generator |
| `generator_hidden_dim` | `128` | Generator hidden size |

---

## Ablation Modes

E-HSFP supports 6 progressive ablation presets via `--ablation`:

| Mode | Memory | Dropout | Reliability | PRC | Serverless Sim | Generator |
|------|--------|---------|-------------|-----|----------------|-----------|
| `baseline_hsfp` | - | - | - | - | - | - |
| `hsfp_memory` | Y | - | - | - | - | - |
| `hsfp_memory_dropout` | Y | Y | - | - | - | - |
| `hsfp_memory_reliability` | Y | - | Y | - | - | - |
| `hsfp_memory_reliability_prc` | Y | - | Y | Y | - | - |
| `full_e_hsfp` | Y | Y | Y | Y | Y | Y |

```bash
# Run specific ablation
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ablation hsfp_memory_reliability_prc

# Batch ablation study
./run.sh ablation
```

---

## Journal Experiments

Complete experiment scripts for the E-HSFP journal paper are in `scripts/journal_experiments/`.

```bash
cd scripts/journal_experiments

# Quick smoke test
./run_smoke.sh

# Run all journal experiments
./run_all_journal.sh

# Run specific experiment groups
./run_all_journal.sh convergence ablation

# Collect results into summary CSVs
python collect_results.py
```

### Experiment Groups

| Script | Description |
|--------|-------------|
| `run_convergence.sh` | H-SFP vs E-HSFP vs 6 baselines across all datasets, 5 seeds |
| `run_ablation.sh` | 6 progressive ablation presets (CIFAR + HAM10000 + ISIC) |
| `run_dropout_staleness.sh` | Dropout rate sweep (0.0–0.7) + staleness sweep (tau 0–10) |
| `run_intervals.sh` | Aggregation intervals (Ic,Ie) = (5,10), (10,20), (25,50) |

### Features

- **5 seeds** (0–4) for statistical significance
- **Resume support** via `.done`/`.failed` markers — re-run safely after crashes
- **GPU selection** via `GPU_ID` env var
- **Dry-run mode** via `DRY_RUN=1`
- **W&B integration** via `WANDB_ARGS="--wandb --wandb-project E-HSFP"`

See [scripts/journal_experiments/README.md](scripts/journal_experiments/README.md) for full documentation.

---

## Baselines

All baselines use the same hyperparameters and data splits as H-SFP for fair comparison.

| Baseline | Classification Configs | Segmentation Configs |
|----------|----------------------|---------------------|
| FedAvg | CIFAR (AlexNet, ResNet50), HAM10000 (ResNet50, VGG) | ISIC (ResNet50) |
| FedProx | CIFAR (AlexNet, ResNet50), HAM10000 (ResNet50, VGG) | ISIC (ResNet50) |
| FedNova | CIFAR (AlexNet, ResNet50), HAM10000 (ResNet50, VGG) | ISIC (ResNet50) |
| FedSGD | CIFAR (AlexNet, ResNet50), HAM10000 (ResNet50, VGG) | ISIC (ResNet50) |
| HierFL | CIFAR (AlexNet, ResNet50), HAM10000 (ResNet50, VGG) | ISIC (ResNet50) |
| SplitFL | CIFAR (AlexNet, ResNet50), HAM10000 (ResNet50, VGG) | — |
| HeteroSFL | CIFAR (AlexNet, ResNet18), HAM10000 (ResNet50, VGG) | ISIC (ResNet50) |
| HSFL | CIFAR (AlexNet, ResNet50), HAM10000 (ResNet50, VGG) | CIFAR, HAM10000 |

---

## Weights & Biases Integration

```bash
# Run with W&B
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --wandb --wandb-project E-HSFP

# Batch with W&B
./run.sh --wandb --wandb-project=E-HSFP
```

**Metrics logged per epoch:**
- Classification: `train_loss`, `f1`, `best_f1`, client resource usage, comm costs
- Segmentation: `train_loss`, `iou`, `dice`, client resource usage, comm costs
- E-HSFP extras: memory stats, reliability weights, PRC loss, dropout stats, serverless metrics
- Final summary: `test_f1` (or `test_iou`/`test_dice`), `total_time_s`

---

## Serverless Roadmap

H-SFP transmits only prototypes (small tensors) — not model weights — making each round stateless and naturally suited for function-as-a-service deployment.

The `serverless/` directory contains abstract interfaces (`CommunicationBackend`, `AggregatorBackend`) and a local adapter. Future work will add cloud transport adapters (AWS SQS, GCP Pub/Sub, Redis) to enable true serverless deployment without changing model or training code.

See `.claude/serverless.md` for the full roadmap.

---

## Citation

```
@inproceedings{hsfp2026,
  title     = {H-SFP: Hierarchical Split-Federated Learning with Prototypes},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```
