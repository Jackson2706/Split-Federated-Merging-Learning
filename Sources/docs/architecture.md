# Architecture

## Overview

This repository implements a **3-tier split-federated hierarchy**:

```
Client (SSL)  →  Edge (SSL)  →  Cloud (Supervised)
```

- **H-SFP** (ECCV) — *Hierarchical Federated Learning with Decoupled Split-Model
  Prototyping*. Clients learn local representations via self-supervised learning
  (SSL) and transmit only class-wise prototype statistics `(μ, σ)` upstream. Edges
  aggregate and refine; the cloud synthesizes features from the statistics and
  trains the classifier. Only prototypes/statistics are communicated between tiers
  — never raw activations or full model weights every round — which makes the
  method communication-efficient.
- **E-HSFP** (journal extension) — adds episodic prototype memory, learnable
  reliability-aware aggregation, a prototype replay consistency (PRC) loss,
  serverless prototype dropout, and a lightweight serverless episode simulator for
  stateless cloud-edge settings. All E-HSFP features are **off by default**, so the
  default pipeline is exactly H-SFP.

## Per-round training phases (E-HSFP additions in *italics*)

1. **Client SSL** — learn features, extract `(prototype, std)` per class; *store in episodic memory*.
2. *Mix current prototypes with memory prototypes* (configurable replay ratio).
3. **Edge SSL** — aggregate client prototypes (*reliability-weighted* when enabled), refine; *PRC loss*.
4. **Cloud supervised** — aggregate edge prototypes, synthesize features, train classifier; *PRC loss*.
5. *Age memories, train reliability network, log E-HSFP metrics.*
6. **Aggregation & validation** — FedAvg at `t1` (edge) and `t2` (cloud) intervals.

## Directory structure

```
.
├── main.py                  # Single entry point (task/method dispatch via importlib)
├── run.sh                   # Batch runner (all experiments, --wandb support)
├── requirements.txt
├── pyproject.toml           # Installs ehsfp / camera_ready / serverless as packages
├── configs/                 # All YAML configs (base: inheritance)
│   ├── classification/{h-sfp,federated,hierfl,splitfl,hetero-sfl,hsfl}/
│   ├── segmentation/{...}/
│   └── camera_ready/        # Smoke configs + baseline_fairness manifest
├── classification/
│   ├── H-SFP/               # PRIMARY METHOD
│   │   ├── runner.py        # run(cfg_path) called by main.py
│   │   ├── hierarchy.py     # Self-contained 5-phase training pipeline
│   │   ├── models/          # 3-tier model splits
│   │   ├── data/            # Dataset loaders + IID / non-IID / Dirichlet sampling
│   │   └── config/          # ConfigLoader (YAML inheritance)
│   └── {Federated,HierFL,SplitFL,HeteroSFL,HSFL}/   # Baselines
├── segmentation/            # Same method set for ISIC-2018 segmentation
├── ehsfp/                   # E-HSFP shared library (memory, reliability, losses, dropout, ...)
├── camera_ready/            # Reusable utilities for camera-ready experiments
│   ├── partition.py         # Dirichlet + two-level (edge/client) Dirichlet
│   ├── synthesis.py         # Prototype-synthesis covariance modes + comm accounting
│   ├── feature_distance.py  # RBF-MMD
│   ├── lstat.py             # Empirical Lipschitz sensitivity
│   ├── profiling.py         # CUDA-synchronized phase timers
│   ├── inversion.py         # Feature-inversion exposure analysis
│   ├── latex.py / io_utils.py
├── serverless/              # Serverless backend interfaces (future deployment)
├── scripts/
│   ├── journal_experiments/ # E-HSFP journal experiment launchers
│   └── camera_ready/        # Camera-ready experiment launchers + table/figure generators
├── tools/                   # Standalone plotting scripts
└── docs/                    # This documentation
```

## How `main.py` dispatches

`main.py` holds a `REGISTRY` mapping `task → method → directory`. It uses
`importlib` to load `runner.py` from the selected method directory, temporarily
adding that directory to `sys.path` so the method's relative imports
(`from hierarchy import ...`, `from data import get_dataset`, ...) resolve. After
the run the directory is removed and cached modules are cleared.

`--ablation`, `--seed`, and `--set KEY=VALUE` overrides are injected into a
temporary copy of the YAML config before the runner is invoked, so the on-disk
configs stay clean and the base pipeline stays reproducible.

## Notes for readers

- Each method directory is self-contained (its own `runner.py`, `hierarchy.py`/
  `models/`, `data/`, `config/`). This duplication is intentional: it keeps each
  baseline independently runnable and easy to read in isolation.
- New experiment functionality is added behind config flags / CLI overrides rather
  than by changing default behavior.
