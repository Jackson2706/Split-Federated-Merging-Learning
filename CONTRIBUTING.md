# Contributing

Thanks for your interest in improving H-SFP / E-HSFP!

## Development setup

```bash
git clone https://github.com/Jackson2706/Split-Federated-Merging-Learning.git
cd Split-Federated-Merging-Learning
python -m venv .venv && source .venv/bin/activate
# Install a CUDA build of torch first if you have a GPU:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .          # installs ehsfp / camera_ready / serverless as packages
```

## Project layout

- `main.py` — unified entry point; dispatches `--task {classification,segmentation}`
  and `--method {h-sfp,federated,hierfl,splitfl,hetero-sfl,hsfl}` to the matching
  `runner.py`.
- `classification/`, `segmentation/` — per-method implementations.
- `ehsfp/` — E-HSFP shared library (episodic memory, reliability, dropout, losses).
- `camera_ready/` — reusable utilities for the camera-ready experiments.
- `configs/` — YAML configs with `base:` inheritance.
- `scripts/` — experiment launchers (`journal_experiments/`, `camera_ready/`).
- `docs/` — architecture and design notes.

## Conventions

- Add new behavior **behind config flags / CLI overrides** (`--set key=value`) so the
  default H-SFP pipeline stays reproducible.
- Keep new code consistent with the surrounding style; prefer reusing the shared
  utilities in `ehsfp/` and `camera_ready/`.
- Comments and identifiers in English.
- Do not commit datasets, checkpoints (`*.pt`), logs, `wandb/`, or generated outputs
  (see `.gitignore`).

## Before opening a PR

```bash
python -m py_compile $(git ls-files '*.py')      # syntax
python main.py --list                            # registry sanity
python -m camera_ready.partition                 # partition self-test
```

Please describe what you changed, how you tested it, and (for new experiments) the
exact command to reproduce.
