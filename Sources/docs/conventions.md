# Code & Config Conventions

## Adding a new method

1. Create `classification/<MethodName>/` or `segmentation/<MethodName>/`
2. Add `runner.py` with a single `run(cfg_path: str)` function — no argparse
3. Follow the same sub-structure: `config/` (ConfigLoader only), `data/`, `models/`
4. Add configs to `configs/<task>/<method-name>/` (including a `default.yaml`)
5. Register in `main.py` → `REGISTRY` dict

## Adding a new dataset

1. Add loader in `<method>/data/utils/<dataset>.py`
2. Update `<method>/data/get_data.py` to handle the dataset name
3. Add YAML configs in `configs/<task>/<method>/`
4. Update `<method>/models/__init__.py` if new architectures are needed

## Config file format

All configs inherit from `default.yaml` in the same directory via `base: default.yaml`.

```yaml
base: default.yaml          # inherit defaults

strategy: "hier_fedavg"     # algorithm identifier
dataset:  "ham10000"        # cifar10 | cifar100 | ham10000 | isic-2018
model:    "vgg"             # alexnet | resnet50 | vgg
iid:      false             # data distribution
num_users: 200              # 200 for classification, 50 for segmentation
epochs:   200
t1: 5                       # H-SFP aggregation interval tier-1
t2: 10                      # H-SFP aggregation interval tier-2
```

### Unified base params (fair comparison)

All methods share the same default hyperparameters per task:

| Param         | Classification | Segmentation |
|---------------|---------------|--------------|
| `num_users`   | 200           | 50           |
| `epochs`      | 200           | 200          |
| `frac`        | 0.1           | 0.1          |
| `local_bs`    | 16            | 16           |
| `local_ep`    | 5             | 5            |
| `optimizer`   | adam          | adam         |
| `lr`          | 1e-4          | 1e-4         |
| `iid`         | true          | true         |

**Note:** HAM10000 experiment configs override `local_bs: 4` for GPU memory.

**Config naming:** `{dataset}_{strategy}_{model}_{t1}_{t2}.yaml` for H-SFP;
`{dataset}_{strategy}_{model}.yaml` for baselines.

## Output / results

Each runner saves a JSON file to its own `Figure/data/` folder.
The output path is relative to the method directory (not the project root).

## Wandb integration pattern

All runners use this pattern to log to wandb without a hard dependency:

```python
try:
    import wandb
except ImportError:
    wandb = None

# In training loop:
if wandb is not None and wandb.run is not None:
    wandb.log({"epoch": epoch, "train_loss": loss, ...})

# After training:
if wandb is not None and wandb.run is not None:
    wandb.summary["test_f1"] = test_f1
```

`wandb.init()` / `wandb.finish()` are handled in `main.py`, **not** in runners.
Runners only call `wandb.log()` and set `wandb.summary` — they never init or finish.

## Known tech debt

- JSON output paths use relative paths from the method dir — this is correct.
- Some segmentation runners still use `f1_score` instead of IoU/Dice as primary metric;
  fix if publishing segmentation results.
