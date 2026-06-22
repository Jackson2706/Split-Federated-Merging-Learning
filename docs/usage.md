# Running Experiments

Always use the root `main.py`. Never call method runners directly.

```bash
# Show available methods
python main.py --list

# Show available configs (filter with --task / --method)
python main.py --list-configs --task classification --method h-sfp

# Run primary method (H-SFP)
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/ham10000_our_vgg_5_10.yaml

# Run a baseline
python main.py --task classification --method federated \
    --cfg configs/classification/federated/cifar_fedavg_resnet50.yaml

# Run segmentation primary method (H-SFP)
python main.py --task segmentation --method h-sfp \
    --cfg configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml

# Run segmentation baseline
python main.py --task segmentation --method hierfl \
    --cfg configs/segmentation/hierfl/isic_hierfl_resnet50.yaml

# Enable W&B logging (requires: pip install wandb && wandb login)
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/ham10000_our_vgg_5_10.yaml \
    --wandb --wandb-project H-SFP

# Run E-HSFP with full features
python main.py --task classification --method h-sfp \
    --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
    --ablation full_e_hsfp

# Run E-HSFP ablation (memory + reliability + PRC)
python main.py --task segmentation --method h-sfp \
    --cfg configs/segmentation/h-sfp/isic_our_resnet50_5_10.yaml \
    --ablation hsfp_memory_reliability_prc
```

## Batch runs (`run.sh`)

```bash
./run.sh                                       # all experiments
./run.sh classification                        # classification only
./run.sh segmentation                          # segmentation only
./run.sh --wandb                               # all experiments with W&B
./run.sh classification --wandb                # classification + W&B
./run.sh --wandb --wandb-project=MyProject     # custom W&B project
./run.sh --wandb --wandb-entity=MyTeam         # custom W&B entity
./run.sh ablation                              # E-HSFP ablation experiments only
```

Logs are saved to `logs/<timestamp>/` with format `<task>_<dataset>_<method>_<name>.log`.

## Method names (`--method`)

| Value         | Description                                   | Tasks                        |
|---------------|-----------------------------------------------|------------------------------|
| `h-sfp`       | **PRIMARY** — H-SFP (prototype-based)         | classification, segmentation |
| `federated`   | FedAvg / FedNova / FedProx / FedSGD           | classification, segmentation |
| `hierfl`      | Hierarchical Federated Learning               | classification, segmentation |
| `splitfl`     | Split Federated Learning                      | classification, segmentation |
| `hetero-sfl`  | Heterogeneous Split FL                        | classification, segmentation |
| `hsfl`        | Hierarchical Split FL                         | classification, segmentation |

## Config path convention

All configs live under `configs/<task>/<method>/`. Pass paths relative to the project root:
```
configs/classification/h-sfp/ham10000_our_vgg_5_10.yaml
configs/segmentation/federated/isic_fedavg_resnet50.yaml
```

## Weights & Biases (wandb)

All runners log per-epoch metrics to wandb when `--wandb` is passed to `main.py`.
wandb is initialized in `main.py` and auto-detected by runners via `wandb.run`.

| Flag               | Default   | Description                    |
|--------------------|-----------|--------------------------------|
| `--wandb`          | off       | Enable W&B logging             |
| `--wandb-project`  | `H-SFP`  | W&B project name               |
| `--wandb-entity`   | `None`    | W&B team/entity                |

**Metrics logged per epoch** (varies by method):
- Classification: `train_loss`, `f1`, `best_f1`, client resource usage, comm costs
- Segmentation: `train_loss`, `iou`, `dice`, client resource usage, comm costs
- H-SFP/HSFL hierarchy: `cloud_loss`, `validation_f1`/`validation_iou`/`validation_dice`, `best_f1`, comm tracker
- Final summary: `test_f1` (or `test_iou`/`test_dice`), `total_time_s`
