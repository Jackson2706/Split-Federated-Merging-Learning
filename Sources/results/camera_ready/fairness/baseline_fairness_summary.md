# Baseline Fairness & Reproducibility Summary

Common protocol applied to **all** methods (H-SFP and baselines) for the
CIFAR-100 camera-ready experiments. Source: `configs/camera_ready/baseline_fairness.yaml`.

| Hyperparameter | Value |
|---|---|
| `dataset` | cifar100 |
| `num_classes` | 100 |
| `dataset_root` | data/cifar/ |
| `num_clients` | 200 |
| `num_edges` | 5 |
| `client_sampling_ratio` | 0.1 |
| `local_epochs` | 5 |
| `global_rounds` | 200 |
| `aggregation_intervals.t1` | 5 |
| `aggregation_intervals.t2` | 10 |
| `optimizer` | adam |
| `learning_rate` | 0.0001 |
| `lr_search_space` | 0.001, 0.0005, 0.0001, 5e-05 |
| `batch_size` | 16 |
| `weight_decay` | 0.0001 |
| `momentum` | 0.9 |
| `model` | resnet50 |
| `split_points.client` | stem + layer1 |
| `split_points.edge` | layer2 - layer4 + GAP |
| `split_points.cloud` | linear classifier head |
| `synthesis_mode` | diag_sigma |
| `syn_samples_per_class` | 50 |
| `ssl_epochs_client` | 10 |
| `ssl_epochs_edge` | 10 |
| `syn_epochs_cloud` | 10 |
| `seeds` | 0, 1, 2, 3, 4 |
| `determinism` | random/numpy/torch seeded; cudnn.deterministic=True, benchmark=False (main._set_seed) |
| `hardware` | 1x NVIDIA GeForce RTX 3080 Ti (12 GB), CUDA 12.8, PyTorch 2.7 |
| `comm_accounting.prototype_payload_MB` | sum_t numel(t)*element_size(t) / 1024^2 over (proto_dict, dist_dict) |
| `comm_accounting.model_payload_MB` | sum_p numel(p)*4 / 1e6 over state_dict (float32) |
| `comm_accounting.covariance_modes` | per-class float counts: mu_only=D, diag=2D, full=D+D(D+1)/2, low_rank=D(r+2), mixture=K(2D)+K |
| `methods` | h-sfp, hsfl, splitfl, hierfl, federated |
| `federated_strategies` | fedavg, fedprox, fednova, fedsgd |
| `sanity_checks.centralized_oracle` | single-node supervised upper bound on CIFAR-100 ResNet50 |
| `sanity_checks.tuned_fedavg` | FedAvg with LR swept over lr_search_space, IID, 200 clients |

## Sanity checks
- **Centralized oracle** (upper bound) and **tuned FedAvg** commands are in `results/camera_ready/README.md`.

## Notes
- All methods share dataset, client/edge counts, sampling ratio, local/global schedule, optimizer, batch size, and seeds.
- Communication is accounted with identical formulas across methods (see `comm_accounting`).
