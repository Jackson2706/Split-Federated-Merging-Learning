## EXECUTION REPORT

PLAN-15 build completed locally. No GPU training was launched.

### Configs and verification

Created:

- [plan15_cifar100_hierfl.yaml](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan15_cifar100_hierfl.yaml)
- [plan15_cifar100_hsfl.yaml](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan15_cifar100_hsfl.yaml)
- [plan15_cifar100_heterosfl.yaml](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan15_cifar100_heterosfl.yaml)

All resolve to CIFAR-100, IID, seed `20260714`, 200 clients, 20 selected, 10 rounds, and preserve each baseline’s five local epochs and optimizer.

Recomputed partition hashes using each method’s actual sampler:

```text
HierFL:    93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1 MATCH
HSFL:      93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1 MATCH
HeteroSFL: 93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1 MATCH
```

The audit matrix was extended in [FAIR_COMPARISON_CIFAR100.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/FAIR_COMPARISON_CIFAR100.md).

### Aggregation harness

Implemented [aggregate.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/aggregate.py), with append-safe deduplication on `method+seed+config`.

Populated [fair_comparison_cifar100.csv](/home/jackson/Desktop/Split-Federated-Merging-Learning/results/fair_comparison_cifar100.csv):

| method | seed | best_val_top1 (%) | comm MB | peak VRAM MB | status |
|---|---:|---:|---:|---:|---|
| federated | 20260714 | 34.76 | — | — | complete, legacy metrics missing |
| h-sfp | 20260714 | 9.76 | 122.1332 | 445.0156 | complete, runtime missing |
| h-sfp | 20260715 | 10.55 | 122.0502 | 418.8340 | complete, runtime missing |
| h-sfp | 20260716 | 10.56 | 121.9770 | 437.1611 | complete, runtime missing |
| splitfl | 20260714 | 44.57 | — | — | complete, legacy metrics missing |

The legacy SplitFL/Federated artifacts predate communication, runtime, and peak-VRAM persistence; these fields remain blank and are explicitly reflected in `status`. Future baseline outputs now persist these measurements and honor unique `output_dir` values.

Validation passed: Python compilation, shell syntax, config resolution, partition hash checks, aggregator execution, and `git diff --check`.

### Exact missing-run commands

Also saved in [run_grid.sh](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/run_grid.sh).

```bash
python main.py --task classification --method splitfl --cfg configs/proxy/plan2_cifar100_splitfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan15_splitfl_seed20260715
python main.py --task classification --method splitfl --cfg configs/proxy/plan2_cifar100_splitfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan15_splitfl_seed20260716
python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260715 --set output_dir=outputs/proxy/plan15_federated_seed20260715
python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260716 --set output_dir=outputs/proxy/plan15_federated_seed20260716
python main.py --task classification --method hierfl --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260714 --set output_dir=outputs/proxy/plan15_hierfl_seed20260714
python main.py --task classification --method hierfl --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan15_hierfl_seed20260715
python main.py --task classification --method hierfl --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan15_hierfl_seed20260716
python main.py --task classification --method hsfl --cfg configs/proxy/plan15_cifar100_hsfl.yaml --seed 20260714 --set output_dir=outputs/proxy/plan15_hsfl_seed20260714
python main.py --task classification --method hsfl --cfg configs/proxy/plan15_cifar100_hsfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan15_hsfl_seed20260715
python main.py --task classification --method hsfl --cfg configs/proxy/plan15_cifar100_hsfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan15_hsfl_seed20260716
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan15_cifar100_heterosfl.yaml --seed 20260714 --set output_dir=outputs/proxy/plan15_heterosfl_seed20260714
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan15_cifar100_heterosfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan15_heterosfl_seed20260715
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan15_cifar100_heterosfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan15_heterosfl_seed20260716
```

H-SFP is already complete. Any extension must use:

```bash
HSFP_FREEZE_BACKBONE=1 python main.py ... --ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0
```