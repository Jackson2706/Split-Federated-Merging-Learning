#!/usr/bin/env bash
set -euo pipefail

# FULL-60 CIFAR-100 fair grid. Run from the repository root on the host GPU.
python main.py --task classification --method splitfl --cfg configs/proxy/plan2_cifar100_splitfl.yaml --seed 20260714 --set epochs=60 --set output_dir=outputs/full60/splitfl_seed20260714
python main.py --task classification --method splitfl --cfg configs/proxy/plan2_cifar100_splitfl.yaml --seed 20260715 --set epochs=60 --set output_dir=outputs/full60/splitfl_seed20260715
python main.py --task classification --method splitfl --cfg configs/proxy/plan2_cifar100_splitfl.yaml --seed 20260716 --set epochs=60 --set output_dir=outputs/full60/splitfl_seed20260716
python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260714 --set epochs=60 --set output_dir=outputs/full60/federated_seed20260714
python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260715 --set epochs=60 --set output_dir=outputs/full60/federated_seed20260715
python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260716 --set epochs=60 --set output_dir=outputs/full60/federated_seed20260716
HSFP_FREEZE_BACKBONE=1 python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 --set prototype_space=centered_cosine --set num_workers=0 --set epochs=60 --set output_dir=outputs/full60/hsfp_seed20260714
HSFP_FREEZE_BACKBONE=1 python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260715 --set prototype_space=centered_cosine --set num_workers=0 --set epochs=60 --set output_dir=outputs/full60/hsfp_seed20260715
HSFP_FREEZE_BACKBONE=1 python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260716 --set prototype_space=centered_cosine --set num_workers=0 --set epochs=60 --set output_dir=outputs/full60/hsfp_seed20260716
python main.py --task classification --method hierfl --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260714 --set epochs=60 --set output_dir=outputs/full60/hierfl_seed20260714
python main.py --task classification --method hierfl --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260715 --set epochs=60 --set output_dir=outputs/full60/hierfl_seed20260715
python main.py --task classification --method hierfl --cfg configs/proxy/plan15_cifar100_hierfl.yaml --seed 20260716 --set epochs=60 --set output_dir=outputs/full60/hierfl_seed20260716
python main.py --task classification --method hsfl --cfg configs/proxy/plan15_cifar100_hsfl.yaml --seed 20260714 --set epochs=60 --set output_dir=outputs/full60/hsfl_seed20260714
python main.py --task classification --method hsfl --cfg configs/proxy/plan15_cifar100_hsfl.yaml --seed 20260715 --set epochs=60 --set output_dir=outputs/full60/hsfl_seed20260715
python main.py --task classification --method hsfl --cfg configs/proxy/plan15_cifar100_hsfl.yaml --seed 20260716 --set epochs=60 --set output_dir=outputs/full60/hsfl_seed20260716
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan15_cifar100_heterosfl.yaml --seed 20260714 --set epochs=60 --set output_dir=outputs/full60/heterosfl_seed20260714
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan15_cifar100_heterosfl.yaml --seed 20260715 --set epochs=60 --set output_dir=outputs/full60/heterosfl_seed20260715
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan15_cifar100_heterosfl.yaml --seed 20260716 --set epochs=60 --set output_dir=outputs/full60/heterosfl_seed20260716

python tools/fair_compare/aggregate.py --dataset cifar100_full60 outputs/full60/*_seed*
