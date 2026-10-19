#!/usr/bin/env bash
set -euo pipefail

# PLAN-15 missing runs only. Run from the repository root on the host GPU.
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

# H-SFP is already complete for all three seeds. Any extension must retain:
# HSFP_FREEZE_BACKBONE=1 python main.py ... --ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0

python tools/fair_compare/aggregate.py outputs/proxy/plan2_cifar100_splitfl_seed20260714 outputs/proxy/plan15_splitfl_seed20260715 outputs/proxy/plan15_splitfl_seed20260716 outputs/proxy/plan2_cifar100_federated_seed20260714 outputs/proxy/plan15_federated_seed20260715 outputs/proxy/plan15_federated_seed20260716 outputs/proxy/plan11_freeze_centered_seed20260714 outputs/proxy/plan14_freeze_centered_s20260715 outputs/proxy/plan14_freeze_centered_s20260716 outputs/proxy/plan15_hierfl_seed20260714 outputs/proxy/plan15_hierfl_seed20260715 outputs/proxy/plan15_hierfl_seed20260716 outputs/proxy/plan15_hsfl_seed20260714 outputs/proxy/plan15_hsfl_seed20260715 outputs/proxy/plan15_hsfl_seed20260716 outputs/proxy/plan15_heterosfl_seed20260714 outputs/proxy/plan15_heterosfl_seed20260715 outputs/proxy/plan15_heterosfl_seed20260716
