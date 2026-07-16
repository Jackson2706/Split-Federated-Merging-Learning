#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the host GPU. No command in this file was run here.
for seed in 20260714 20260715 20260716; do
  HSFP_FREEZE_BACKBONE=1 python main.py --task classification --method h-sfp --cfg configs/proxy/plan_ham10000_hsfp.yaml --seed "$seed" --ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0 --set output_dir="outputs/proxy/plan_ham10000_hsfp_seed${seed}"
  python main.py --task classification --method splitfl --cfg configs/proxy/plan_ham10000_splitfl.yaml --seed "$seed" --set output_dir="outputs/proxy/plan_ham10000_splitfl_seed${seed}"
  python main.py --task classification --method federated --cfg configs/proxy/plan_ham10000_federated.yaml --seed "$seed" --set output_dir="outputs/proxy/plan_ham10000_federated_seed${seed}"
  python main.py --task classification --method hierfl --cfg configs/proxy/plan_ham10000_hierfl.yaml --seed "$seed" --set output_dir="outputs/proxy/plan_ham10000_hierfl_seed${seed}"
  python main.py --task classification --method hsfl --cfg configs/proxy/plan_ham10000_hsfl.yaml --seed "$seed" --set output_dir="outputs/proxy/plan_ham10000_hsfl_seed${seed}"
  python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan_ham10000_heterosfl.yaml --seed "$seed" --set output_dir="outputs/proxy/plan_ham10000_heterosfl_seed${seed}"
done

python tools/fair_compare/aggregate.py --dataset ham10000 outputs/proxy/plan_ham10000_*
