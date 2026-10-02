#!/usr/bin/env bash
set -euo pipefail

# ISIC-2018 segmentation grid. SplitFL and HSFL are omitted because their
# segmentation model registries do not implement the isic-2018 dataset.
python main.py --task segmentation --method federated --cfg configs/proxy/plan_isic2018_federated.yaml --seed 20260714 --set output_dir=outputs/isic2018/federated_seed20260714
python main.py --task segmentation --method federated --cfg configs/proxy/plan_isic2018_federated.yaml --seed 20260715 --set output_dir=outputs/isic2018/federated_seed20260715
python main.py --task segmentation --method federated --cfg configs/proxy/plan_isic2018_federated.yaml --seed 20260716 --set output_dir=outputs/isic2018/federated_seed20260716
python main.py --task segmentation --method hierfl --cfg configs/proxy/plan_isic2018_hierfl.yaml --seed 20260714 --set output_dir=outputs/isic2018/hierfl_seed20260714
python main.py --task segmentation --method hierfl --cfg configs/proxy/plan_isic2018_hierfl.yaml --seed 20260715 --set output_dir=outputs/isic2018/hierfl_seed20260715
python main.py --task segmentation --method hierfl --cfg configs/proxy/plan_isic2018_hierfl.yaml --seed 20260716 --set output_dir=outputs/isic2018/hierfl_seed20260716
python main.py --task segmentation --method hetero-sfl --cfg configs/proxy/plan_isic2018_heterosfl.yaml --seed 20260714 --set output_dir=outputs/isic2018/heterosfl_seed20260714
python main.py --task segmentation --method hetero-sfl --cfg configs/proxy/plan_isic2018_heterosfl.yaml --seed 20260715 --set output_dir=outputs/isic2018/heterosfl_seed20260715
python main.py --task segmentation --method hetero-sfl --cfg configs/proxy/plan_isic2018_heterosfl.yaml --seed 20260716 --set output_dir=outputs/isic2018/heterosfl_seed20260716
python main.py --task segmentation --method h-sfp --cfg configs/proxy/plan_isic2018_hsfp.yaml --ablation baseline_hsfp --seed 20260714 --set output_dir=outputs/isic2018/hsfp_seed20260714
python main.py --task segmentation --method h-sfp --cfg configs/proxy/plan_isic2018_hsfp.yaml --ablation baseline_hsfp --seed 20260715 --set output_dir=outputs/isic2018/hsfp_seed20260715
python main.py --task segmentation --method h-sfp --cfg configs/proxy/plan_isic2018_hsfp.yaml --ablation baseline_hsfp --seed 20260716 --set output_dir=outputs/isic2018/hsfp_seed20260716

python tools/fair_compare/aggregate.py --dataset isic2018 outputs/isic2018/*_seed*
