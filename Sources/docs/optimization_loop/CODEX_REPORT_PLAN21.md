Implemented locally without touching existing uncommitted changes.

The FULL-60 runner is [run_full60_cifar100.sh](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/run_full60_cifar100.sh). It contains exactly 18 training commands: 6 methods × 3 seeds. Its final aggregation command writes `results/fair_comparison_cifar100_full60.csv`.

```bash
bash tools/fair_compare/run_full60_cifar100.sh
```

ISIC verdict: data verified successfully.

- 2,594 usable `.jpg`/`.png` training pairs
- 100 usable validation pairs
- All four proxy configs resolve to existing directories
- Available ISIC methods: Federated, HierFL, HeteroSFL, H-SFP
- SplitFL and HSFL were excluded because their model registries do not implement `isic-2018`
- Matching partition hashes across all four methods:

```text
20260714  092c1cb24abeb85519ed23564139c7dc358a65c274a69285d945d88f4fdb5cf9
20260715  5255ce760a8fe6994a3a2b29e654e2a1298acc269f89831220a1362ccfa51a8c
20260716  77ea9f46f2205d39d01c3d726f2e804daff2c06f8ccdb8564ecda8898446a463
```

The exact 12 host-GPU launches and aggregate command are in [run_grid_isic2018.sh](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/run_grid_isic2018.sh):

```bash
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
```

Both runners pass `bash -n`; the aggregator passes Python compilation and now supports `cifar100_full60` and `isic2018`.