Implemented locally without running GPU/training and without touching existing uncommitted changes.

Config files:

- [H-SFP](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_ham10000_hsfp.yaml)
- [SplitFL](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_ham10000_splitfl.yaml)
- [Federated](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_ham10000_federated.yaml)
- [HierFL](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_ham10000_hierfl.yaml)
- [HSFL](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_ham10000_hsfl.yaml)
- [Hetero-SFL](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_ham10000_heterosfl.yaml)

Each resolves to HAM10000, ResNet-50, 10 rounds, IID partitioning, 200 users, 10% participation, and evaluation metrics `[top1, macro_f1]`.

Partition verification:

- Seed `20260714`: `09e04bfb8e2c74fc3c58f9570a57581b3f3e194873d849b43d21f01565eedbd2`
- Seed `20260715`: `d234dddfa64732204c275404e58ab538039d2562a860db09815fced70c6bf5db`
- Seed `20260716`: `a5e5ddd8aad48ce2a46364bf46b44c72852b7a0769cbdb173dccd6518b907dd5`

For each seed, all six methods produce the identical hash using the repository’s HAM10000 IID partition algorithm.

The updated [aggregate.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/aggregate.py) supports `--dataset ham10000`, defaults to `results/fair_comparison_ham10000.csv`, and captures:

`best_val_top1`, `macro_f1` when available, `total_comm_MB`, `peak_vram_MB`, `runtime_s`, and partition/config metadata.

Exact launch commands are also recorded in [run_grid_ham10000.sh](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/run_grid_ham10000.sh).

```bash
HSFP_FREEZE_BACKBONE=1 python main.py --task classification --method h-sfp --cfg configs/proxy/plan_ham10000_hsfp.yaml --seed 20260714 --ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0 --set output_dir=outputs/proxy/plan_ham10000_hsfp_seed20260714
python main.py --task classification --method splitfl --cfg configs/proxy/plan_ham10000_splitfl.yaml --seed 20260714 --set output_dir=outputs/proxy/plan_ham10000_splitfl_seed20260714
python main.py --task classification --method federated --cfg configs/proxy/plan_ham10000_federated.yaml --seed 20260714 --set output_dir=outputs/proxy/plan_ham10000_federated_seed20260714
python main.py --task classification --method hierfl --cfg configs/proxy/plan_ham10000_hierfl.yaml --seed 20260714 --set output_dir=outputs/proxy/plan_ham10000_hierfl_seed20260714
python main.py --task classification --method hsfl --cfg configs/proxy/plan_ham10000_hsfl.yaml --seed 20260714 --set output_dir=outputs/proxy/plan_ham10000_hsfl_seed20260714
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan_ham10000_heterosfl.yaml --seed 20260714 --set output_dir=outputs/proxy/plan_ham10000_heterosfl_seed20260714

HSFP_FREEZE_BACKBONE=1 python main.py --task classification --method h-sfp --cfg configs/proxy/plan_ham10000_hsfp.yaml --seed 20260715 --ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0 --set output_dir=outputs/proxy/plan_ham10000_hsfp_seed20260715
python main.py --task classification --method splitfl --cfg configs/proxy/plan_ham10000_splitfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan_ham10000_splitfl_seed20260715
python main.py --task classification --method federated --cfg configs/proxy/plan_ham10000_federated.yaml --seed 20260715 --set output_dir=outputs/proxy/plan_ham10000_federated_seed20260715
python main.py --task classification --method hierfl --cfg configs/proxy/plan_ham10000_hierfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan_ham10000_hierfl_seed20260715
python main.py --task classification --method hsfl --cfg configs/proxy/plan_ham10000_hsfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan_ham10000_hsfl_seed20260715
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan_ham10000_heterosfl.yaml --seed 20260715 --set output_dir=outputs/proxy/plan_ham10000_heterosfl_seed20260715

HSFP_FREEZE_BACKBONE=1 python main.py --task classification --method h-sfp --cfg configs/proxy/plan_ham10000_hsfp.yaml --seed 20260716 --ablation baseline_hsfp --set prototype_space=centered_cosine --set num_workers=0 --set output_dir=outputs/proxy/plan_ham10000_hsfp_seed20260716
python main.py --task classification --method splitfl --cfg configs/proxy/plan_ham10000_splitfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan_ham10000_splitfl_seed20260716
python main.py --task classification --method federated --cfg configs/proxy/plan_ham10000_federated.yaml --seed 20260716 --set output_dir=outputs/proxy/plan_ham10000_federated_seed20260716
python main.py --task classification --method hierfl --cfg configs/proxy/plan_ham10000_hierfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan_ham10000_hierfl_seed20260716
python main.py --task classification --method hsfl --cfg configs/proxy/plan_ham10000_hsfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan_ham10000_hsfl_seed20260716
python main.py --task classification --method hetero-sfl --cfg configs/proxy/plan_ham10000_heterosfl.yaml --seed 20260716 --set output_dir=outputs/proxy/plan_ham10000_heterosfl_seed20260716
```

Afterward, aggregate with:

```bash
python tools/fair_compare/aggregate.py --dataset ham10000 outputs/proxy/plan_ham10000_*
```

Data availability: **HAM10000 is present locally** at the configured paths:

- `data/HAM10000/HAM10000_metadata.csv` — 10,015 records
- `data/HAM10000/HAM10000_images_part_1` — 5,000 files
- `data/HAM10000/HAM10000_images_part_2` — 5,015 files