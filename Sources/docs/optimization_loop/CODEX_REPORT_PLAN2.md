## EXECUTION REPORT

- PLAN-1d recorded as `smoke-pass` for both presets in [EXPERIMENT_REGISTRY.csv](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/EXPERIMENT_REGISTRY.csv) and [DECISIONS.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/DECISIONS.md).
- Fairness audit: [FAIR_COMPARISON_CIFAR100.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/FAIR_COMPARISON_CIFAR100.md).
- No GPU, training, network, default-config changes, or destructive git operations performed.

Fair-match verdict: conditional match for partition/seed and metric, but overall **FLAGGED**.

- All methods produce identical IID partitions with explicit seed `20260714`: `partition_hash=93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1`.
- All primary metrics compute top-1 accuracy, despite misleading `f1` variable names.
- Fairness mismatches remain: official test split reused as validation; different evaluation paths/frequencies, local work, optimizers/schedulers, and effective architectures.
- These proxies cannot establish `B_d`.

Proxy configs and resolved hashes:

- [SplitFL](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan2_cifar100_splitfl.yaml): `cb01b9aa...c21cc`
- [Federated](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan2_cifar100_federated.yaml): `da304a30...7f2c1`
- [H-SFP baseline](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan2_cifar100_hsfp_baseline.yaml): `817f3265...7555b`
- [Full E-HSFP](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan2_cifar100_hsfp_full_e.yaml): `0018e9c9...43bb`

Legacy SplitFL/Federated result collisions were detected. Optional `output_dir` isolation was therefore wired into their runners; H-SFP uses each declared directory as its hash-addressed root. Existing/default behavior is unchanged when `output_dir` is absent.

Exact sequential launch commands:

```bash
python main.py --task classification --method splitfl --cfg configs/proxy/plan2_cifar100_splitfl.yaml --seed 20260714
python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260714
python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714
python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_full_e.yaml --ablation full_e_hsfp --seed 20260714
```

Verification: four configs resolved, hashes/output roots are distinct and collision-free, runner syntax passed, registry CSV passed structural validation, and 9/9 Phase-0 unit tests passed. SplitFL/Federated still do not persist their resolved hashes in result JSON.