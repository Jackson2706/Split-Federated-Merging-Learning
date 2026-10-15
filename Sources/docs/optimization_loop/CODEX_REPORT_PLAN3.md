## EXECUTION REPORT

Implemented [oracle_probe.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/diag/oracle_probe.py). It writes `diag_results.json`, `SUMMARY.md`, and a checkpoint-bound real-feature cache.

Exact host-GPU command, from repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/diag/oracle_probe.py \
  --config configs/proxy/plan2_cifar100_hsfp_baseline.yaml \
  --checkpoint outputs/proxy/plan2_cifar100_hsfp_baseline_seed20260714/e4b67d7d5c12/817f3265d5bec70c72df50cb009c9de5770c8cc660e00418c49d30ca77a7555b/checkpoint.pt \
  --output-dir outputs/diag/plan3_hsfp_baseline_oracle_seed20260714 \
  --device cuda \
  --seed 20260714 \
  --batch-size 256 \
  --workers 2 \
  --epochs 5 \
  --linear-epochs 10 \
  --samples-per-class 50
```

CPU smoke passed using the real epoch-1 baseline checkpoint:

- Deterministic repeated results
- Finite metrics for all four feature sources
- L1 shapes `[32,128]` / `[16,128]`
- L2 shapes `[32,256]` / `[16,256]`
- Note: configured L1 is 128-D, despite one contradictory 256-D line in PLAN-3; this matches `PROJECT_STATE.md` and the checkpoint.

Recorded Phase-1 proxies in [EXPERIMENT_REGISTRY.csv](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/EXPERIMENT_REGISTRY.csv) and [RESULTS_SUMMARY.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/RESULTS_SUMMARY.md): SplitFL 44.57%, Federated 34.76%, H-SFP baseline 3.81%, all flagged test-as-val.

`full_e_hsfp` remains pending: no final `metrics.json` exists, so no final number was recorded. Its output directory was untouched.

The deferred held-out validation protocol is drafted in [DECISIONS.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/DECISIONS.md).