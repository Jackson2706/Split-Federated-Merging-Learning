## EXECUTION REPORT

Implemented PLAN-6 locally.

- Added `client_objective={ssl,ssl_supcon,supervised}` in [client_objective.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/client_objective.py:13).
- Integrated client-local objectives into [hierarchy.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/hierarchy.py:693).
- `ssl` remains the default and its actual Adam client step is bitwise identical to the legacy formula.
- `ssl_supcon`: existing SSL plus weighted one-view local SupCon.
- `supervised`: existing SSL plus weighted per-client 100-way linear CE head.
- Default auxiliary weight: `0.1`; commands record it explicitly.
- No cross-tier activations, samples, labels, or gradients were added. Prototype transport, memory, reliability, PRC, dropout, and `prototype_space` are unchanged.
- No default/baseline YAML was modified.

Added [encoder_linear_probe.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/diag/encoder_linear_probe.py:1), reusing the oracle probe’s feature extraction and standardized linear-head training. It reports `client_encoder_linear_probe_top1` from checkpointed real client features.

Tests:

- PLAN-6: 7/7 passed.
- Full existing suite: 22/22 passed.
- `ssl` loss identity and full client Adam step: bitwise identical.
- SupCon and supervised losses: finite, nonzero, with nonzero encoder gradients.
- Client-local API and distinct objective hashes verified.
- Probe CLI/import/compilation and `git diff --check`: passed.
- No GPU or proxy training executed.

Exact host-GPU commands, from repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 --set epochs=10 --set client_objective=ssl --set client_objective_weight=0.1 --set proxy_id=plan6_cifar100_hsfp_ssl_seed20260714 --set output_dir=outputs/proxy/plan6_cifar100_hsfp_ssl_seed20260714
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 --set epochs=10 --set client_objective=supervised --set client_objective_weight=0.1 --set proxy_id=plan6_cifar100_hsfp_supervised_seed20260714 --set output_dir=outputs/proxy/plan6_cifar100_hsfp_supervised_seed20260714
```

Control linear probe:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/diag/encoder_linear_probe.py --config configs/proxy/plan2_cifar100_hsfp_baseline.yaml --checkpoint outputs/proxy/plan6_cifar100_hsfp_ssl_seed20260714/e394190834d8/c78dedddbcd5bb56081502955fa93b8a295e3c4edb9e606eb874c6f7c35d3c2b/checkpoint.pt --output-dir outputs/diag/plan6_hsfp_ssl_encoder_probe_seed20260714 --seed 20260714 --device cuda --linear-epochs 10
```

Supervised linear probe:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/diag/encoder_linear_probe.py --config configs/proxy/plan2_cifar100_hsfp_baseline.yaml --checkpoint outputs/proxy/plan6_cifar100_hsfp_supervised_seed20260714/e394190834d8/861bdb361fde223d3cd02d587ee3f2249ef7157665a33b02ec6685825e603131/checkpoint.pt --output-dir outputs/diag/plan6_hsfp_supervised_encoder_probe_seed20260714 --seed 20260714 --device cuda --linear-epochs 10
```

Current resolved hashes are `c78deddd…d3c2b` for control and `861bdb36…03131` for supervised; both use architecture `e394190834d8`.

Comms/compute:

- Communication delta: exactly zero. Local heads are neither transmitted nor aggregated.
- `ssl`: zero compute/model-state delta.
- `supervised`: one 128×100 linear head per client—12,900 parameters and about 0.41M forward MACs per batch of 16 across two views. At 200 clients, maximum persistent head weights are about 10.3 MB; roughly 41 MB including gradients and Adam state.
- No extra encoder forward pass is introduced.
- Local heads are intentionally absent from the pipeline checkpoint; the trained encoder is checkpointed.

Recorded PLAN-5’s neutral `centered_cosine` result—3.48% versus raw 3.81%—and PLAN-6 in [EXPERIMENT_REGISTRY.csv](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/EXPERIMENT_REGISTRY.csv:14), [RESULTS_SUMMARY.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/RESULTS_SUMMARY.md:68), and [DECISIONS.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/DECISIONS.md:48).

Unresolved risks: the existing “SSL” is already a two-view supervised-contrastive loss using local labels, so `ssl_supcon` is incremental rather than first-time label supervision. GPU AMP behavior remains untested locally; sparse client class coverage may weaken 100-way local heads; the probe is diagnostic-only and materializes real activations; and the proxy retains the known test-as-validation limitation.