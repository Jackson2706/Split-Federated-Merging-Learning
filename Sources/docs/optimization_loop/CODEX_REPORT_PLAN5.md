## EXECUTION REPORT

Implemented PLAN-5 locally.

- Added `prototype_space={raw,centered_cosine}`; omitted flag defaults to `raw`.
- `raw` preserves the existing H-SFP pipeline and cloud classifier.
- `centered_cosine` maintains per-boundary global means, centers prototypes/samples and real eval features identically, recenters memory records, and uses a learnable-scale cosine head.
- Reliability, PRC, memory, prototype dropout, and residual sampling remain in the centered coordinate space.
- No baseline/default YAML was changed.

Key files: [prototype_space.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/prototype_space.py), [hierarchy.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/hierarchy.py), [CPU tests](/home/jackson/Desktop/Split-Federated-Merging-Learning/tests/test_plan5_prototype_space.py).

Tests: **15/15 passed** using full `unittest` discovery.

- PLAN-5 tests: 6/6.
- Existing effect/regression tests: 9/9.
- Raw full-pipeline regression: bitwise identical.
- Raw CIFAR cloud-head regression: bitwise identical to current H-SFP formula.
- Centered weighted global mean: approximately zero.
- Cosine logits: verified as `s·cosine`.
- Train/eval centering: exact same boundary transformations.
- `git diff --check`: passed.

Proxy base config: [plan2_cifar100_hsfp_baseline.yaml](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan2_cifar100_hsfp_baseline.yaml). Both runs retain seed `20260714`, IID partition, 200 clients, fraction 0.1, and 10 rounds.

Exact host-GPU commands, from repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 --set prototype_space=raw --set proxy_id=plan5_cifar100_hsfp_raw_seed20260714 --set output_dir=outputs/proxy/plan5_cifar100_hsfp_raw_seed20260714
```

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714 --set prototype_space=centered_cosine --set proxy_id=plan5_cifar100_hsfp_centered_cosine_seed20260714 --set output_dir=outputs/proxy/plan5_cifar100_hsfp_centered_cosine_seed20260714
```

Resolved launch hashes are distinct: raw `72cd671c…543c3`, centered `53e1de0e…42966`. The untouched PLAN-2 config still resolves to its recorded `817f3265…7555b`; the raw launch differs only because its unique output/proxy identifiers are hashed.

Communication delta: **0 prototype tensor bytes**. Means are derived from existing prototype means and support-count metadata; an assertion verifies payload size is unchanged. Runtime mean state is only 384 FP32 values for CIFAR-100’s two boundaries (~1.5 KiB). The cosine cloud head has fewer parameters than the current MLP, so expected VRAM is slightly lower, but GPU VRAM was not measured locally.

Unresolved risks: no GPU integration/proxy training was run; accuracy, peak VRAM, and multi-round mean dynamics remain for Claude’s host-GPU validation. The current hierarchy has no edge supervised classifier, so only the cloud angular head applies.