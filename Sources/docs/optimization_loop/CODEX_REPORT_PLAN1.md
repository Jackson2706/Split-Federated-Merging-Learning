## EXECUTION REPORT — FAIL

PLAN-1 did not satisfy the promotion criterion: `baseline_hsfp` crashed during round 1 at `classification/H-SFP/hierarchy.py:781` due to a CPU `bfloat16` input / FP32 convolution mismatch. `full_e_hsfp` completed two rounds in 514.14s with finite losses.

| preset | resolved hash | output dir | mem r/w/replay | reliab var | PRC loss | dropout active-N | final train loss |
|---|---|---|---|---|---|---|---:|
| baseline_hsfp | `64ed61e…d8e37b` | [baseline output](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/Figure/data/runs/10abc078df3f/64ed61e13fa99deb68a2c79741772a4a1aa3d1e2be24abf8d1978c59b2d8e37b) | `0/0/0`¹ | `0`¹ | `0`¹ | 8 selected; dropout off | N/A — crash |
| full_e_hsfp | `2321d950…cd27b0` | [full-E output](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/Figure/data/runs/10abc078df3f/2321d950d0bf0b6fec5303adbc338de6d87a18a90a6ddd6fe9948236b9cd27b0) | `812/436/20`² | `>0` — 461 nonuniform calls | `>0` — 564/564 nonzero calls | 15/16 cumulative source events | 4.514366662248652 |

¹ Baseline components were disabled, but the crash occurred before runtime metrics were persisted.  
² Persisted proxies: memory-changed classes / final client-memory records / replay calls. Exact read/write counters are not currently emitted.

Gate checks:

- Both finish with finite losses: **FAIL** — baseline crashed.
- Full-E nonzero and differing activity: **PARTIAL** — demonstrated, but baseline runtime artifacts are absent.
- Distinct hashes/output directories: **PASS**.
- Same seed and partition: **PASS** — seed `20260713`, partition hash `2a0d4f39…402ef`.

Evidence:

- [Smoke config](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/smoke/plan1_cifar100_ehsfp_gate.yaml:1)
- [Baseline log](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan1_smoke_20260713/baseline_hsfp.log)
- [Full-E log](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan1_smoke_20260713/full_e_hsfp.log)
- [Full-E JSONL metrics](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/Figure/data/runs/10abc078df3f/2321d950d0bf0b6fec5303adbc338de6d87a18a90a6ddd6fe9948236b9cd27b0/metrics_unrounded.jsonl)
- [Registry rows](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/EXPERIMENT_REGISTRY.csv:4)
- [Failure decision](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/DECISIONS.md:17)

Unresolved risks: the baseline CPU dtype boundary requires a targeted Phase-0 fix; exact memory read/write, reliability variance, scalar PRC loss, and per-call dropout active-set values are not persisted. No grid, full schedule, push, PR, or external network action was performed.