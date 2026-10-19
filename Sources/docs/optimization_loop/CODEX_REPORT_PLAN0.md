## Outcome

**PASS — PLAN-0 complete.** No training grid, long training, network action, push, or PR was performed. Existing uncommitted changes were preserved. Audited commit: `e74388c084984e7eaeb3625ce46bff5e37051dac`.

## Changes

- Fixed ablation precedence that previously reset presets to inherited `false` defaults.
- Fixed dead PRC wiring: source IDs were incorrectly compared with class IDs, and edge PRC used the wrong-tier memory.
- Added resolved-config/architecture/partition hashes, recorded seed and commit, checkpoint mismatch rejection, runtime counters, and unrounded JSONL metrics.
- Added architecture/config-specific result directories with overwrite refusal.
- Confirmed CIFAR `model: resnet50` actually resolves to a pretrained ResNet-18 client split:
  `[3,32,32] → [128,1,1] → [256,1,1] → [256] → 100`.
- Added deterministic Phase-0 tests and persistent loop-state documents.

## Verification

`python -m unittest -v tests.test_phase0_effects`: **7/7 PASS**.

- Memory tensor effect: PASS
- Reliability nonuniform weights: PASS
- PRC trainable-gradient effect: PASS
- Dropout active-client effect: PASS
- Six distinct ablation configs/hashes: PASS
- Checkpoint mismatch rejection: PASS
- Architecture/output overwrite guard: PASS
- Compile checks, unified runner listing, and `git diff --check`: PASS

## Experiments

| ID | Type | Result |
|---|---|---|
| `plan0_effects` | Deterministic CPU effects | PASS, 7/7 |
| `plan0_model_audit` | No-gradient model forwards | PASS, 3 paths |
| Training grid | Not run | N/A |

## Comparison

No performance comparison was run. The provisional H-SFP/SplitFL accuracy numbers remain unverified and should not be interpreted after discovering the ablation-collapse and PRC plumbing defects.

## Artifacts

- [PROJECT_STATE.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/PROJECT_STATE.md)
- [DECISIONS.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/DECISIONS.md)
- [RESULTS_SUMMARY.md](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/RESULTS_SUMMARY.md)
- [EXPERIMENT_REGISTRY.csv](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/EXPERIMENT_REGISTRY.csv)
- [Effect tests](/home/jackson/Desktop/Split-Federated-Merging-Learning/tests/test_phase0_effects.py)
- [Test log](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan0_effect_tests_20260713.log)
- [Model audit log](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan0_model_audit_20260713.log)
- [Ablation hashes](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan0_ablation_hashes_20260713.log)

## Risks

- Optional residual-generator and dropout-consistency flags remain unintegrated.
- Runtime counters have not yet been observed during a real-data smoke run.
- Segmentation currently selects its best checkpoint using `test_dataset`, requiring split-semantics review.

## Recommended next action

Claude should plan one bounded, single-seed smoke run to verify nonzero runtime counters and output metadata before authorizing any performance grid.