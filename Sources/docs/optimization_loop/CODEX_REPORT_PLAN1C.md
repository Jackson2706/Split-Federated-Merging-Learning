## EXECUTION REPORT

Implemented PLAN-1c without training or config changes. CPU-backed memory storage remains unchanged.

Device coercions:

- Reliability features: [reliability.py:75](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/reliability.py:75), [reliability.py:96](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/reliability.py:96), [reliability.py:98](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/reliability.py:98)
- Reliability bootstrap targets: [reliability.py:125](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/reliability.py:125), [reliability.py:135](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/reliability.py:135), [aggregation.py:210](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:210), [aggregation.py:211](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:211)
- Mixed-source aggregation before stacking: [aggregation.py:36](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:36), [aggregation.py:37](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:37), [aggregation.py:93](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:93), [aggregation.py:97](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:97), [aggregation.py:107](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:107), [aggregation.py:108](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:108)
- Memory prototype mixtures: [memory.py:186](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:186), [memory.py:187](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:187), [memory.py:189](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:189), [memory.py:190](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:190), [memory.py:205](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:205), [memory.py:206](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:206), [memory.py:210](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:210), [memory.py:211](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/memory.py:211)
- PRC current and memory tensors: [losses.py:50](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/losses.py:50), [losses.py:51](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/losses.py:51), [losses.py:52](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/losses.py:52), [losses.py:53](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/losses.py:53)
- Classification tier-device propagation: [hierarchy.py:812](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/hierarchy.py:812), [hierarchy.py:938](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/hierarchy.py:938), [hierarchy.py:1148](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/hierarchy.py:1148), [hierarchy.py:1211](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/hierarchy.py:1211)

Verification:

- Device-consistency test added at [test_phase0_effects.py:37](/home/jackson/Desktop/Split-Federated-Merging-Learning/tests/test_phase0_effects.py:37), covering reliability, aggregation, replay mixture, PRC, and retained CPU storage.
- Existing CPU effect/regression suite plus new test: **9/9 PASS**.
- Python compilation: **PASS**.
- `git diff --check`: **PASS**.
- Decision recorded at [DECISIONS.md:29](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/DECISIONS.md:29).
- All pre-existing uncommitted changes were preserved.

GPU verification remains pending Claude’s host-GPU rerun of the unchanged two-preset smoke gate.