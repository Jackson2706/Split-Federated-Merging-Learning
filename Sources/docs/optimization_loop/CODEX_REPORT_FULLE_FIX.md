Diagnosed and fixed the remaining CUDA futex hotspot.

Identified cause:

- No full-only loss component is currently active:
  - `dropout_consistency_loss` is defined but never invoked.
  - `use_dropout_consistency` remains false in `full_e_hsfp`.
  - `use_residual_generator` is true, but classification always sets `self.residual_generator = None`.
- Prototype dropout and serverless simulation only perform CPU RNG/dictionary operations and contain no blocking wait.
- The actual remaining synchronization was PRC instrumentation: `.item()` forced a CUDA-to-host synchronization on every PRC call—previous runs recorded 600 calls.
- Reliability instrumentation similarly synchronized once per class.

Changes:

- [ehsfp/losses.py:45](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/losses.py:45)
  - Re-verified PRC batches every shared class into exactly two representation forwards.
  - Removed per-call `.item()` synchronization.
  - Accumulates detached device scalars asynchronously.

- [ehsfp/aggregation.py:132](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/aggregation.py:132)
  - Removed eager Python boolean and `.item()` conversions from per-class reliability instrumentation.

- [ehsfp/integrity.py:114](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/integrity.py:114)
  - Materializes accumulated device counters once at explicit phase-reporting boundaries.

- [tests/test_phase0_effects.py:144](/home/jackson/Desktop/Split-Federated-Merging-Learning/tests/test_phase0_effects.py:144)
  - Added regression coverage proving multi-class PRC uses exactly two forwards and still propagates gradients.

Semantics are preserved: PRC sampling, MSE reduction, reliability weights, loss values, gradients, and reported counters are unchanged. Only the timing of instrumentation synchronization moved out of hot CUDA loops.

Validation:

- 23/23 unit tests passed.
- Python compilation passed.
- `git diff --check` passed.
- No GPU verification performed, as requested.
- Existing uncommitted changes were preserved.