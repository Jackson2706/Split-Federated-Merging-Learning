## EXECUTION REPORT — BLOCKED

### Outcome

PLAN-1b’s CUDA promotion criterion could not be evaluated. This environment exposes no CUDA device (`device_count=0`; NVML unavailable), so neither preset trained. The Phase-1 gate remains closed.

### Changes

- Guarded autocast to CUDA-only execution.
- Added defensive bfloat16/FP32 convolution input casting.
- Added exact memory read/write/replay counters, reliability-weight variance, scalar PRC loss, and dropout active-set metrics with explicit baseline zeros.
- Flushes counters after each round phase for crash-safe partial evidence.
- Records device and peak VRAM.
- Added fail-closed `require_cuda: true` to prevent accidental CPU smoke execution.

### Verification

- Compilation: PASS.
- Effect/regression tests: 8/8 PASS.
- CPU dtype regression: bfloat16 edge input → FP32 output without crash.
- Registry CSV validation and `git diff --check`: PASS.
- Existing uncommitted changes preserved.

### Experiments

| preset | hash | output dir | mem r/w/replay | reliab var | PRC loss | dropout active-N | final train loss | device | peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| baseline_hsfp | `c9d23d23…ead6bb` | None | N/A | N/A | N/A | N/A | N/A | CUDA unavailable | N/A |
| full_e_hsfp | `c792f82d…3749f` | None | N/A | N/A | N/A | N/A | N/A | CUDA unavailable | N/A |

Both launches also encountered the active environment’s missing declared `psutil` dependency during imports. No training, output directories, grid, full schedule, network action, push, or PR occurred.

### Comparison

Promotion criterion: **NOT VERIFIED**. Distinct hashes exist, but CUDA completion, finite losses, runtime-counter differences, and VRAM compliance have no experimental observations.

### Artifacts

- [CUDA preflight](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan1b_smoke_20260713/cuda_preflight.log)
- [Verification log](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan1b_smoke_20260713/verification.log)
- [Baseline launch log](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan1b_smoke_20260713/baseline_hsfp.log)
- [Full-E launch log](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/logs/plan1b_smoke_20260713/full_e_hsfp.log)
- [Experiment registry](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/EXPERIMENT_REGISTRY.csv:6)
- [Decision record](/home/jackson/Desktop/Split-Federated-Merging-Learning/docs/optimization_loop/DECISIONS.md:23)

### Risks

CUDA behavior and peak VRAM remain untested. The active Python environment is also missing `psutil`, `kornia`, and `tensorboardX`.

### Recommended next action

Restore the declared project dependencies in a CUDA-visible environment, then rerun only these two presets sequentially with the unchanged seed/config. Do not promote or expand to Phase 1 until both complete and produce the required counter/VRAM table.