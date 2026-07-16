# Optimization Loop Decisions

## PLAN-0 decisions

1. Preserve the dirty worktree as authoritative. The initial status and commit are captured in `logs/codex_plan0_20260713_225256.log`; no cleanup/revert was performed.
2. Treat named ablations as effective presets. The former ordering in `ehsfp/config.py` applied inherited explicit `false` values after the preset, silently disabling it. The resolver now applies the preset last and rejects unknown names.
3. Repair PRC as plumbing, not an algorithm change. Both hierarchy implementations built current dictionaries using source IDs while memory uses class IDs, creating an empty intersection; the edge phase also read edge memory despite operating on client-tier tensors. Current outputs are now class-aggregated and matched to client memory at edge / edge memory at cloud.
4. Hash the fully resolved YAML together with effective E-HSFP settings. The full SHA-256 is checkpoint/output identity; a separate architecture hash incorporates class, source digest, state shapes, and parameter counts.
5. Never overwrite a prior run. Architecture/config-specific run directories are atomically reserved and an existing path raises `FileExistsError`.
6. Default an omitted seed to 42 so every unified-runner invocation has an explicit recorded seed.
7. Do not promote provisional accuracy values. PLAN-0 ran only CPU effect tests and model-forward audits; performance comparison remains unverified.

## Phase-0 disposition

No mandated core component remains a demonstrated dead path after the plumbing repairs: memory, reliability, PRC, dropout, distinct config hashes, checkpoint mismatch rejection, and result isolation all pass deterministic tests. Optional residual-generator and dropout-consistency flags remain unintegrated and must not be claimed as verified. Phase 1 may be planned, but it should begin with one bounded smoke run that inspects runtime counters before any grid.

## PLAN-1 decision

1. Do not promote to Phase 1. The bounded `baseline_hsfp` CPU smoke crashed at `classification/H-SFP/hierarchy.py:781`: CPU autocast emitted `bfloat16` client prototypes, but the edge convolution bias remained FP32 (`RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same`). The `full_e_hsfp` preset completed because reliability-weighted aggregation promoted its inputs to FP32, so the two-preset gate is asymmetric and criterion (a) is not met.
2. Stop after the declared two presets. No grid or hyperparameter tuning is authorized; a targeted Phase-0 plan must make the baseline CPU dtype path consistent before PLAN-1 is rerun.
3. Treat exact memory read/write counts, reliability min/max/variance, scalar PRC loss, and per-call dropout active-set size as unresolved instrumentation gaps. Persisted substitutes show 812 memory-changed classes, 436 final client-memory records, 20 replay calls, 461 nonuniform reliability-weight calls, 564 nonzero PRC calls, and 15/16 cumulative active source events, but the requested exact values are not written to JSONL.

## PLAN-1b decision

1. Keep the Phase-1 gate closed. The dtype fix and incremental counter persistence are implemented and pass eight local effect/regression tests, including a direct CPU `bfloat16` input to the FP32 edge convolution, but the declared CUDA smoke could not execute because this environment exposes no CUDA device (`torch.cuda.is_available() == False`, device count 0; NVML unavailable).
2. Fail closed when a smoke config declares `require_cuda: true`; do not silently consume the bounded gate on CPU. Neither preset trained and no run output directory was reserved. Both local launches also stopped during imports because the active environment lacks the declared `psutil` dependency; CUDA availability remains the gating infrastructure issue even after dependencies are restored.
3. Preserve the predeclared matrix and hashes for retry: seed `20260713`, 8 clients, 2 rounds, `baseline_hsfp` hash `c9d23d23…ead6bb`, and `full_e_hsfp` hash `c792f82d…3749f`. Retry exactly these two presets sequentially in a CUDA-visible environment after restoring the existing project dependencies; do not expand to a grid.

## PLAN-1c decision

1. Keep episodic prototype records CPU-backed and copy their tensors to the active tier device only at consumption boundaries. Reliability feature construction now builds directly on its passed device, including record means/stds and the class center.
2. Treat mixed-source stacking and PRC as part of the same device-consistency fix. Prototype aggregation moves each source tensor before `torch.stack`; memory replay mixtures accept the tier compute device, including memory-only classes; and PRC moves both current and memory distributions to the passed model device before synthesis.
3. Keep the Phase-1 gate closed until Claude reruns the unchanged two-preset smoke on the host GPU. Local verification is CPU-only and performs no training.

## PLAN-1d decision

1. Mark the unchanged two-preset host-GPU smoke gate `smoke-pass`. Both `baseline_hsfp` and `full_e_hsfp` exited 0 on CUDA, completed two rounds without a traceback, used the same partition hash (`379ce6b4...3908`), and produced distinct resolved hashes and run directories.
2. Accept the component-effect gate: baseline counters remained zero, while `full_e_hsfp` recorded memory reads/writes/replays `4000/2000/4000`, 600 nonzero PRC calls (mean loss `0.0896928405`), 551 nonuniform reliability calls, and four dropout applications (mean active set `3.75`).
3. Do not promote the smoke accuracies (`2.93%` and `2.70%`) or declare a strongest baseline. They are two-round gate observations on the CIFAR-100 test split aliased as validation, not fair performance evidence. Phase 1 may proceed only to the declared fair-comparison audit and bounded proxy preparation.

## PLAN-3 validation-protocol draft (implementation deferred)

1. Replace test-as-validation with a deterministic, stratified carve-out from the official CIFAR-100 **training** split. A proposed default is 45,000 fit examples and 5,000 held-out validation examples, stratified by class with seed `20260714`; build federated user partitions only from the 45,000-example fit subset.
2. Keep the official 10,000-example CIFAR-100 test split untouched throughout training, scheduler decisions, checkpoint selection, ablation selection, and tuning. Evaluate it exactly once for a promoted final checkpoint. Persist fit/validation/test index hashes and the federated partition hash in run metadata.
3. Apply the identical split indices, transforms, metric definition, and checkpoint-selection rule to every method. Existing proxy results remain explicitly labeled test-as-val and are not retroactively promoted.
4. Implementation and baseline reruns are deferred to a later isolated plan. This protocol change must not block or be mixed into the PLAN-3 oracle diagnostic, which intentionally reuses the current loader to isolate the representation/reconstruction question.

## PLAN-5/PLAN-6 decision

1. Classify PLAN-5 `centered_cosine` as neutral: its 3.48% best 10-round proxy result does not improve
   on the 3.81% raw result. Do not promote prototype-space centering from this evidence.
2. Move to PLAN-6 encoder supervision while leaving prototype transport, memory, reliability, PRC,
   dropout, and `prototype_space` orthogonal. `client_objective=ssl` is the default and must remain
   bitwise-equivalent to the current client step. Alternative losses consume only each client's own
   features and labels; no sample or gradient crosses tiers.
3. Run only the predeclared `ssl` control and `supervised` candidate first. Promote `supervised` only
   if client-encoder linear-probe top-1 reaches at least 40% and end-to-end proxy top-1 reaches at
   least 10%. If the probe rises but end-to-end remains near 4%, return to reconstruction; if it does
   not rise above roughly 30%, investigate split point or optimization.
4. Record an interpretation caveat: the code's existing `ssl` implementation is already a two-view
   supervised-contrastive loss using local labels. PLAN-6 preserves it for regression integrity;
   `ssl_supcon` is consequently an additional one-view SupCon term, not the first use of labels.

## PLAN-7 whitened_cosine — FAIL (worse than raw); STRATEGIC CHECKPOINT (2026-07-14)
- whitened_cosine proxy best = 1.31% (raw 3.81%, centered 3.48%, supervised 4.14%). Whitening HURT.
- Pattern across 4 well-motivated candidates: prototype-space geometry fixes (center/cosine/whiten)
  and a client CE head do NOT transfer from the static diagnostic to the co-trained pipeline.
- Structural evidence of a ceiling far below baselines (SplitFL 44.6%):
  (1) client-encoder linear-probe stuck ~30% and identical across checkpoints (pretrained-dominated,
  barely trains); (2) tier degradation L1 30% -> L2 edge 26% -> pipeline ~4%; (3) cloud overfits
  synthetic prototypes (loss 0.25 while real-val declines from a round-1 peak).
- DECISION: stop tweaking prototype geometry. Raised a strategic checkpoint to the user
  (attack encoder-training/synthetic-dynamic vs reframe to comms-accuracy Pareto vs formal pivot
  review vs deeper client split). Implementer = Claude while Codex is out until Jul 20.

## PLAN-9 EVAL-BUG FIX — corrected raw baseline (2026-07-14)
- BUG: _run_validation graded client-0's LOCAL model (never selected under seed => frozen pretrained
  encoder). Confounded ALL prior proxy results. FIX: evaluate FedAvg GLOBAL client (avg of trained
  clients in client_cache). classification/H-SFP/hierarchy.py:_run_validation.
- Corrected RAW = 7.92% (was 3.81% frozen-eval) — ~2x, plateaus ~7.5-7.9 after epoch4.
- Corrected encoder linear-probe = 0.2607 (frozen pretrained was 0.3006): client SupCon on noise-aug
  [128,1,1] features slightly DEGRADES linear separability. The 2x gain is pipeline CONSISTENCY, not
  better encoder. Two ceilings remain: encoder ~26-30%, reconstruction 26%->8%.
- ACTION: re-run centered_cosine + whitened_cosine on corrected eval (prior 3.48/1.31 were on frozen
  eval, INVALID). Next encoder lever: proper image-aug client SupCon (encoder actually learns) or
  freeze-backbone (keep 30% pretrained).

## PLAN-10 corrected-eval geometry re-test (2026-07-14)
- On the CORRECTED (FedAvg-global) eval: raw 7.92, centered_cosine 9.57 (WIN +1.65), whitened 6.32.
- PROMOTE centered_cosine (comms-free centering + cosine head). REJECT whitened_cosine (noise
  amplification; below raw even on corrected eval, though far better than frozen-eval 1.31).
- Prior frozen-eval conclusions (centered neutral / whitened catastrophic) were INVALID — the eval
  bug confounded them. centered_cosine is a real, reproducible corrected-eval win.
- Next: freeze-backbone (encoder 26->30) + full_e_hsfp E-HSFP ablation, both on corrected eval.

## PLAN-11..13 E-HSFP full_e DEADLOCK — DEFERRED (2026-07-14)
- full_e_hsfp reproducibly DEADLOCKS at PROXY scale (200 clients): futex_wait_queue, ~588MB GPU,
  stalls at setup/epoch-1. Unaffected by num_workers=0 or OMP/MKL/OPENBLAS=1. Works on SMOKE.
  Scale-sensitive AND E-HSFP-specific (baseline/centered/freeze run fine).
- Cannot introspect (py-spy needs sudo). DEFERRED. Future debug: bisect ablation ladder
  (hsfp_memory -> +dropout -> +reliability -> +prc) at proxy scale; or sudo py-spy.
- num_workers now configurable (hierarchy.py:408). PIVOT to Phase-6-lite: 3-seed confirm of best
  config freeze+centered (9.76% seed0) + comms/VRAM Pareto vs SplitFL 44.57.

