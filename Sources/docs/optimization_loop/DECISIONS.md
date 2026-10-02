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


## E-HSFP PRC/full_e DEADLOCK — DEFERRED after 5 fixes (2026-07-16)
- The PRC component (and full_e which includes it) deadlocks in torch.autograd _engine_run_backward
  at hierarchy.py:908 (edge SSL backward) whenever PRC is active. 5 Codex fixes failed: PRC batching,
  .item() sync removal, tensorboardX writer removal (red herring), graph-bounding via detach.
  Root cause is a genuine autograd-backward hang under PRC that cannot be reproduced by Codex
  (no GPU) and resists blind fixes. Needs a GPU-attached debugger (py-spy+sudo / gdb / minimal repro).
- DEFERRED. E-HSFP reported via WORKING components: memory / +dropout / +reliability (IID gains tiny,
  as expected; E-HSFP targets non-IID/staleness). Running a non-IID comparison of these vs baseline.
- IID ablation (10-round, corrected eval, seed 20260714): baseline 9.98, +memory 10.26,
  +memory+dropout 9.89, +memory+reliability (see registry).

## ISIC-2018 segmentation — 3/4 methods fixed and validated; H-SFP unresolved (2026-07-19)
Five real bugs found and fixed across this investigation:
1. Proxy configs weren't proxies (inherited full 60-round budget) -> added epochs:10 override.
2. HeteroSFL shape mismatch (pred_wide 112x112 vs mask 224x224) -> added align_prediction_to_mask()
   upsample before the loss.
3. Federated training collapse (IoU 26.8%->0.0% over rounds) -> uniform alpha=0.25 in DiceFocalLoss
   didn't reweight the lesion/background imbalance -> made alpha foreground-weighted (alpha_t =
   targets*0.75 + (1-targets)*0.25) across all 4 methods' DiceFocalLoss.py.
4. HeteroSFL aggregation dtype crash (Long dest vs Float source in index_put) -> generic
   is_floating_point() guard in aggregate_hetero() (and the same latent bug fixed in
   Federated/HierFL's FedAvgAggregator.py, which hadn't crashed but was silently wrong for
   integer buffers like num_batches_tracked).
5. Frozen-client-0 eval bug ported from classification/H-SFP/hierarchy.py's PROVEN fix into
   segmentation/H-SFP/hierarchy.py's _run_validation()/_train_decoder_phase() (use FedAvg-global
   client via client_cache, not structure[-1][0]) — a real, valid correctness fix (AST-identical
   to the working classification pattern, 27/27 tests pass) but did NOT resolve H-SFP-seg's issue.

RESULT: Federated 27.76+-0.86%, HierFL 63.17+-1.56%, HeteroSFL 25.80+-0.26% IoU — all clean,
stable, validated across 3 seeds. H-SFP-seg: DETERMINISTIC 0.00% IoU across all 3 seeds (not noise,
not resolved by the eval fix). Investigated and ruled out: decoder missing sigmoid (has one),
image/mask pairing (correctly identifier-based), edge-model selection (identical pattern to working
classification code), which client feeds the decoder (fix applied, no change). Remaining unexplored
hypotheses: decoder capacity/training budget (5 decoder epochs x 10 rounds = 50 total, on frozen
SSL-contrastive features never optimized for dense/spatial tasks) may be a genuine, real limitation
paralleling H-SFP classification's established structural encoder-ceiling finding, OR a
not-yet-found bug in decoder gradient flow / feature statistics between client-encoder output and
decoder input. DEFERRED pending user decision on further investigation.

## ISIC-2018 H-SFP 0% IoU — ROOT CAUSE FOUND AND FIXED (2026-07-20)
Bug #6, missed by the 2026-07-19 investigation above (which checked decoder-side hypotheses):
the NaN originates upstream, in client-side prototype extraction, not the decoder.

**Mechanism**: `_client_ssl_extraction_phase` (segmentation/H-SFP/hierarchy.py) extracts per-class
prototype features with `out = model(data)` inside `torch.amp.autocast` (fp16), then computes
`stacked.mean(0)` / `stacked.std(0, unbiased=False)` on those fp16 tensors. Squaring large ResNet
layer1 activations for the variance calculation routinely overflows fp16's ~65504 max, silently
producing Inf/NaN prototype stds (no crash, no exception). These NaN stds feed
`_generate_synthetic_data` (ehsfp/aggregation.py), so every synthetic feature sampled for Phase-2
edge SSL training is poisoned. Once one such NaN batch hits the edge model's BatchNorm layers in
`train()` mode, `running_mean`/`running_var` are corrupted permanently via the momentum-based EMA
update — `GradScaler` only guards the optimizer step against bad *gradients*, it does not protect
this forward-pass side effect, so once poisoned the buffers stay NaN forever (confirmed via
targeted `named_buffers()` NaN checks added at every phase boundary: clean after Phase 1, NaN in
100% of edge BatchNorm running stats after Phase 2, on all 4 edges, every run). Downstream,
`DiceFocalLoss`'s `torch.nan_to_num(preds, nan=0.0, ...)` (added by the earlier `64f8c4d` stability
fix) then silently converts the resulting NaN predictions to all-background, which is why training
looked normal (finite, non-NaN loss ~3-4) while actually learning nothing — hence the deterministic
0.00% IoU across all seeds.

**Fix** (segmentation/H-SFP/hierarchy.py):
1. `_client_ssl_extraction_phase`: cast `out = model(data)` to `.float()` immediately after the
   autocast forward, before it's stored/reduced — the actual root-cause fix.
2. `_edge_ssl_extraction_phase`: force the edge-model SSL forward (both the SupCon training loop
   and the eval-mode prototype-extraction loop) to fp32 (`autocast(enabled=False)` + explicit
   `.float()` casts on the loaded features), matching the precedent already set for the decoder
   in `64f8c4d`. Defense-in-depth given the edge model is a deep 13-block ResNet stack (layer2+3+4)
   fed unbounded-magnitude synthetic features.
Diagnosed via `HSFP_SEG_DIAG=1`-gated weight/buffer NaN checks added at every phase boundary
(post-client-SSL, post-client-aggregation, post-edge-SSL, post-edge-aggregation, pre-decoder-
forward) — left in place, zero cost when the env var is unset.

**RESULT** (3-seed re-run, same proxy config/methodology as the 2026-07-19 run):
H-SFP 50.28% / 45.14% / 46.38% IoU (mean 47.27 +- 2.19%), all clean exits, no NaN, no crashes.
Now ahead of Federated (27.76+-0.86%) and HeteroSFL (25.80+-0.26%), behind HierFL (63.17+-1.56%).
See RESULTS_SUMMARY.md for the updated table. No remaining unresolved bugs for H-SFP-seg.

## "Full E-HSFP" claimed-component scope — DECISION (2026-07-20)
A full implementation-status audit (docs/JOURNAL_SIMULATION_RESULTS.md Section 4.5) cross-checked
against the paper's 9 claimed novel components found 4 are not genuinely functional in a live
training run today:

| Component | Status |
|---|---|
| Episodic prototype memory | Real, wired, exercised |
| Prototype dropout (dropping mechanism) | Real, wired, exercised |
| Prototype dropout consistency loss | Implemented, imported, never called - dead code |
| Reliability-aware aggregation | Real, wired, exercised |
| Prototype replay consistency (PRC) | Real, wired, exercised - but deadlocks at proxy scale (see PRC entry above) |
| Diagonal Gaussian synthesis | Real, always-on default path (not really a distinguishing "component") |
| Low-rank covariance synthesis | Implemented, but only as an offline analysis script (scripts/camera_ready/run_covariance.py) - never reachable from a live E-HSFP run, no config flag exists to enable it live |
| Residual prototype generation | Implemented (ehsfp/generator.py), but both hierarchy.py files hardcode self.residual_generator = None regardless of the flag - complete stub. Every stored full_e_hsfp run's metadata falsely claims use_residual_generator: true. |
| Prototype fidelity regularization | Not implemented as a training-time regularizer at all. Only an offline MMD measurement exists; the sibling Frechet-distance function has zero call sites anywhere in the repo (dead code) |
| Prototype-induced drift control | Not implemented anywhere under this name. Closest adjacent code (prototype_space centering/whitening) is a different, classification-only ablation, not framed as drift control |

**Decision**: the `full_e_hsfp` ablation preset and any "full E-HSFP" result must NOT be reported
as validating all 9 claimed components - as currently implemented it validates at most 4 (memory,
dropout-masking, reliability, PRC-when-not-deadlocked), plus the always-on default synthesis. Two
components (residual generator, dropout-consistency loss) are inert despite their flags reading
`true`; two more (fidelity regularization, drift control) have no training-time implementation at
all; low-rank covariance synthesis is real but never live.

Rather than silently narrow the paper's claims or silently patch in throwaway implementations
under time pressure, this is escalated for explicit scoping: either (a) implement the 4 gaps for
real before any "full E-HSFP" number is used in the paper, or (b) reduce the journal's empirically-
claimed contribution to the subset that is genuinely implemented and validated (memory + dropout +
reliability + PRC, once unblocked), reframing low-rank synthesis / residual generation / fidelity
regularization / drift control as motivating design directions or future work rather than validated
components. This document does not choose between (a) and (b) - that is a paper-scope decision,
not an engineering one - but no further work should silently proceed as if `full_e_hsfp` already
means what its name implies. Tracked as an open item in docs/JOURNAL_SIMULATION_RESULTS.md Section 20.

## classification/H-SFP reliability-aggregation NaN crash — ROOT CAUSE FOUND AND FIXED (2026-07-21)
`--ablation hsfp_memory_reliability` on HAM10000 crashed at epoch 1 with `ValueError: Out of range
float values are not JSON compliant: nan` while flushing runtime counters. This is the same fp16-
autocast-overflow-in-std-computation bug already fixed in `segmentation/H-SFP/hierarchy.py`
(2026-07-20 entry above) but never ported to the separate classification codebase:
`calculate_prototypes_and_distribution` and the client SSL extraction loop computed `mean`/`std`
on fp16 autocast output; squaring large activations for the variance overflowed fp16's ~65504 max,
producing non-finite prototype sigmas. These fed `build_reliability_features()`'s `sigma_magnitude`
feature into `PrototypeReliabilityNetwork` (`Sigmoid`-terminated, so its *output* is bounded in
(0,1) — but `Sigmoid(NaN) = NaN`, so a NaN input still poisons the output), and from there into the
`reliability.weight_variance_sum` runtime counter, which accumulates via `+=` and so stays NaN
forever once poisoned.

**Fix** (`classification/H-SFP/hierarchy.py`): cast to `.float()` immediately after the autocast
forward, before mean/std reduction, in both `calculate_prototypes_and_distribution` and the client
SSL extraction loop — same pattern as the segmentation fix, root-caused rather than patched with
`nan_to_num`. Diagnosed and fixed by Codex; reviewed and GPU-verified by Claude (the exact
previously-crashing command now completes cleanly end-to-end). This also explains 3 earlier
2026-07-14 `scripts/journal_experiments` interval failures with the identical crash signature.

## segmentation/SplitFL ISIC all-foreground collapse — ROOT CAUSE FOUND AND FIXED (2026-07-22)
The newly-added `segmentation/SplitFL` ISIC segmentation support (2026-07-21) ran without
crashing but wasn't learning: a 3-seed, 10-round proxy campaign showed IoU stuck at exactly
0.2600 (Dice≈0.4095) across all 3 seeds, completely flat across all 10 epochs within each run.
Directly measuring the ISIC validation set's foreground-pixel fraction (0.2685, sampled over 100
masks) confirmed the model had collapsed to predicting all-foreground for every input — for a
binary task, `IoU = foreground_pixels/total_pixels` exactly when every pixel is predicted
positive, independent of image content or seed, which is exactly what was observed.

**Root cause**: `segmentation/SplitFL/runner.py` hard-coded `SGD(momentum=0.9)` for both the
client and server optimizer regardless of the config's `optimizer: "adam"` (a pre-existing,
already-documented cross-method protocol inconsistency, Section 21 of
`docs/JOURNAL_SIMULATION_RESULTS.md` — never caught before because it didn't obviously break a
classification head fine-tuning on a pretrained backbone, but starves a completely
randomly-initialized decoder of an adequate training signal within only 10 rounds). A second,
independent bug compounded it: the server optimizer was recreated from scratch every global round
even though the server model itself persists across rounds, discarding all accumulated
momentum/Adam moment-estimate state each time.

**Fix**: added a `_make_optimizer()` factory respecting the config's actual optimizer choice, and
moved server-optimizer construction outside the round loop so its state persists. Added an opt-in
`SPLITFL_SEG_DIAG=1` gradient-flow diagnostic (prediction stats + per-tier gradient norms on the
first batch), left in place at zero cost when disabled, matching this repo's established
`HSFP_SEG_DIAG` pattern. Diagnosed and fixed by Codex; reviewed and GPU-verified by Claude — a
full 10-round rerun shows genuinely varying per-epoch IoU (not flat), non-trivial gradient norms
reaching both client and server, and a real result (40.74% Test IoU, single seed) rather than the
collapsed 26.00%. The 2 pre-fix SplitFL seeds are discarded, not reused. HSFL was not touched by
this fix (separate file) and its status remains independently unconfirmed.

**Update 2026-07-22 (later same day)**: SplitFL's 3-seed campaign completed and confirmed healthy
across all seeds (40.74% / 41.60% / 41.98% Test IoU, mean 41.44 +- 0.52%, genuinely varying
per-epoch trajectories in every run). HSFL's own 3-seed campaign — the first real multi-epoch
check HSFL had ever received — also completed and was independently confirmed healthy, with no
code changes required: 32.01% / 32.00% / 39.50% Test IoU (mean 34.50 +- 4.33%), val-vs-test IoU
diverging within every seed and Dice/IoU ratios consistent with genuine segmentation in all three
(HSFL only validates at its final epoch by config, so this check relied on val/test divergence and
Dice/IoU cross-consistency rather than an intra-run per-epoch curve). Both methods' results are
recorded in `results/fair_comparison_isic2018.csv` and `docs/JOURNAL_SIMULATION_RESULTS.md`
Section 6 / Section 20 item 8, which is now fully closed.

## ISIC-2018 H-SFP segmentation ablation campaign — COMPLETE (2026-07-22)
Following up on the ablation-precedence bug fix (root-caused earlier this session: pre-`cf6454c`
config resolution silently collapsed all non-baseline E-HSFP presets to baseline behavior), the
classification-side reruns (CIFAR-100, HAM10000) had already been completed in earlier session
work. Auditing what remained open in Section 20 item 2 found the **segmentation (ISIC-2018)
ablation study had never been run at all** — zero markers, empty `results/journal/ablation/`.

Ran 4 of the 6 ablation presets (`baseline_hsfp`, `hsfp_memory`, `hsfp_memory_dropout`,
`hsfp_memory_reliability`) on ISIC-2018 H-SFP, single seed (20260714). Deliberately used the
**proxy 10-round protocol** (`configs/proxy/plan_isic2018_hsfp.yaml`) rather than the frozen
60-round default, to bound runtime to ~4 hours instead of ~24 — consistent with the precedent set
by the ISIC fair-comparison campaign earlier this session. `hsfp_memory_reliability_prc` and
`full_e_hsfp` were excluded, both still blocked on the unresolved autograd deadlock (sudo-blocked
`py-spy`/`gdb` access).

Results: `baseline_hsfp` 50.28% / `hsfp_memory` 47.92% / `hsfp_memory_dropout` 47.83% /
`hsfp_memory_reliability` 46.74% Test IoU, Dice 64.95%/62.63%/62.37%/61.69% respectively. All 4
values genuinely distinct (no collapse, no regression of the config-precedence bug), all exit code
0, no crashes. `baseline_hsfp`'s value exactly reproduces the existing fair-comparison H-SFP result
for the same seed — strong cross-check that this campaign used identical settings to the earlier
work. Notably, IoU decreases monotonically with each added component here, the opposite direction
from the CIFAR-100 classification ablation (where memory improved accuracy by +3.18pp) — reported
honestly as a genuine single-seed data point, not explained away or over-interpreted; multiple
seeds are needed to know whether this is a real segmentation-specific effect or single-seed noise.

This result is proxy-protocol and explicitly **not comparable** to the existing 60-round
CIFAR-100/HAM10000 ablation rows — recorded as its own row in
`docs/JOURNAL_SIMULATION_RESULTS.md` Section 4 audit table, not merged with them. Section 20 item
2's segmentation piece is closed; the PRC/full presets remain blocked, and extending all ablation
work (classification and segmentation) to the full 5-seed protocol remains queued.

## HAM10000 H-SFP seed-invariant 10.9835% — ROOT CAUSE CONFIRMED (2026-07-23)
The three invalid fair-comparison rows were a **training collapse (Hypothesis B), not a seed-
plumbing failure**. Their archived `run_metadata.json` files have three distinct partition hashes
(`09e04b...`, `d234dd...`, `a5e5dd...`), and the round-1 edge composition also differs by seed.
Thus NumPy partitioning and client sampling were consuming the configured global seed. The exact
10.9835247% is also not uniform guessing over nine classes: HAM10000 has seven configured classes,
and every validation prediction was class 0. Class 0 occurs 220/2003 times in the fixed validation
split, exactly 10.9835247%.

**Mechanism**: every archived HAM run reports `Edge ... final loss: nan` and `Cloud ... final loss:
nan` from round 1, followed by `NaN detected in model output` and predictions `[0]`. The client and
edge prototype paths retained fp16 autocast activations and reduced variance in fp16. Squaring
moderately large ResNet activations overflowed fp16, non-finite prototype sigmas contaminated the
synthetic features, and the NaN logits made `argmax` deterministically choose index 0. This is the
same numerical mechanism fixed on the classification side on 2026-07-21 by promoting autocast
features to fp32 before prototype mean/std; the invalid campaign predates that fix. It differs from
the known-healthy CIFAR-100 proxy, whose cloud loss is finite and falls from 3.27 to 0.23 over its
first seven rounds.

**Reproducibility audit/fix**: each classification H-SFP run now writes
`run_fingerprints.json`, containing the seed, exact initial client/edge/cloud state hashes,
partition hash, and ordered first-round client-selection hash/list. Tests cover same-seed identity,
different-seed divergence, and the fp16-variance regression (400-valued fp16 features now produce
finite fp32 statistics). `python -m unittest -v tests.test_hsfp_seed_and_collapse` passes 3/3;
`pytest` is not installed in this execution environment.

**Verification limitation**: the requested two CUDA smoke commands were launched with seeds
20260714/20260715 and their logs were saved under
`docs/optimization_loop/logs/seed_fix_verification/`, but both stop at the config's explicit CUDA
guard because this container cannot communicate with an NVIDIA driver. No replacement CPU metric
is claimed. The logs are retained so a GPU host can rerun the exact review commands.

**GPU verification COMPLETE, 2026-07-23 (Claude, host GPU)**: Codex's first fingerprinting diff
crashed on real GPU hardware — `state_dict_hash()` called `.view(torch.uint8)` on ResNet
BatchNorm's 0-dim `num_batches_tracked` buffer, which `view()` cannot reshape (`nn.Linear`, the
toy model in the original unit test, has no such buffer, so the bug was invisible to Codex's
GPU-less sandbox). Dispatched a second bounded Codex task; fix was `.reshape(-1)` before the dtype
view, plus a new regression test using a real `nn.BatchNorm2d`. Independently re-ran all 6 tests
(`tests.test_hsfp_seed_and_collapse` + `tests.test_classification_metric_reporting`) — genuinely
pass.

Reran the 2-seed, 3-epoch HAM10000 H-SFP verification on the actual host GPU
(`docs/optimization_loop/logs/seed_fix_verification_gpu/`):

| Seed | Exit | Cloud final loss | Prediction class spread | Test Acc |
|---|---|---:|---|---:|
| 20260714 | 0 | 0.0108 (finite) | `[0 1 2 3 4 5 6]` — all 7 classes | 11.98% |
| 20260715 | 0 | 0.0136 (finite) | `[0 1 2 3 4 5 6]` — all 7 classes | 11.93% |

Both genuinely distinct from each other and from the old collapsed 10.9835247%/all-class-0
pattern. Compared `run_fingerprints.json` between the two seeds: `partition_hash`,
`first_sampled_clients`, and `initial_model_hashes.cloud` all differ (confirming seed-sensitive
partitioning, client sampling, and the freshly-initialized cloud classifier head); `client`/`edge`
initial-model hashes are identical across seeds, which is *expected*, not a bug — both load
`torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)` (ImageNet-pretrained, deterministic
regardless of seed; confirmed in `classification/H-SFP/models/resnet50_ham10000.py`).

**Section 5.2 (seed correctness) is closed for classification H-SFP.** This 3-epoch smoke is
verification evidence only, not a replacement result — the original 3-seed, 10-round HAM10000
proxy campaign that produced the `INVALID` row remains `INVALID` and requires a full rerun with
this fix before a new valid aggregate can be reported (tracked in
`docs/journal_campaign/MASTER_EXPERIMENT_MATRIX.csv`).

## E-HSFP PRC/full_e DEADLOCK — in-process diagnostics added, no blind fix (2026-07-23)
Added strictly opt-in `EHSFP_DEADLOCK_DIAG=1` instrumentation to the classification H-SFP runner.
Once the per-run output directory exists, it enables autograd anomaly detection and Python's
in-process `faulthandler.dump_traceback_later(..., repeat=True, exit=False)`; this needs neither
ptrace nor sudo and periodically dumps every Python thread to
`ehsfp_deadlock_diagnostics.log`. The same log receives timestamped CUDA allocated/reserved bytes
and `torch.cuda.utilization()` when the installed PyTorch exposes it (otherwise an explicit
unavailable/error field). `EHSFP_DEADLOCK_DIAG_SECONDS` controls both cadences and defaults to
60 seconds. Disabled behavior is only an environment-variable branch; no thread, file, anomaly
detection, timer, or CUDA sampling is created. No deadlock fix was attempted.

Static audit findings:

1. `EpisodicPrototypeMemory.add()` is a plausible scale-sensitive *slowness* source, although not
   consistent with a stack observed specifically in backward. The 200-client proxy selects 20
   clients per round. With IID CIFAR-100 each 250-example client partition can cover most of the
   100 classes, so client memory can receive roughly 1,800-2,000 class records per round. After
   its 500-record cap, every individual addition sorts the entire ~501-element Python list before
   slicing it; that is commonly well over 1,000 full sorts per round, plus edge-memory writes.
   This path was audited but deliberately not optimized in this instrumentation-only round.
2. The PRC `representation_fn` is called twice while the edge model is explicitly in `train()`
   mode. Thus stochastic layers may differ across `z_current`/`z_memory`, as intended by the
   existing training path. However the concrete CIFAR-100 edge model used by the proxy contains
   GroupNorm (no running statistics), not BatchNorm, so no BatchNorm running-stat buffer can be
   mutated between the two forwards in the known repro. The active AlexNet edge implementation
   likewise has no BatchNorm. No version-counter hazard was found in this path.
3. The 2026-07-16 line reference is stale: current `hierarchy.py:908` adds weighted PRC to the
   loss; the actual `scaler.scale(loss).backward()` is now line 911. Each edge owns one Adam
   optimizer and GradScaler, initialized once before the global-round loop and reused normally.
   Each minibatch executes `zero_grad`, scaled backward, scaler step, and scaler update in order;
   the loss is reduced to a Python scalar only after the update and is not retained across loop
   iterations. No suspicious extra tensor, optimizer, or scaler state was found around this loop.

Added a CPU regression that performs 20 PRC calls while accumulating two class records per round
in `EpisodicPrototypeMemory`. Every loss remains finite and differentiable, every parameter
gradient norm remains finite and bounded, and memory growth is asserted each round. The complete
`tests.test_phase0_effects` module passes 13/13; a short diagnostic smoke also confirmed repeated
all-thread dumps and resource snapshots are written without ptrace.

**GPU repro attempt 1, 2026-07-23 (Claude, host GPU, `EHSFP_DEADLOCK_DIAG=1` enabled)**: launched
`configs/proxy/plan2_cifar100_hsfp_full_e.yaml` (the documented repro config: CIFAR-100, 200
clients, `full_e_hsfp`, seed 20260714) — the first time this specific bug has ever run with real
in-process instrumentation. Result: **segfault (exit 139)**, not the historically-described
indefinite hang, ~30s into Epoch 1 Phase 1. The diagnostic log's single periodic dump (fired at the
30s mark, right as the crash occurred — the log file is truncated mid-write of the final thread's
frame) shows every background thread parked normally (DataLoader worker feed queues, pin_memory
loop, the diagnostics sampler itself, all in ordinary `Event.wait()`/`queue.get()` states) except
one: the hot/main thread was captured inside `torch/_ops.py:1123` in `__call__` — PyTorch's
operator-dispatch entry point — at the instant of the crash. CUDA memory was still at 0 bytes
allocated/reserved at the single snapshot taken (t=0s baseline only; no further snapshot survived
the crash).

**Isolation test — baseline rerun, identical config/seed, `EHSFP_DEADLOCK_DIAG` unset (no
anomaly-detection, no faulthandler)**: ran cleanly. Completed Epoch 1 in full (all 5 edges'
SSL losses finite, ~1.01–1.05; cloud supervised loss 3.7152; validation predictions spanning 62 of
100 classes, not collapsed; Acc 4.57%; checkpoint saved) and proceeded into Epoch 2 without
incident, running 8+ minutes past the point where the diagnosed run crashed at 30s.

**Conclusion so far (strong, not yet final)**: the segfault is best explained by
`torch.autograd.set_detect_anomaly(True)`, which the new diagnostic module enables unconditionally
whenever `EHSFP_DEADLOCK_DIAG=1` is set — anomaly-detection mode adds extra native-level
introspection around every backward op and is documented upstream to be able to destabilize
certain code paths, unlike the (verified passive/read-only) `faulthandler` timer, which cannot by
itself crash a healthy process. This is NOT yet proof that the original 2026-07-14/16
"deadlock" (an indefinite hang, not a crash) is resolved — the baseline has only reached early
Epoch 2 so far. Letting it continue to build confidence before concluding whether the 3 previously-
applied fixes (PRC batching, `.item()`-sync removal, `max_replay_batch` graph-capping) actually
already resolved the underlying issue and it was simply never re-verified end-to-end, or whether a
hang still appears later in the 10-round run. **Action taken**: `set_detect_anomaly(True)` should
be made independently toggleable (separate from the faulthandler/CUDA-sampling diagnostics, which
are the actually-useful, non-invasive parts) rather than bundled — tracked as a follow-up fix to
`ehsfp/deadlock_diagnostics.py`, not yet applied pending the baseline's full-run outcome.

**Diagnostic safety follow-up (2026-07-23)**: split the invasive anomaly detector from the passive
diagnostics after the observed full-e CIFAR-100 proxy segfault. `EHSFP_DEADLOCK_DIAG=1` now enables
only the confirmed-safe faulthandler traceback timer and CUDA memory/utilization sampling.
`torch.autograd.set_detect_anomaly(True)` additionally requires the distinct, explicit
`EHSFP_DEADLOCK_DIAG_ANOMALY=1` opt-in and otherwise remains off. Session shutdown restores the
prior anomaly state only when that extra mode was enabled, and startup reports which pieces are
active. This is a diagnostic safety fix motivated by the observed segfault; it is **not** a fix
for, or evidence of resolution of, the original deadlock. Independently reviewed: diff is minimal
and correct, both new unit tests (`tests/test_deadlock_diagnostics.py`) genuinely pass on rerun.

**DEADLOCK DOES NOT REPRODUCE — full 10-round verification, 2026-07-23 (Claude, host GPU)**:
Reran `configs/proxy/plan2_cifar100_hsfp_full_e.yaml` (CIFAR-100, 200 clients, `full_e_hsfp`, seed
20260714 — the exact documented repro config) to completion, no diagnostics, no artificial
timeout beyond a generous 90-minute bound. **Result: exit code 0, `Total Run Time: 4052.31s`,
all 10 rounds completed cleanly.** Losses stayed finite throughout (cloud supervised loss fell
3.72 → 2.13 → ... → 0.90 → 1.09, never NaN/Inf); validation predictions spread across all 100
CIFAR-100 classes by round 8 onward (not collapsed); Best Validation/Test Acc 7.20% (best
checkpoint at epoch 9).

Confirmed via `runtime_counters.jsonl` that every claimed E-HSFP component was genuinely active
for the full run, not silently degraded to baseline: by the final round, `memory.reads=112250`,
`memory.writes=21841`, `memory.replay_calls=235`; `prc.calls=180960` with **100% nonzero PRC
loss** (`prc.nonzero_loss_calls=180960`, mean loss 0.406); `reliability.aggregation_calls=58` with
nonzero, nonuniform weight variance (`0.00035`, `nonuniform_weight_calls=5085`);
`dropout.apply_calls=48`, `dropout.changed_calls=27`. All four components (episodic memory, PRC,
reliability-aware aggregation, prototype dropout) were live and doing real work throughout.

**Conclusion: the long-standing `full_e_hsfp`/PRC autograd deadlock (documented since
2026-07-14/16) does not reproduce with the current codebase at 10-round/200-client CIFAR-100 proxy
scale.** The most likely explanation is that the 3 earlier fix rounds (PRC batching, `.item()`-sync
removal, `max_replay_batch` graph-capping — all already present in `ehsfp/losses.py`, confirmed by
this session's static audit) actually did resolve the underlying issue at the time they were
applied, but the fix was never re-verified with a real end-to-end full-length run afterward —
everyone continued treating it as blocked based on the older evidence. It is equally possible that
environment/library drift since 2026-07-14/16 (PyTorch/CUDA version changes are the most plausible
candidate) changed timing-sensitive behavior enough to avoid whatever triggered the original hang.
Both explanations are consistent with the observed evidence; this session cannot distinguish
between them further without the original environment to compare against, and it doesn't need to —
what matters for the campaign is that current code, as it stands, does not hang.

**This is a major unblock, with explicit caveats, not yet a full clearance**:
- Single seed (20260714), single dataset (CIFAR-100), 10-round **proxy** protocol only. Not yet
  verified on HAM10000, ISIC-2018, the other 4 final seeds, or the full 5-seed frozen (200-round)
  protocol — those are all still required before `full_e_hsfp`/`hsfp_memory_reliability_prc` can be
  marked `VERIFIED` in the final campaign matrix.
- Only `full_e_hsfp` was tested end-to-end this round; `hsfp_memory_reliability_prc` (PRC without
  the residual generator, memory-only up to reliability+PRC) has not been independently re-verified
  — it shares the same PRC code path so is very likely also fine, but "very likely" is not
  "verified."
- The segfault encountered on the first attempt was conclusively isolated to this session's own
  diagnostic instrumentation (`torch.autograd.set_detect_anomaly(True)`), not the training code —
  already fixed by making it a separate, off-by-default flag.

**Section 5.1 is no longer a blocking item for queuing more `full_e_hsfp`/PRC runs** — it moves
from `BLOCKED_RESOURCE` to requiring ordinary verification (more seeds/datasets), same as every
other not-yet-final cell in the matrix, rather than being structurally unable to run at all.

## Section 5.3 metric/checkpoint correctness — auditable artifacts and frozen segmentation protocol (2026-07-23)

Completed the CPU-only Section 5.3 implementation and verification.

Classification now has one shared prediction-artifact schema and independent validator in
`classification/training_metrics.py`, with the standalone CLI
`scripts/journal_experiments/validate_metrics.py`. Artifacts contain only raw `predictions` and
`labels`; the validator ignores training-loop metrics and recomputes accuracy, zero-safe macro-F1,
per-class recall, and the confusion matrix with sklearn. H-SFP, HSFL, SplitFL, and HeteroSFL all
fit the in-memory prediction/label pattern, so none were skipped. Each writes
`best_val_predictions.npz` when its accuracy-selected best validation state changes and
`last_round_predictions.npz` on the final round. HeteroSFL validation now also runs on the final
round when `print_every` does not divide the configured epoch count, which is required to create a
truthful last-round artifact. Its misleading legacy `best_f1` variable/key (which actually held
accuracy) was renamed `best_val_top1`; checkpoint/reporting selection remains unchanged.

All four classification JSON outputs now expose separate `last_round_validation`,
`best_validation`, and `selected_checkpoint` records. Existing selection policies were preserved:
H-SFP, HSFL, and SplitFL select their historical accuracy-best validation checkpoint; HeteroSFL
still tests/selects the last trained model because it has never saved or restored a best model.
Legacy scalar/list fields remain for compatibility, but the new structured fields make the
checkpoint-tied macro-F1 and accuracy unambiguous.

Segmentation's five duplicate `compute_iou_and_dice` definitions were replaced by the one
canonical function in `segmentation/training_metrics.py`. Its docstring freezes the existing
protocol: strict `> 0.5` threshold, whole-batch flattened pixel aggregate rather than per-image
averaging, and exact IoU/Dice `0/0` for empty prediction/label masks. HierFL,
Federated `Client.py`, Federated `test.py`, HeteroSFL, and H-SFP import the same function.
SplitFL and HSFL now import it directly from the shared module; HeteroSFL's package re-export also
continues to resolve for compatibility.

The six segmentation output paths (HierFL, Federated, HeteroSFL, SplitFL, H-SFP, and HSFL) now
also expose separate `last_round_validation`, `best_validation`, and `selected_checkpoint`
records without changing their existing selection rules. Selected-checkpoint test metrics are
written only after that checkpoint is evaluated, preventing the former H-SFP/HSFL ordering from
persisting a JSON file before selected-checkpoint test results existed.

CPU verification: all edited Python files compile; 7/7 tests in
`tests.test_classification_metric_reporting` plus
`tests.test_segmentation_metric_protocol` pass, including the deliberately imbalanced
accuracy-versus-macro-F1 case, a hand-computed three-class validator case, literal canonical-import
checks for all five segmentation call sites, the strict-threshold/global-flatten regression, and
the empty-mask `0/0` edge case. The existing `tests.test_research_metrics` suite also passes 8/8.
No GPU validation was required or attempted. Nothing in deadlock diagnostics, seed handling,
communication accounting, or unrelated campaign code was changed as part of Section 5.3.

## Section 5.4 communication-ledger reconciliation — CIFAR anomaly definitively traced (2026-07-23)

**Conclusion (definitive): explanation (c), a cross-version comparison artifact.** The reported
`30,414.23 -> 1,430.47 MB` change is not an episodic-memory communication saving and is not a
feature flag skipping a required transmission. The table combines a July 9 `baseline_hsfp` run
with July 20 memory-preset reruns made after commit `cf6454c` (July 16) changed CIFAR-100's client
feature boundary in `classification/H-SFP/models/cnn_cifar.py`. The old client emitted a
`[64,32,32]` feature map per sample; the new client includes `layer2` and global average pooling
and emits `[128,1,1]`. Since extraction sends a mean and standard-deviation tensor for each class,
one old class entry costs `2*64*32*32*4 = 524,288` bytes (0.5 MiB), while one new entry costs
`2*128*1*1*4 = 1,024` bytes (0.0009765625 MiB): exactly **512x smaller per class**.

The archived route subtotals prove this mechanism against the real runs. The old baseline's
`client_to_edge_data_MB=27,326.25` represents about 54,652 old-size class entries; the memory
reruns' `53.39 MB` represents about 54,671 new-size class entries. Those nearly identical counts
(the small difference is ordinary sampled-client/class-coverage variation) rule out a silent
memory-gated skip. The edge payload also follows the architecture change: the old edge emitted
512-dimensional tensors (`57.80 MB` archived), while the new edge emits 256-dimensional tensors
(`29.10 MB`), approximately a 2x reduction. Model-transfer subtotals changed too because the tier
partition and parameter sizes changed. Therefore the four archived totals are not
communication-comparable; a current-code baseline rerun would be required for an ablation
communication comparison.

The control-flow audit independently agrees. `_client_ssl_extraction_phase` never reads
`use_episodic_memory`, `client_memory`, or `edge_memory`; it groups every loader activation by
label and returns exactly one mean and one population-standard-deviation tensor for every class
present. Its `add_communication(..., "client_to_edge_MB")` call occurs immediately after every
successful extraction and before memory insertion/mixing. The analogous edge-to-cloud call occurs
immediately after every successful edge extraction and before edge-memory insertion/mixing.
Memory changes downstream reconstruction inputs, not whether the current prototype packet is
charged. The only skip branches are serverless simulated client/edge timeouts; `hsfp_memory` and
`hsfp_memory_reliability` do not enable serverless simulation, yet have identical totals, so those
cannot explain the shared collapse. `t1=5` and `t2=10` only gate model aggregation in Phase 4;
both archived logs contain the same 12 edge and 6 cloud aggregation events. They do not gate
prototype extraction or its ledger calls.

CPU regression evidence now executes `_client_ssl_extraction_phase` with identical synthetic data
under both memory-flag values and asserts identical class dictionaries, tensors, support counts,
and byte cost. A second synthetic test holds class keys fixed and proves the historical-to-current
boundary shapes produce an exact 512x payload ratio. The ledger itself now has a from-first-
principles one-round reconciliation test: six explicitly listed packet groups across four routes
sum by hand to exactly 51 MiB, with both unused flat-FL routes at zero and
`assert_communication_total` verifying the route sum.

Section 5.4's enabled-module checklist is also satisfied by the already-completed real GPU
`full_e_hsfp` run documented in the preceding **DEADLOCK DOES NOT REPRODUCE** entry:
`memory.reads/writes`, `prc.calls` (100% nonzero loss), reliability aggregation calls with
nonuniform weights, and dropout apply/changed calls were all nonzero. No rerun was needed.

## Section 5.5 per-round history/series logging — durable shared CSV (2026-07-23)

Completed the CPU-only Section 5.5 implementation for both H-SFP task families. The new shared
`ehsfp/history_logger.py` freezes a one-row-per-round `history.csv` schema and appends, flushes,
and fsyncs each completed round immediately. The schema contains the round and elapsed wall time,
training loss, primary validation metric name/value, current-round deltas for every route in the
existing shared communication tracker, client/edge/cloud CPU/RSS/CUDA resource fields, client and
edge episodic-memory occupancy, and an event field. No second communication ledger was introduced:
each loop snapshots `ehsfp.communication`'s cumulative public counters at round start and subtracts
that snapshot when writing the row.

Classification H-SFP writes cloud training loss and validation accuracy; segmentation H-SFP fit
the same lifecycle cleanly and was also wired, writing cloud training loss and validation IoU.
Both runners place `history.csv` beside `metrics_unrounded.jsonl` in the reserved run directory.
When `eval_every > 1`, non-evaluation rounds have an empty validation value rather than repeating
a stale prior-round value. Existing serverless timeout records produce tier-specific events such
as `client_timeout`; no recovery event exists in the simulator, so no recovery signal was
invented. Memory-disabled runs record zero occupancy.

The H-SFP loops do not, contrary to the initial campaign assumption, currently accumulate
phase-specific CPU/RSS or CUDA allocated/reserved histories: `psutil` is imported but unused for
measurement, and the runner only records a whole-run CUDA allocated peak after training. Those
frozen resource columns are therefore present but empty for H-SFP. Adding new sampling would have
violated Section 5.5's instruction to reuse existing measurements and would create semantics not
shared with the methods that already track them. Nothing was skipped for segmentation; its
resource cells are empty for the same reason.

The five requested research-metric definitions were confirmed already frozen without formula
changes. `docs/JOURNAL_SIMULATION_RESULTS.md` Section 3.9 explicitly freezes stability (W=5),
rounds-to-convergence (moving-average window 3, 0.95 threshold, patience 3), per-class/tier L2
prototype drift, recovery gap plus 95%-recovery rounds, and MMD prototype fidelity.
`ehsfp/research_metrics.py` implements those definitions with function docstrings and points back
to Section 3.9. The separately discussed stability-bound term remains deliberately
`BLOCKED_NO_THEORY`; it is not one of these five and was not fabricated or changed here.

CPU verification passes: the new synthetic regression writes exact multi-round loss, validation,
route-delta, resource, memory, and timeout values, checks the complete frozen column order, and
simulates interruption after two of five intended rounds to prove the partial file remains valid
and parseable. The history, communication-accounting, and research-metric suites pass 10/10, and
all five touched Python implementation files compile. No GPU run was attempted. Sections 5.1
through 5.5 are now closed; producing the Section 8 figure-series bank still requires new runs,
because historical runs cannot retroactively acquire trajectories that were never persisted.
