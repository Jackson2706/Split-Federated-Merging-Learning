# PLAN-29: Fix ISIC-2018 segmentation (3 real bugs) + re-run

## Root causes (diagnosed by Claude, GPU logs)
1. **Proxy configs aren't proxies.** `configs/proxy/plan_isic2018_*.yaml` only set `output_dir`;
   they inherit the FULL base configs (`configs/segmentation/*/isic_*.yaml` -> `default.yaml`),
   which run the full round budget (~60 rounds, ~240s/round => ~4h/run). This is why hierfl/h-sfp
   timed out at a 2.5h cap (only ~37-45/60 rounds). Unlike the CIFAR-100/HAM10000 proxies, no
   `epochs: 10` override was ever added.
2. **HeteroSFL crash (code=1, 5s):** `segmentation/HeteroSFL/runner.py:159` calls
   `criterion(pred_wide, mask)` where `pred_wide` is `[16,1,112,112]` but `mask` is
   `[16,1,224,224]` -> `DiceFocalLoss._sanity_check` raises `ValueError: Shape mismatch`. The wide
   prediction head's output is never upsampled to the mask resolution before the loss.
3. **Federated training collapse (code=0, but degenerate):** `Avg Stats after N rounds` shows IoU
   peaking at 26.83% (round 2), then monotonically decaying to 0.00% by round 38 while loss
   INCREASES (0.40 -> 0.75) — the model collapses to an all-background trivial solution. The
   runner reports the LAST round's IoU, not the BEST, so this collapse directly corrupts the
   final reported number. Root cause per earlier diagnosis: `DiceFocalLoss`'s focal term uses a
   single `alpha=0.25` applied uniformly (not foreground-weighted), so it does not correct the
   lesion/background class imbalance, and nothing prevents/recovers from the collapse.

## Hypothesis
(1) is a pure config fix. (2) is a pure shape-alignment fix. (3) needs both a training-stability
fix (foreground-weighted loss so the degenerate solution stops being attractive) AND a reporting
fix (track+report the BEST-IoU checkpoint across rounds, matching the classification pattern
already used everywhere else in this codebase, e.g. H-SFP's "Saved best model").

## Invariants
- Do not weaken/change the compared methods' capacity or unfairly advantage any one method.
- Keep the SAME partition/seed scheme across all 4 seg methods for the proxy table.
- Preserve uncommitted changes; local only; no GPU work from Codex (Claude verifies + runs).
- Full-budget (60-round) base configs stay untouched/available for a later full run; only the
  `configs/proxy/plan_isic2018_*.yaml` proxy layer gets the reduced round budget.

## Codex implementation tasks (NO GPU)
1. Add `epochs: 10` (or the config's equivalent round-count key — check each `default.yaml` under
   `configs/segmentation/*/`) to each of `configs/proxy/plan_isic2018_{federated,hierfl,heterosfl,
   hsfp}.yaml`, overriding the inherited full budget. Verify each config still resolves and all 4
   still share the same partition_hash (same as before — only round count changes).
2. Fix the HeteroSFL shape mismatch: in `segmentation/HeteroSFL/runner.py` (~line 159) and/or the
   model's wide-head forward, upsample `pred_wide` (and the narrow head if it has the same issue)
   to the target mask's spatial resolution (e.g. `F.interpolate(pred_wide, size=mask.shape[-2:],
   mode="bilinear", align_corners=False)`) before calling `criterion`. Check other segmentation
   methods (Federated/HierFL/H-SFP) don't have the same latent mismatch (they didn't crash, but
   verify their pred/target shapes are actually equal, not silently broadcast-compatible).
3. Fix the training collapse in `segmentation/*/clients/DiceFocalLoss.py` (all 4 methods, same
   pattern as before): make the focal term class-imbalance-aware — use separate foreground/
   background alpha weights (e.g. `alpha_fg` higher than `alpha_bg`, or compute alpha from the
   batch's actual foreground pixel ratio) instead of a single uniform `alpha=0.25`. Do not change
   the Dice term's math. Keep the change minimal and consistent across all 4 methods' loss files.
4. Add BEST-checkpoint tracking to the segmentation runners that currently only report the last
   round (Federated, HierFL, HeteroSFL — check which ones lack this; H-SFP already does via
   "Saved best model"): track validation IoU (or Dice) each round, save/report the best round's
   metric at the end (matching the classification runners' `best_f1` pattern), not just the final
   round. This protects the reported number even if a specific run later drifts.
5. Update `tools/fair_compare/aggregate.py`'s ISIC path if the "best" metric now lives in a new
   field/log line, so it extracts the BEST not the last round's IoU/Dice.

## Tests
1. Config resolution: all 4 proxy configs resolve with the new epochs override; partition_hash
   still matches across methods.
2. HeteroSFL shape fix: a CPU unit/smoke test constructing a wide-head output at a smaller spatial
   size than the mask and confirming the loss now computes without raising, with a sane value.
3. Loss re-weighting: a CPU test with a synthetic heavily-imbalanced mask (e.g. 95% background)
   confirms the new loss penalizes an all-background prediction more than before (or at least
   confirms it is no longer symmetric alpha=0.25 on both classes).
4. Best-checkpoint tracking: unit test that a runner's tracked "best" metric only increases/stays
   (never decreases) across a synthetic sequence of round IoUs including a later collapse.

## Experiment matrix
CIFAR partition-equivalent for ISIC (existing seed/partition scheme), 4 methods
(federated/hierfl/heterosfl/h-sfp) x 3 seeds (20260714/15/16), NEW reduced round budget
(~10 rounds). Claude runs on host GPU with a 2-round smoke-verify gate before the full grid
(require: no crash, no NaN, IoU > a small positive threshold, and NOT collapsing to 0 by the
smoke run's last round) — matches the pattern already used for the CIFAR/HAM proxies.

## Promotion criterion
All 4 methods x 3 seeds complete without crash/timeout/NaN, with a final BEST-round IoU/Dice > 0
(ideally in a plausible range, e.g. >10% given ISIC lesion segmentation baselines). Produces
`results/fair_comparison_isic2018.csv`.

## Rejection/rollback criterion
If federated (or any method) still collapses to 0 even after the loss reweighting, keep the
BEST-checkpoint reporting (so the run is still usable) but flag the instability explicitly in
DECISIONS.md rather than silently reporting a degenerate final-round number.

## Evidence Codex must return
File:line diffs for all 4 changes (config epochs, HeteroSFL shape fix, loss reweighting x4,
best-checkpoint tracking), confirmation of CPU test results, and the aggregator update if any.
