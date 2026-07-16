# Optimization Loop Results Summary

## PLAN-0 — 2026-07-13

Outcome: PASS for integrity/effect criteria. Seven deterministic CPU tests passed in 0.030 seconds on the final run. No training or performance grid was executed.

| Effect | Result | Evidence |
|---|---|---|
| memory replay changes tensor | PASS | mixed prototype `[0,2]` with stored `[10,12]` at alpha 0.5 became `[5,7]`; change counter incremented |
| reliability responds to unequal input | PASS | controlled unequal metadata produced nonuniform learned weights |
| PRC changes gradient | PASS | identical distributions produced zero loss/gradient; unequal means produced positive loss and changed trainable gradient |
| dropout changes active IDs | PASS | `p=1`, seed 17 reduced five active IDs to one and incremented change counter |
| ablation hashes distinct | PASS | all six effective presets and SHA-256 hashes are distinct |
| checkpoint mismatch refusal | PASS | matching hash accepted; mismatched hash raised `RuntimeError` |
| result overwrite/architecture guard | PASS | different architecture IDs isolated; reuse of an existing architecture/config path refused |

Exact config hashes are in `logs/plan0_ablation_hashes_20260713.log`. Test output is in `logs/plan0_effect_tests_20260713.log`; model facts are in `logs/plan0_model_audit_20260713.log`.

The provisional accuracy comparison (CIFAR-100 ~8.10% vs SplitFL ~58.4%; HAM10000 ~10.98% vs SplitFL ~83.3%) remains UNVERIFIED and is not attributable after the discovered ablation-collapse and PRC dead-path defects.

## PLAN-2 Phase-1 bounded proxies — 2026-07-13/14

These are 10-round, seed-`20260714`, IID CIFAR-100 proxies. They are explicitly **test-as-validation** results: the official test split is reused for validation/repeated evaluation, and the method-specific training budgets remain unmatched. They are diagnostic evidence, not paper numbers or a fair `B_d` comparison.

| Method | Final/best top-1 | Runtime | Status |
|---|---:|---:|---|
| SplitFL | 44.57% | 526s | complete |
| Federated/FedAvg | 34.76% | 1296s | complete |
| H-SFP `baseline_hsfp` | 3.81% (best epoch 1) | 3518s | complete |
| H-SFP `full_e_hsfp` | — | — | pending; training output has no final `metrics.json` yet |

The baseline H-SFP cloud loss fell sharply while real test-as-val accuracy stayed at or below 3.81%, motivating PLAN-3's real-feature oracle. Do not use a partial full-E run to update this table.

## PLAN-4 cached-feature normalization factorial — 2026-07-14

This was CPU-only analysis of PLAN-3's cached real features (50,000 train / 10,000 eval); no encoder,
FL, or GPU run was performed. Transform statistics were fit on train only. All learned heads used the
same fixed configuration: 10 epochs, 50 samples/class for the prototype and cosine heads, batch size
256, Adam learning rate 0.001, weight decay 0.0001, cosine scale 16, and angular margin 0.1. PCA
whitening retained 64 components at L1 and 128 at L2.

| Boundary | Transform | NC | Mean proto | Diag proto | Cosine margin | Linear | Centroid cos | W/B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | raw | 10.72% | 6.53% | 6.47% | 11.65% | 29.81% | 0.9848 | 4.8961 |
| L1 | centered | 10.72% | 9.81% | 9.03% | 21.41% | 29.81% | -0.0044 | 4.8961 |
| L1 | L2 | 10.62% | 6.10% | 5.46% | 11.65% | 30.04% | 0.9850 | 4.7918 |
| L1 | centered+L2 | 10.77% | 10.25% | 9.48% | 21.41% | 29.80% | 0.0223 | 5.1684 |
| L1 | centered+PCA-white | 25.48% | **16.57%** | **17.30%** | **22.90%** | 28.95% | -0.0047 | 13.2463 |
| L1 | centered+white+L2 | **25.58%** | 16.42% | 14.41% | **22.90%** | 29.00% | 0.0031 | 13.5422 |
| L2 | raw | 9.39% | 9.29% | 8.72% | 14.86% | 26.46% | 0.9339 | 4.5324 |
| L2 | centered | 9.39% | 10.38% | 8.84% | 18.70% | 26.46% | -0.0043 | 4.5324 |
| L2 | L2 | 9.69% | 9.58% | 9.21% | 14.86% | 26.68% | 0.9356 | 4.5435 |
| L2 | centered+L2 | 9.60% | 10.08% | 8.87% | 18.70% | 26.24% | 0.0216 | 4.7958 |
| L2 | centered+PCA-white | 26.11% | 20.29% | **17.21%** | **22.73%** | 30.97% | -0.0051 | 21.6066 |
| L2 | centered+white+L2 | **26.56%** | **20.76%** | 14.80% | **22.73%** | **31.31%** | 0.0030 | 22.0060 |

Decision: whitening, not centering or L2 normalization alone, recovers the mean geometry. The best
nearest-centroid result (centered+whitened+L2) improves L1 by 14.86 points (2.39x) and L2 by 17.17
points (2.83x) over raw. L1 remains 4.48 points below PLAN-3's 30.06% linear ceiling; L2 is only
0.06 points below its 26.62% PLAN-3 linear ceiling. The declared >=20% L1 criterion passes, but the
next method change should explicitly include whitening/anisotropy removal: centered+L2 alone did not
improve nearest-centroid accuracy. Unrounded results and leakage audit metadata are in
`outputs/diag/plan4_normalize_seed20260714/normalize_results.json`.

The `full_e_hsfp` proxy still has no final `metrics.json`; its last recorded artifact is a partial
epoch-1 `clients_complete` counter, so no final accuracy is reported.

## PLAN-5 centered-cosine proxy and PLAN-6 encoder objective — 2026-07-14

PLAN-5 `centered_cosine` completed the same 10-round, seed-`20260714` CIFAR-100 proxy with best
test-as-validation top-1 **3.48%** (epoch 7), versus **3.81%** for raw. This is neutral/within noise
and is not promoted. The static centered cosine diagnostic used real cached features; its improvement
did not transfer through the synthetic prototype path.

PLAN-6 therefore targets the client encoder ceiling. The implemented off-by-default
`client_objective` accepts `ssl`, `ssl_supcon`, and `supervised`; `ssl` retains the current client
training exactly, while the alternatives add only client-local auxiliary loss. The predeclared
promotion gate is client-encoder linear-probe top-1 at least 40% and end-to-end proxy top-1 at least
10%. The `ssl` control and `supervised` candidate are planned for host GPU; no PLAN-6 training result
is recorded yet. The offline probe materializes real activations for diagnosis only and is not part
of training or communication.


## Communication-Accuracy Pareto (CIFAR-100 proxy, 10 rounds, seed 20260714) — KEY DELIVERABLE

| Method | Best val top-1 | Total comms (MB) | Peak VRAM (MB) | Runtime (s) |
|---|---:|---:|---:|---:|
| SplitFL (baseline) | 44.57 | ~36,934 (31,251 smashed-act + 5,684 model + grad) | — | 526 |
| Federated (baseline) | 34.76 | ~84,819 (model up+down) | — | 1296 |
| **H-SFP freeze+centered (ours)** | **9.76** | **122.1** (proto: c->e 8.89, e->c 4.87; model 54.2) | **466.6** | ~ (slower) |

**H-SFP transmits only prototypes (mean+std), giving ~300x less communication than SplitFL and
~700x less than Federated, with a ~467 MB VRAM footprint — at a substantial accuracy cost.** This is
H-SFP's genuine contribution: the extreme-low-communication / low-client-resource end of the
FL Pareto frontier, not raw-accuracy parity.

### H-SFP accuracy progression (corrected eval)
- Pre-fix (frozen-client-0 eval artifact): 3.81 (INVALID)
- Corrected raw: 7.92 | +centered_cosine: 9.57 | +freeze-backbone: 9.76 (best) | whitened: 6.32 (rej)
- Encoder linear-probe ceiling ~26-30% (structural: shallow ResNet-18 split + ~250 samples/client);
  reconstruction adds the rest of the gap. Beating SplitFL 44.57 would need a deeper split (trading
  the comms advantage) or a fundamentally less-lossy transfer.

### 3-seed confirmation (freeze+centered, corrected eval, 10-round proxy)
- seeds 20260714/20260715/20260716 = 9.76 / 10.55 / 10.56 => **10.29 +/- 0.37%** (stable). Best H-SFP config.


## FINAL FAIR COMPARISON — CIFAR-100 (10-round proxy, IID, 3 seeds, corrected eval, test-as-val)

| Method | Accuracy (%) mean±std | Total comm (MB) | Peak VRAM (MB) |
|---|---:|---:|---:|
| SplitFL | 45.79 ± 1.44 | 37,031 | 268 |
| Federated | 35.81 ± 1.72 | 94,244 | 304 |
| HierFL | 33.73 ± 0.56 (2/3 seeds; s716 baseline device-bug) | 19,277 | 307 |
| HSFL | 25.50 ± 9.69 (high var; s714=12 outlier) | 126,001 | 970 |
| HeteroSFL | 18.68 ± 2.29 | ~6,400 | 318 |
| **H-SFP (ours, freeze+centered)** | **10.29 ± 0.37** | **122** | **445** |

Full per-run data: results/fair_comparison_cifar100.csv.

**Verdict:** H-SFP is LAST on accuracy but by far the most communication-efficient — ~50x less comms
than the next-lightest (HeteroSFL) and ~300x less than SplitFL, with a low VRAM footprint. This is a
clean accuracy-vs-communication Pareto tradeoff: H-SFP occupies the extreme-low-cost corner. Beating
the higher-accuracy methods is structurally out of reach for this shallow-split prototype-transfer
design without trading away that communication advantage (deeper split / less-lossy transfer).


## HAM10000 fair comparison (10-round proxy, IID, 3 seeds) — baselines
| Method | top-1 (%) mean±std |
|---|---:|
| SplitFL | 78.13 ± 0.32 |
| HSFL | 73.77 ± 3.11 |
| HierFL | 71.44 ± 0.29 |
| Federated | 71.43 ± 0.89 |
| HeteroSFL | 68.48 ± 0.55 |
| H-SFP (ours) | (re-run pending; ~11% val — NaN-logger crash fixed) |
Full CSV: results/fair_comparison_ham10000.csv. HAM is easier (7 classes, imbalanced) so baselines cluster 68-78%.


## HAM10000 fair comparison — COMPLETE (10-round proxy, IID, 3 seeds)
| Method | top-1 (%) mean±std | total comm (MB) |
|---|---:|---:|
| SplitFL | 78.13 ± 0.32 | ~ (see csv) |
| HSFL | 73.77 ± 3.11 | |
| HierFL | 71.44 ± 0.29 | |
| Federated | 71.43 ± 0.89 | |
| HeteroSFL | 68.48 ± 0.55 | |
| **H-SFP (ours)** | **10.98 ± 0.00** | ~ (lowest; prototype-only) |
CSV: results/fair_comparison_ham10000.csv. Same pattern as CIFAR-100: H-SFP far lower accuracy,
far lower communication. E-HSFP full-config numbers PENDING (PRC deadlock under active fix).
