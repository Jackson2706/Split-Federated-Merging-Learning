# Source-to-claim map

Status vocabulary: **inherited**, **implemented**, **proposed**, **unresolved**.

| Claim/design element | Status | Repository evidence | Draft treatment |
|---|---|---|---|
| Three-tier split topology, client contrastive training, synthetic upper-tier training, slow model averaging | inherited | `Submissions/ECCV-2026/sec/3_method.tex`; `Sources/classification/H-SFP/hierarchy.py` | Credited to H-SFP, not a MiSFP contribution. |
| Local multiple components | implemented | `Sources/misfp/fit.py:fit_class_mixture`, `_fit_fixed`, `stream_moments`; `Sources/classification/MiSFP/hierarchy_misfp.py:_client_ssl_extraction_phase` | Fixed K=2/4 and support-limited fallback described. |
| Adaptive K | implemented heuristic | `Sources/misfp/fit.py:fit_class_mixture`; defaults in `Sources/misfp/config.py` | Held-out NLL/dimension plus penalty and improvement threshold; no optimality claim. |
| Retain different clients' components | implemented | `Sources/classification/MiSFP/hierarchy_misfp.py` pooling path; `Sources/misfp/merge.py:concat/compress_classes` | MiSFP-Edge isolates across-client retention; IDs explicitly local. |
| Budgeted merging | implemented | `Sources/misfp/merge.py:compress_classes`, `best_pair`, `merge_pair`; `Sources/misfp/packets.py:packet_nbytes` | Cap first, then smallest heuristic cost/byte; class never dropped; infeasibility reported. |
| Exact mass/mean/diagonal second-moment preservation | implemented and derived | `Sources/misfp/merge.py:pool_two`; population variance in `Sources/misfp/packets.py` | Proposition with proof; excludes density/decision-boundary preservation and wire rounding. |
| Actual packet convention and bytes | implemented | `Sources/misfp/packets.py:MixturePacket`, `serialize_packet`, `transmit` | Count + mean + variance; int/float layout and header accounted. |
| Sampling preserves components | implemented | `Sources/misfp/sampling.py:sample_mixture`, `synthesize` | Component categorical precedes Gaussian; balanced class prior separate from within-class weights. |
| Nonlinear edge boundary re-estimation | implemented | `Sources/classification/MiSFP/hierarchy_misfp.py` around boundary modes `propagate` and `refit` | Propagate uses equal MC count/component and original mass; refit estimates effective mass. |
| Shared-encoder diagnostic | implemented | `Sources/misfp/config.py:misfp_extraction`; `Sources/classification/MiSFP/hierarchy_misfp.py` shared snapshot path; `Sources/misfp/compat.py` | Diagnostic control, not a guarantee of semantic alignment. |
| Multiple modes always improve accuracy | unresolved | No completed matched multi-seed evidence imported | Not claimed; evaluation distinguishes alternative outcomes. |
| Full covariance alternative | proposed comparison only | No full-covariance packet path in `Sources/misfp` | Discussed conceptually; not listed as implemented. |
| Formal privacy guarantee | unresolved/not provided | No DP or secure aggregation in MiSFP modules | Explicit limitation. |

## Execution-path trace

`Sources/main.py` selects the classification MiSFP runner → `Sources/classification/MiSFP/runner.py` reuses H-SFP data/models → `MiSFPHierarchicalFL` overrides client extraction, edge, and cloud phases → client training features are clustered by `fit_class_mixture` → packet is serialized/dequantized → edge concatenates/compresses, samples, trains, transforms and re-estimates output statistics → edge packet is serialized → cloud concatenates/compresses, samples, and trains its supervised head. The inherited H-SFP loop performs same-tier model synchronization and overall evaluation.

## Open empirical/method decisions

- Resolve the primary Dirichlet alpha, participation rate, round count, and optimizer settings from the final experiment configs before replacing `resolved config` table fields.
- Decide whether the journal primary result uses propagate or refit; keep the other as an ablation.
- Pre-register mode-coverage clustering and projection scope before looking at results.
- Verify HAM10000 patient split and MiSFP integration before promoting it from planned to supported.
- Literature leads FedGMKD/covariance feature-transfer require primary-source identity and protocol verification before citation.
