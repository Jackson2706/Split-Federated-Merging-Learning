# MiSFP: Mixture-Preserving Prototyping for Hierarchical Split-Federated Learning

MiSFP is the method of the second journal extension. It extends **H-SFP** (the
ECCV 2026 conference baseline) by representing each class with **several
mean–variance prototypes**, preserving within-class modes through the
client → edge → cloud hierarchy and compressing them to a communication budget.

MiSFP is not E-HSFP. **E-HSFP** (the first journal extension) studies episodic,
missing and stale packets: memory, reliability weighting, dropout and replay.
Those mechanisms are not part of MiSFP. MiSFP variants refuse to run with them
enabled (`misfp/config.py`); they remain available under `--method h-sfp`.
The provisional name MC-HSFP is retired, and no code or outputs use it.

**Claim boundary.** Gaussian mixtures are an established tool. The candidate
contribution is *adaptive preservation and compression of within-class modes
across hierarchical split boundaries*. Nothing here shows that a GMM improves
accuracy by construction. The pilot below is exploratory.

---

## 1. Implementation map (actual code, verified 2026-10-02)

| Concern | Location | Finding |
|---|---|---|
| Entry point | `main.py` → `REGISTRY["classification"]["misfp"] = classification/MiSFP` | One added registry line; nothing else in `main.py` changed |
| Runner | `classification/MiSFP/runner.py` | Puts `classification/H-SFP` on `sys.path` and reuses its data, models and `HierarchicalFL` |
| MiSFP round logic | `classification/MiSFP/hierarchy_misfp.py` (`MiSFPHierarchicalFL`) | Subclass that overrides only the client packet, edge and cloud phases |
| Shared library | `misfp/` (`packets.py`, `fit.py`, `merge.py`, `sampling.py`, `config.py`, `compat.py`) | Task-agnostic |
| Original H-SFP | `classification/H-SFP/hierarchy.py` | **Not modified** by MiSFP (it has other uncommitted work in progress) |
| Split boundaries (CIFAR-100/ResNet path) | `classification/H-SFP/models/cnn_cifar.py` | Client = ResNet-18 up to layer2 + GAP → `[B,128,1,1]`. Edge = 1×1 conv MLP → `[B,256,1,1]`, flattened to 256 |
| Transmitted representation | `hierarchy.py:_client_ssl_extraction_phase` | Raw, un-normalized post-GAP features (`prototype_space=raw`). The cloud L2-normalizes its own input. MiSFP keeps the raw convention and never renormalizes samples |
| Variance convention | `hierarchy.py:894`, `calculate_prototypes_and_distribution` | Population std (`unbiased=False`, denominator n). Packets carry **std**. MiSFP carries **variance** internally |
| Baseline aggregation | `ehsfp/aggregation.py` (`aggregation_mode: average`) | **Unweighted** mean of client means, and σ = sqrt(mean of client variances). Ignores support counts and between-client spread, so it does not preserve moments |
| Baseline sampling | `hierarchy.py:generate_synthetic_data` | `syn_samples_per_class` (50) per class, balanced class prior, global torch RNG |
| Edge → cloud support | `hierarchy.py:1097` | Counts are **synthetic sample counts** (50/class), not data support. MiSFP carries underlying data mass instead |
| Baseline extraction detail | `hierarchy.py:875` | Iterates the training loader, which has `drop_last=True` and shuffling. Up to `local_bs-1` random samples per client are silently left out of packets. MiSFP extracts from every local sample |
| Client feature coordinates | `_build_hierarchy`, `edge_server_aggregation` | In the legacy path each client trains a **private encoder copy**, averaged only every `t1` rounds among recently cached clients. Client packets are therefore **not guaranteed to be in a shared coordinate system** (see §6) |
| Validation | `get_data.py` | Without `val_from_train` the validation set **is** the test set. MiSFP configs set `val_from_train: 0.1` |
| Checkpoint/resume | `train_end_to_end` | H-SFP has no mid-run resume, and MiSFP does not add one (a full state would be ≈1.6 GB: 200 client copies plus Adam). Resumability is per run, through the manifest |

## 2. Representation

A packet holds, for each represented class c and each local component r:
`{class_id, component_id, count, mean[d], variance[d]}` plus a header with
the feature space, source, round and precision.

- Moments are float64 **population** statistics. Variances are stored
  **unfloored**, so pooling stays exact. Floors apply only when computing
  NLL or sampling.
- Component ids are local bookkeeping. Aggregation never matches components by id.
- `FeatureSpace(boundary, dim, feature_shape, space_id, model_version)`.
  Pooling refuses packets whose spaces differ. Matching metadata is a necessary
  guard; it does not prove the coordinates are aligned.

Mixture: q(z|c) = Σ_r π_cr N(μ_cr, diag v_cr), with π_cr = n_cr / Σ_s n_cs (empirical support).

Synthesis draws r ~ Cat(π_c), then z = μ_r + sqrt(max(v_r, ε))·ϵ with ϵ ~ N(0, I).
Every component keeps label c. The class prior stays H-SFP's balanced
`syn_samples_per_class`, separate from π. All generators are seeded per
(seed, purpose, round, node); see `misfp/sampling.py:derive_seed`.

## 3. Local construction (`misfp/fit.py`)

The fit is seeded k-means (k-means++ initialization, Lloyd iterations,
first-index tie breaking, empty-cluster reseeding). Exact empirical
population moments come from streaming **all** eligible features through the
chosen centers (Chan/Welford pooling, no E[x²]−μ² cancellation).

Edge cases:

- Non-finite rows are dropped and counted.
- An absent class produces no packet.
- K is capped at the number of distinct rows.
- Every component must have at least `misfp_min_component_support` (3)
  samples, otherwise K is reduced.
- Tiny and singleton classes fall back to K=1.
- A bounded reservoir (`misfp_reservoir_size`) is used only to fit centers.
  `info.fit_mode` records whether it was `exact` or `reservoir`.

**Adaptive K is a heuristic.** It works as follows:

1. Split the class's local *training* features into fit and validation parts
   (seeded, disjoint).
2. Fit K = 1..cap on the fit part.
3. Score each K as score(K) = NLL_val(K)/d + λ·(K−1).
4. Keep K=1 unless the best K improves its score by at least
   `misfp_adaptive_min_improvement` nats/dim.
5. Refit the chosen K on all local training features.

If there are too few samples for validation, it falls back to K=1 and logs
`insufficient_validation_support`. Test data is never used.

## 4. Hierarchical aggregation and compression (`misfp/merge.py`)

Merge cost (heuristic):

    W2²(a,b) = ||μa−μb||² + ||√va−√vb||²
    cost(a,b) = na·nb/(na+nb) · W2²(a,b)

The merge is exact moment pooling:

    n = na+nb
    μ = (na μa + nb μb)/n
    v = [na(va+(μa−μ)²) + nb(vb+(μb−μ)²)]/n

`compress_classes` applies limits in this order:

1. A **per-class cap**, using greedy smallest-cost merges.
2. A **serialized-byte budget**, using the same-class merge with the
   smallest cost per byte saved. Ties go to the lower class id, then the
   lower canonical pair.

At least one component per represented class is always kept. If that minimum
still does not fit, the status is `infeasible` (logged, or raised when
`misfp_on_infeasible: raise`). Classes are never dropped. Merge costs,
component counts and minority mass are logged per packet.

**Across a split boundary** the edge segment changes the feature space, so
input mixture parameters are never reused as output parameters. After the
edge trains on its synthetic inputs, it re-estimates statistics in its
**output** space in one of two modes:

- `propagate` (default): draw `misfp_boundary_samples_per_component` (50)
  samples from *each* input component in a separate pass, push them through
  the trained edge, and take empirical output moments per component.
  Component mass is the underlying data support, carried exactly, never the
  MC count. Gaussianity is a moment-matched approximation in the output space.
- `refit`: draw `misfp_boundary_samples_per_class` samples from the target
  mixture (no oversampling) and fit a new mixture (adaptive by default) in
  the output space. Component mass is an MC estimate: assignment fraction ×
  class support.

The edge→cloud packet is then compressed to `misfp_edge_out_cap` and the
edge budget. The cloud pools edge packets and synthesizes for CE training
with H-SFP's loop. Every compression and re-estimation is logged per round in
`misfp_rounds.jsonl`.

## 5. Variants (same method; `misfp/config.py`)

| Label | `misfp_variant` | Client K | Edge input | Edge→cloud cap | Notes |
|---|---|---|---|---|---|
| H-SFP (A) | `hsfp` | (1, H-SFP code) | H-SFP average | 1 | Unchanged `HierarchicalFL` |
| MiSFP-K1 (B) | `k1` | 1 | moment-pooled to 1 | 1 | New packet path |
| MiSFP-Edge (C) | `edge` | 1 | retain all (safety cap 32) | 4 | Across-client retention only |
| MiSFP-K2 (D) | `k2` | 2 (fixed, support-limited) | retain all | 4 | |
| MiSFP-K4 (E) | `k4` | 4 | retain all | 4 | |
| MiSFP-Adaptive (F) | `adaptive` | adaptive, cap 4 | retain all | 4 | |

There are two **matched-byte** configs: `*_fp16_matched` (float16 moments plus
`match_k1_fp32` budgets). Each client→edge and edge→cloud packet is capped,
**per packet per round**, at the byte size of the K=1 float32 packet for the
same classes, so cumulative totals are also ≤ K1. The full matrix also sweeps
`match_k1_fp32:1.5` and `:2.0`.

`misfp_pooling: baseline_average` reproduces H-SFP's aggregation rule on the
K=1 path. It exists only for regression comparison.

## 6. Representation compatibility

`misfp_extraction: shared_snapshot` is a diagnostic control. At the start of
each round it freezes one encoder snapshot: the FedAvg of the cached client
models, which is the same encoder validation grades. **Every** client extracts
its packet with that snapshot after its normal local training.

- **Different semantics:** packets no longer reflect this round's local
  updates.
- **Extra cost:** one model download per client is added to `edge_to_client_MB`
  and recorded as `snapshot_broadcast_MB`.

`between_source_dispersion` is logged per edge per round. It is the ratio of
between-client variance of class means to within-client variance. In IID data
a large ratio points to coordinate mismatch rather than data diversity. Compare
it between `local` and `shared_snapshot` runs.

## 7. Communication and resources

- MiSFP packets are actually serialized (`misfp/packets.py`), and receivers see
  the dequantized values.
- The inherited loop counts the tensor share of each packet, and MiSFP adds the
  remainder (header, ids, counts). Route totals in `comm_tracker` therefore
  equal the exact serialized bytes, tested in `test_transport_accounting_equals_serialized_bytes`.
- `misfp_summary.json` reports **prototype payload bytes** separately from
  **total training communication**, including H-SFP's model synchronization at
  t1/t2.
- H-SFP's own accounting counts only (mean, std) tensor bytes, with no counts
  or header.

Per-round timings (extraction, local fit, compression, edge merge, synthesis,
re-estimation, cloud merge/synthesis, snapshot) and MC sample counts are written
to `misfp_rounds.jsonl`, along with RSS, CUDA peak and the selected clients.
The run-level wall time and peak CUDA memory are in the summary.

## 8. Tests and diagnostics

```bash
python -m pytest tests/test_misfp_math.py -q        # 29 tests, CPU, ~5 s
python scripts/misfp/synthetic_diagnostic.py        # -> results/misfp/synthetic/
```

The tests cover:

- Moment preservation against concatenated data.
- Streamed moments.
- Sampling proportions and moments.
- Degenerate, NaN and duplicate inputs, and variance floors.
- Budgets, infeasibility and deterministic ties.
- Permutation and component-id invariance.
- Exact serialization byte counts.
- fp16 round-trip.
- Checkpoint/resume of packet and generator state.
- Feature-space refusal.
- Equality of K=1 with H-SFP's mean and population std.
- Reproduction of H-SFP's average pooling, and its documented difference from
  exact pooling.
- Variant and E-HSFP guards.
- Adaptive selection.

The synthetic diagnostic uses one class built from equal-weight N(−2, 0.1²) and
N(2, 0.1²), held by 4 clients that each run local K=2. It compares true
samples, the moment-matched single Gaussian, the retained 8-component pool, and
the pool W2-merged to 2 components. It reports held-out LL and the fraction of
synthetic points in the gap |x1| < 1. It demonstrates a failure mode only.

## 9. Running

```bash
# one run
python main.py --task classification --method misfp \
  --cfg configs/classification/misfp/cifar100_misfp_k2_resnet50_5_10.yaml --seed 0 \
  --set epochs=10 --set partition=dirichlet --set dirichlet_alpha=0.1

# manifest-driven, sequential, resumable
python scripts/misfp/experiments.py status   --matrix pilot
python scripts/misfp/experiments.py run      --matrix pilot
python scripts/misfp/experiments.py commands --matrix full   # prints; does not launch
```

Artifacts go to `$MISFP_OUT_ROOT` (default `/media/jackson/Data/misfp_runs`,
because the root disk is ~99% full), under `<matrix>/<run_id>/<arch>/<config_hash>/`.
Each run directory holds:

- `run_metadata.json`: resolved config, resolved MiSFP config, E-HSFP state,
  seed, partition hash, git commit, dirty paths and diff hash, versions.
- `client_partition.npz` and `client_to_edge.json`.
- `status.json`.
- `misfp_rounds.jsonl`.
- `misfp_summary.json`: test accuracy, macro-F1 and per-class metrics for the
  final round and the best-val checkpoint, communication, timings.
- The held-out diagnostic and PCA payload.

The manifests are `results/misfp/manifest_{pilot,full}.json`.

## 10. Pilot results

See `results/misfp/PILOT_REPORT.md` (generated by `scripts/misfp/report.py`
from completed runs only).
