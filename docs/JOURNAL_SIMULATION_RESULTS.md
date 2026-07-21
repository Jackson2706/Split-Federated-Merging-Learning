# E-HSFP Journal Simulation Results

**This is the single source of truth for the E-HSFP journal simulation campaign.** It is a
living document, updated continuously as audits, fixes, and runs land. Do not create
disconnected result reports — link everything from here.

---

## 1. Campaign Overview

- Repository: `/home/jackson/Desktop/Split-Federated-Merging-Learning`
- Current branch: `DungTT`
- Git commit at campaign start: `ba27944` (`fix(seg): resolve ISIC-2018 segmentation across all 4 methods, incl. H-SFP's NaN root cause`)
- Hardware: 1x NVIDIA GeForce RTX 3080 Ti, 12288 MiB VRAM, driver 570.195.03
- CUDA: 12.4 (torch-bundled)
- PyTorch: 2.6.0+cu124
- Python: 3.12.9
- **Note:** `.claude/CLAUDE.md` documents "CUDA 12.8, PyTorch 2.7" for install — the actual
  installed environment is CUDA 12.4 / PyTorch 2.6.0. Flagged, not changed (out of scope to fix
  silently; a version bump is a scientifically-relevant environment change).
- Campaign start date: 2026-07-20
- Last updated: 2026-07-20 (initial audit)
- Total experiment configurations required (per brief, all sections 17.1-17.13): **not yet
  fully enumerated** — registry construction in progress (Section 12). Rough scale: 4
  classification datasets x 8 methods, + 1 segmentation dataset x 8 methods, + 7 stress-family
  studies x multiple sweep points x up to 5 seeds each, + scalability to 500 clients. This is a
  multi-week-to-multi-month compute campaign on the available single-GPU hardware (see Section
  19 for time-budget reasoning).
- Total seed-level runs discovered from prior work: 44 marker-tracked runs (`scripts/journal_experiments`,
  seed=0 only) + 26 registry rows (`docs/optimization_loop/EXPERIMENT_REGISTRY.csv`, date-seeded,
  classification/CIFAR-100 proxy only) + 3-seed fair-comparison tables for CIFAR-100/HAM10000/ISIC-2018
  (`results/fair_comparison_*.csv`, date-seeded 20260714/15/16). See Section 4 for the full audit.
- Verified complete (this document's own audit, Section 4): see counts in Section 4 header.
- Running: 0 (no journal-campaign run is active as of last update)
- Missing: the large majority of the full requested grid — see Section 20.
- Failed: 4 confirmed-diagnosed failures from the prior `scripts/journal_experiments` pass (Section 4, Section 19).
- Blocked: `full_e_hsfp` (the complete proposed E-HSFP configuration) at proxy/full scale — unresolved
  autograd-backward deadlock, 5+ fix attempts across PLAN-11/13 and the 2026-07-16 DECISIONS.md entry,
  status unknown-but-likely-still-broken as of this writing (not re-attempted in this audit). This
  blocks the paper's headline "full E-HSFP" configuration across nearly every experiment family. See
  Section 19.

---

## 2. Status Definitions

- `VERIFIED_EXISTING` — a prior result was audited against the frozen protocol and found valid; reused as-is.
- `VERIFIED_NEW` — a new result was run under the frozen protocol, validated, and aggregated.
- `COMPLETE_UNVALIDATED` — a run finished (exit 0, marker present) but has not yet been checked against protocol/metric-validity rules in this document.
- `RUNNING` — currently executing.
- `PARTIAL` — some but not all required seeds/metrics are present.
- `MISSING` — not yet attempted.
- `INVALID` — exists, but fails a scientific-validity check (e.g. accuracy silently reported as F1, wrong dataset in config, known eval bug) and must not be used.
- `NOT_COMPARABLE` — exists and may be individually correct, but cannot be placed in the same table/comparison as sibling rows (different protocol, different accounting method, different seed convention).
- `FAILED_RETRYABLE` — crashed for a fixable reason (bug, OOM, config error); rerun once fixed.
- `FAILED_PERMANENT` — crashed for a structural reason that needs a design decision, not a quick fix (e.g. the full_e_hsfp deadlock).
- `BLOCKED` — cannot proceed until a dependency (data, implementation, fix) lands.

---

## 3. Frozen Experimental Protocol

**Protocol status: PARTIALLY FROZEN.** Values below are frozen where the prior codebase gives a
single, unambiguous, already-validated answer. Items with an explicit **[OPEN]** tag are decisions
this document is making now, per the instruction to make the safest scientifically valid choice,
document it, and continue — not blocking on user confirmation.

### 3.1 Seeds

**[OPEN — DECISION MADE]** The repository currently has **two incompatible seed conventions** in
live use, neither of which is a recovered "original ECCV conference" seed list:

1. **Date-based** (`20260714`, `20260715`, `20260716`) — used throughout `docs/optimization_loop/`
   (PLAN-0 through the 2026-07-20 ISIC fix), `EXPERIMENT_REGISTRY.csv`, and
   `results/fair_comparison_{cifar100,ham10000,isic2018}.csv`. These are literally the calendar
   dates the optimization loop ran (confirmed), not principled seed draws — 3 seeds total.
2. **Small-integer** (`0 1 2 3 4`) — the default in `scripts/journal_experiments/common.sh` and
   `scripts/camera_ready/`, used for the 160.5h prior journal-campaign pass (seed 0 only was
   actually executed).

**Decision:** per Section 13 of the brief ("When the previous conference seed list cannot be
recovered, use five fixed seeds: `[0, 1, 2, 3, 4]`") — no ECCV-original seed list is documented
anywhere in this repo, so this campaign adopts **`[0, 1, 2, 3, 4]`** (5 seeds) as the frozen
convention for all **new** journal-campaign results going forward, matching
`scripts/journal_experiments`'s own existing default. The date-based 3-seed
(`20260714/15/16`) prior results remain valid **on their own terms** for the specific tables
they already appear in (`results/fair_comparison_*.csv`, `RESULTS_SUMMARY.md`) but are treated as
a **separate, non-comparable protocol** — they must not be silently pooled with new `0-4`-seeded
runs in the same mean/std. Where a prior date-seeded result is reused in this document's tables
(Section 5+), it is explicitly marked `VERIFIED_EXISTING (date-seed protocol, n=3)` rather than
merged into a 5-seed statistic.

### 3.2 Dataset partitions

- IID: uniform random shard, `iid=True`.
- Mild non-IID: **[OPEN]** manuscript suggests Dirichlet `alpha=0.7`; **not found in any executed
  run in this repo** (only `alpha ∈ {0.05, 0.1, 1.0}` appear in historical camera-ready sweeps —
  see Section 21). Not yet verified against any manuscript text (no manuscript file was located
  in this repo to cross-check). **Frozen provisionally at alpha=0.7 per the brief**, pending
  manuscript access; flagged for confirmation before final publication tables.
- Severe non-IID: **[OPEN]** manuscript suggests `alpha=0.3`; same caveat — not found in any
  executed run. **Frozen provisionally at alpha=0.3 per the brief.**
- Historical Dirichlet values actually exercised in this repo: `alpha=0.1` (E-HSFP component
  ablation table, flat single-level), and the hierarchical two-level sweep
  `(alpha_edge, alpha_client) ∈ {(1.0,1.0),(1.0,0.1),(0.1,1.0),(0.1,0.1)}` plus a flat sweep at
  `alpha ∈ {0.05, 0.1}` (camera-ready partial-participation study). None of these is 0.7 or 0.3.
  New runs at the frozen 0.7/0.3 values are required (Section 20).
- Partition function: `two_level_dirichlet` when a client→edge Dirichlet mapping is needed (camera-ready
  path), else flat `dirichlet`. **Caveat (confirmed in code):** the legacy non-Dirichlet
  `cifar_noniid`/`mnist_noniid` sharding path assumes ≤100 clients and is force-disabled
  (`iid=True` placeholder swap) whenever `partition` is `dirichlet`/`two_level_dirichlet` — so all
  non-IID scalability runs (Section 17.12) must go through the Dirichlet path, never the legacy
  shard path.

### 3.3 Client and edge topology

- Client count: `num_users` — default **200** (classification), **50** (segmentation). Configurable
  via `--set num_users=N`.
- Edge count: `mid_server` (a list; `mid_server[0]` = edge count) — default **`[5]`** (5 edges),
  used in every historical H-SFP/HierFL/HSFL run found in this repo.
- Client→edge mapping: contiguous random split (`num_users // num_edges`, remainder to last edge)
  by default, or explicit Dirichlet-drawn mapping when `two_level_dirichlet` partitioning supplies
  one.
- **[GOTCHA — documented for scalability campaign]** `mid_server[0]` and the partition script's own
  `num_edges` argument are two independently-read config keys. If only one is overridden via
  `--set` when scaling client count, the built hierarchy and the partition can disagree on edge
  count. **Frozen rule for Section 17.12 scalability configs: always set both `mid_server=[E]` and
  `num_edges=E` explicitly together, or leave `num_edges` unset so it inherits `mid_server[0]`.**

### 3.4 Model backbones and split points

- **[CRITICAL — documented, not yet fixed]** Every classification config labeled `model: resnet50`
  for **CIFAR-10/CIFAR-100** actually instantiates a **modified ResNet-18** (torchvision ResNet-18
  stem+layer1+layer2+GAP, ImageNet `DEFAULT` weights, CIFAR-sized conv1 replaced) — confirmed by
  direct forward-pass audit in `docs/optimization_loop/PROJECT_STATE.md` (PLAN-0). **HAM10000 and
  ISIC-2018 configs labeled `resnet50` are genuine ResNet-50.** This is a real config-naming bug
  affecting Federated/HierFL/HSFL/SplitFL/HeteroSFL/H-SFP identically for CIFAR — relative
  cross-method comparisons on CIFAR remain valid (everyone gets the same wrong backbone), but any
  manuscript claim of "ResNet-50 on CIFAR" is factually incorrect and must be corrected in the
  paper text (out of scope for this document to silently rewrite training code mid-campaign — flag
  for a deliberate, separately-reviewed fix + protocol-version bump if the manuscript claim must be
  literally true).
- Split points (from `PROJECT_STATE.md` forward-pass audit):
  - Classification CIFAR: client → `[1,3,32,32]→[1,128,1,1]`, edge → `[1,256,1,1]`, cloud → `[1,256]→[1,100]`.
  - Classification HAM10000: client → `[1,3,224,224]→[1,256,56,56]`, edge → `[1,2048,1,1]`, cloud → `[1,2048]→[1,7]`.
  - Segmentation ISIC: client → `[1,3,224,224]→[1,256,56,56]`, edge → `[1,2048,7,7]`, prototype-cloud → `[1,2048]→[1,2]`, decoder → `[1,2048,7,7]→[1,1,224,224]`.

### 3.5 Training hyperparameters

- Optimizer: Adam, LR `1e-4`, weight decay `1e-4` (H-SFP/E-HSFP family). Classification adds a
  5-round linear warmup then cosine schedule; segmentation has no scheduler.
- **[INCONSISTENCY — documented]** `classification/SplitFL` and `classification/HeteroSFL` runners
  hard-code SGD (momentum 0.9, no weight decay/scheduler) regardless of `optimizer: adam` in their
  YAML configs (per `docs/optimization_loop/FAIR_COMPARISON_CIFAR100.md`). This is a genuine
  cross-method protocol inconsistency, not just a naming issue — logged in Section 21, fix owed
  before any optimizer-sensitive comparison is treated as final.
- Reliability-network bootstrap: Adam, LR `1e-3`, weight decay `1e-4`.
- Batch sizes / local epochs: per-method `default.yaml` (not yet centrally re-verified across all
  6 method families in this pass — TODO for the registry-construction step, Section 12).
- Global rounds: **60** (full/paper scale), **10** (proxy scale, for cheap iteration only — proxy
  numbers are explicitly flagged non-final everywhere they're used, see Section 3.9 test-leakage note),
  **2** (smoke gate only, never a reportable number).

### 3.6 Aggregation intervals

- Historical sweep values (from `scripts/journal_experiments/run_intervals.sh` and ISIC proxy configs):
  `(client_edge=5, edge_cloud=10)`, `(10, 20)`, `(25, 50)`.
- **[Section 17.13 requirement — not yet resolved]** The brief calls out an existing inconsistency
  where H-SFP accuracy corresponds to interval A but a communication/VRAM figure (10.94 GB) came
  from interval C. This document does not yet have enough audit depth to say which historical
  number used which interval — **treat any existing communication/memory figure as
  `NOT_COMPARABLE` to any existing accuracy figure until both are re-derived from the same run**
  (same experiment_id, same seed). Do not combine them. This is enforced going forward by the
  registry (Section 12), which records interval as a first-class field per experiment.

### 3.7 E-HSFP module parameters

Frozen from `ehsfp/config.py` (`EHSFP_DEFAULTS`, verified by direct code read):

`memory_size=500`, `memory_replay_ratio=0.3`, `memory_top_k=5`, `max_prototype_age=20`,
`reliability_hidden_dim=32`, `reliability_lr=1e-3`, `reliability_weight_decay=1e-4`,
`lambda_prc=0.1`, `prc_num_samples=16`, `prototype_dropout_rate=0.2`,
`dropout_mode="client_prototype"`, `use_dropout_consistency=False`, `lambda_dropout=0.1`,
`cold_start_probability=0.15`, `function_timeout_probability=0.05`, `max_episode_duration=30.0`,
`latency_mean=0.5`, `latency_std=0.2`, `generator_hidden_dim=64`, `generator_scale=0.1`.

Ablation presets (each overrides only the listed keys):

| Preset | memory | aggregation | PRC | dropout | serverless_sim | residual_gen |
|---|---|---|---|---|---|---|
| `baseline_hsfp` | off | average | off | off | off | off |
| `hsfp_memory` | **on** | average | off | off | off | off |
| `hsfp_memory_dropout` | on | average | off | **on** | **on** | off |
| `hsfp_memory_reliability` | on | **learnable_reliability** | off | off | off | off |
| `hsfp_memory_reliability_prc` | on | learnable_reliability | **on** | off | off | off |
| `full_e_hsfp` | on | learnable_reliability | on | on | on | **on** |

Resolution order (confirmed): YAML `base` chain → child YAML → CLI `--set` overrides → E-HSFP
defaults → resolved-YAML E-HSFP values → named ablation preset (**highest precedence**).

**Known-broken component:** `use_residual_generator=True` (only set by `full_e_hsfp`) is
documented in `PROJECT_STATE.md` as an unresolved anomaly — both hierarchy constructors leave
`residual_generator = None` regardless, so the residual generator is **not actually wired in**
even when the preset requests it. `use_dropout_consistency`/`dropout_consistency_loss` is
similarly defined but never invoked by either hierarchy training loop. Both need Codex
implementation work before Section 17.11 (prototype-synthesis ablation, which explicitly
requires "residual generator + consistency loss") can produce a real result.

### 3.8 Stress-protocol parameters

`ServerlessMetricsTracker` config fields exist (`cold_start_probability=0.15`,
`function_timeout_probability=0.05`, `max_episode_duration=30.0`, `latency_mean=0.5`,
`latency_std=0.2`) but this document has **not yet verified** exact simulation semantics (does
`function_timeout_probability` apply per-client-per-round? per-edge? what happens to a timed-out
client's contribution?) — deferred to the pending E-HSFP-component audit agent, to be added on
next update. Partial-execution and missing-edge-update parameter names have not yet been located
in code; **MISSING** pending that audit.

### 3.9 Metric definitions

**Classification:**
- Accuracy: `sklearn.metrics.accuracy_score` — implemented correctly across all methods.
- Macro F1: **[CRITICAL FINDING]** genuinely computed (`sklearn.metrics.f1_score(..., average="macro")`)
  in `classification/Federated`, `classification/HierFL`, `classification/HeteroSFL`. **NOT
  genuinely computed** in `classification/H-SFP`, `classification/HSFL`, `classification/SplitFL`
  — all three instead compute `accuracy_score` and store it under an `"f1"`/`"final_test_f1"` key
  (H-SFP: `hierarchy.py:1125-1129`, comment admits "holds accuracy; name kept for output-key
  compatibility"; HSFL: `hierarchy.py:326`; SplitFL: `runner.py:156,195`). A real `evaluate_model`
  macro-F1 function exists for H-SFP (`classification/H-SFP/evaluation/evaluator.py:47`) but is
  **dead code** — never imported by the training/runner path. **Rule enforced in this campaign:
  any existing "F1"/"final_test_f1" value for H-SFP, HSFL, or SplitFL classification is `INVALID`
  as an F1 metric** (it may be reused as an accuracy value if relabeled, but never plotted/tabled
  as F1) until real macro-F1 is wired in and the run is redone or reprocessed from saved
  predictions.

  **[FIXED, 2026-07-20]** Dispatched Codex to implement genuine `sklearn.metrics.f1_score(...,
  average="macro", zero_division=0)` in all three files, reviewed the diff, and confirmed
  checkpoint/best-model selection was deliberately left accuracy-driven in all three (not silently
  switched to F1-driven, which would have been an undocumented methodology change). New output
  keys: `best_val_top1`/`validation_accuracy`/`accuracy` for honest accuracy, `best_f1`/
  `validation_f1`/`final_test_f1` now genuine macro-F1. A CPU-only test
  (`tests/test_classification_metric_reporting.py`, passing) proves the two metrics now diverge on
  an imbalanced synthetic example (accuracy 0.90 vs. macro-F1 0.316). **Found and fixed one
  follow-up bug during review**: `classification/H-SFP/runner.py:116,120` still printed
  `output["best_f1"]` (now genuine F1) under the label "Best Validation Acc" — would have
  re-introduced the same mislabeling one layer up, and would have fed a mislabeled value into this
  session's own `collect_results.py` fix. Corrected to print `output["best_val_top1"]`.
  **GPU-verified, 2026-07-21** (`outputs/verify/f1_fix_hsfp/`, 2-epoch CIFAR-100 smoke run):
  `metrics.json` shows `validation_accuracy: [4.96%, 5.84%]` vs. genuine `validation_f1: [2.92%,
  4.41%]` — meaningfully distinct, independently computed values, and the console print now
  correctly labels the accuracy value. **Status: `VERIFIED_NEW`.** Historical JSON/log artifacts predate
  this fix and remain mislabeled; only newly generated runs are affected.

**Segmentation:**
- Dice and IoU: identical, correct formula across H-SFP/Federated/HierFL/HeteroSFL
  (`compute_iou_and_dice`, duplicated per-method in `clients/test.py`).
  `Dice = 2·TP/(2·TP+FP+FN)`, `IoU = TP/(TP+FP+FN)`.
- **Segmentation F1 == Dice, exactly**, by construction (Dice is the binary-F1 formula). This
  campaign will **not** report Dice and "segmentation F1" as two independent numbers — they are
  the same statistic. State this explicitly in any table/figure.
- **[CRITICAL FINDING] `segmentation/HSFL` does not perform segmentation.** Its `_validate` is a
  byte-for-byte copy of the classification path (`logits.argmax` + macro-F1 against a target
  vector); its models end in `nn.Linear(256, num_classes)` and never produce a spatial mask, even
  though it's fed `ISICSegmentationDataset` image/mask pairs. Any existing "segmentation HSFL"
  number is meaningless. **Status: `NOT_IMPLEMENTED`, needs a genuine reimplementation** (real
  decoder + Dice/IoU eval), not a rerun or a metric fix. Also: `segmentation/SplitFL` and
  `segmentation/HSFL` both have working ISIC data loaders but **no valid ISIC config** — the
  `configs/segmentation/{splitfl,hsfl}/*.yaml` files present are stray copy-pasted
  CIFAR/HAM10000-valued configs that would crash the ISIC loader if run as-is.

**Communication:** **[FIXED, 2026-07-21]** Every method family (Federated, HierFL, HeteroSFL,
SplitFL, HSFL, H-SFP; both classification and segmentation) now reports into one shared,
route-based schema (`ehsfp/communication.py`): `client_to_server_MB`/`server_to_client_MB` for
flat methods, `client_to_edge_MB`/`edge_to_client_MB`/`edge_to_cloud_MB`/`cloud_to_edge_MB` for
hierarchical methods, all summing into a backward-compatible `total_comm_MB`. Byte counts use
exact `numel() * element_size()` (previously several methods hard-coded `* 4` assuming float32,
undercounting/overcounting for other dtypes). Along the way, several real bugs were found and
fixed, not just relabeled: **HeteroSFL's `download_MB` was never incremented at all** (activation-
gradient return and model distribution were completely untracked — its true communication was far
higher than previously reported); **SplitFL's gradient-size estimate used the server model's
output size instead of the actual cut-tensor gradient shape**; **Federated/SplitFL/HierFL charged
model downlink to every configured client instead of only the selected ones**; several methods
used decimal MB (`/1e6`) inconsistently with others' binary MiB (`/1024²`). Reviewed by Claude
(module design, 2 file diffs read in full, independent test execution, GPU smoke-verified on both
a flat method [HeteroSFL] and a hierarchical method [H-SFP] — correct schema, no crashes, no
training-behavior changes). **The cross-method communication table (Section 17.13) is now valid
to produce** once the underlying experiments are (re)run with this code — existing/historical
communication numbers in already-completed runs predate this fix and are not comparable to new
runs' numbers.

**Memory:** GPU peak memory via `torch.cuda.max_memory_allocated()` in most runners, but
**SplitFL uses `torch.cuda.memory_allocated()` sampled-and-maxed** (current allocation, not the
true CUDA peak-allocator statistic) — a different measurement semantics from every other method.
Only `HSFL` (classification and segmentation) genuinely separates memory by tier
(client/edge/cloud); every other method reports one pooled global number. No true CPU/host
peak-RAM is tracked anywhere (only `psutil`-based *averages*).

**Training time:** wall-clock via `time.time()`, generally per-run total; H-SFP additionally
tracks true per-epoch time internally but **does not persist a `runtime_s` field to
`metrics.json`/`run_metadata.json` at all** — confirmed gap. Any H-SFP row's `runtime_s` must be
recovered from the raw log (regex fallback), not assumed present in the structured output.

**Stability / rounds-to-convergence / prototype drift / stability-bound term / recovery gap /
prototype fidelity:** **[CRITICAL FINDING] none of these six metrics are implemented anywhere in
this repository** (verified by exhaustive grep across the whole tree including `ehsfp/`) — not
approximable from existing logs, because the underlying structured signal (e.g. a prototype
snapshot history to diff for drift) was never captured. All six require ground-up implementation
with an explicitly frozen definition before Sections 7, 9, 10, 12 of the campaign can produce any
numbers. This is the single largest implementation gap blocking the campaign and is queued as a
priority Codex task (Section 12/20).

**[FROZEN DEFINITIONS, 2026-07-21]** No manuscript file exists anywhere in this repository, so
"the journal definition" referenced by the brief for stability/drift cannot be copied from source
— these are operational definitions proposed here (the safest scientifically valid choice per the
brief's own instruction to make such calls and document them, not block on it). 5 of 6 are defined
below; the 6th (stability-bound term) is explicitly left undefined — see note.

- **Stability**: trailing-window variance of the validation metric (accuracy for classification,
  IoU for segmentation) over the last **W = 5 rounds**: `stability(t) = Var(metric[t-W+1 : t+1])`
  (population variance, `ddof=0`, matching the convention already used elsewhere in this repo's
  seed-aggregation code). Lower is more stable. Undefined for `t < W`; report `null`/`N/A` for
  those early rounds rather than a value computed on a partial window. **W=5 is frozen** — record
  the window length with every reported value, as the brief requires.
- **Rounds to convergence**: per the brief's own recommended default, applied literally: (1)
  smooth the validation metric with a trailing moving average, **smoothing window = 3 rounds**;
  (2) `threshold = 0.95 * best_validation_metric` (best-so-far, not final-round, since "best" is
  already the quantity this repo's checkpoint selection optimizes for and is more robust to a
  late-training collapse than "final"); (3) the convergence round is the first round `t` where the
  smoothed metric ≥ threshold; (4) require the smoothed metric to remain ≥ threshold for a
  **patience window = 3 rounds** afterward, else keep scanning forward from the next candidate
  round. If the threshold is never sustained, report `not_converged` (never silently report the
  best epoch as if it were a convergence point, and never report the run's max round number as a
  disguised "not converged" value — a run collapsing right at the end, like the pre-fix HAM10000
  runs did, must show `not_converged`, not the last passing round). **Smoothing window (3) and
  patience window (3) are frozen** alongside the threshold fraction (0.95).
- **Prototype drift**: round-over-round L2 distance between a class's prototype at round `t` and
  the same class's prototype at round `t-1`, computed separately per tier (client/edge/cloud) and
  per class: `drift(t, class, tier) = ||proto_t[class] - proto_{t-1}[class]||_2`. Requires an
  explicit prototype history (current+previous round only, not a full log, to bound memory) that
  does not exist today — this is new state to add, not a metric computable from existing logs.
  **Global drift** (per round) = mean over all classes and all tiers present that round.
  **Per-tier drift** = mean over classes within one tier. **Per-class drift** = the raw per-
  class/per-tier value above. **Max drift** = max over classes (report per tier and globally).
  The very first round has no previous-round prototype to diff against — report `N/A` for round 1,
  not zero (zero would falsely imply "no drift," when the real answer is "undefined").
- **Recovery gap**: exactly the brief's recommended default —
  `recovery_gap = clean_final_metric - stressed_final_metric` (both "final" meaning the best-
  validation-checkpoint metric, for consistency with how every other reported number in this
  campaign is selected). **Recovery rounds** (reported separately, per the brief): under a
  stress condition that is later lifted (e.g. a temporary dropout/staleness/serverless-stress
  window), the number of rounds after stress lifts before the metric returns to within
  `1 - 0.95 = 5%` of its pre-stress value — reuses the same 0.95 threshold fraction as
  rounds-to-convergence for consistency, not a new free parameter.
- **Prototype fidelity**: **freeze MMD** (`rbf_mmd`, already implemented in
  `camera_ready/feature_distance.py`) as the single primary fidelity metric, per the brief's
  instruction to pick one and use it for every synthesis variant. Chosen over Frechet/Wasserstein/
  covariance-reconstruction-error because it is the only one of the four with a working,
  already-debugged implementation in this repo (Frechet has zero call sites anywhere and has never
  been exercised — Section 4.5). Currently only reachable from the offline
  `scripts/camera_ready/run_covariance.py` script; needs wiring into the live training path
  (comparing each round's synthetic prototype-space samples against the real client-encoder
  features they're meant to approximate) to produce a genuine per-round fidelity number, not just
  an end-of-training offline analysis.
**[IMPLEMENTED AND GPU-VERIFIED, 2026-07-21]** Codex implemented all 5 defined metrics as pure,
tested functions in `ehsfp/research_metrics.py` (8 new unit tests, all independently re-run and
passing, including the tricky "crosses threshold then drops back before patience completes"
edge case for rounds-to-convergence). Stability and rounds-to-convergence are wired into both
`classification/H-SFP/hierarchy.py` and `segmentation/H-SFP/hierarchy.py`'s output (new
`stability_per_round`, `stability_window`, `rounds_to_convergence`,
`rounds_to_convergence_definition` fields — the definition string is embedded in every run's
output so results are self-documenting). Prototype drift and the MMD fidelity wrapper are
implemented as reusable functions but intentionally not yet wired into the live training loop
(correctly scoped as a stretch goal, not half-done). Reviewed by Claude (diff read in full,
confirmed no training-behavior changes, no stray stability-bound-term stub anywhere) and
GPU-verified with a 3-epoch H-SFP smoke run: `stability_per_round=[None,None,None]` (correctly
undefined — window=5 needs 5 rounds) and `rounds_to_convergence="not_converged"` (correctly —
patience=3 cannot be satisfied within 3 total rounds), proving the implementation reports honest
"not enough data" rather than fabricating early values.

- **Stability-bound term: left undefined, marked `BLOCKED_NO_THEORY`, not implemented.** The brief
  asks for "the complete per-round term and every individual component used in its calculation" —
  this is necessarily a specific closed-form expression from the paper's theoretical
  analysis (e.g. a PAC-Bayes-style or Lipschitz-based bound on aggregation error), which does not
  exist in any form anywhere in this repository and cannot be reconstructed from code or
  conventions the way the other five metrics above could be. Fabricating a plausible-looking bound
  formula would violate the brief's own rule against inventing results (Section 10.1) — a made-up
  theoretical expression is exactly the kind of thing that rule exists to prevent, even though it's
  a formula rather than a number. This is a genuine, standing blocker pending either the actual
  manuscript text or explicit user-supplied formula; not something to proceed past by guessing.

### 3.10 Hardware and software environment

See Section 1. Recorded once per campaign; re-record if hardware changes mid-campaign.

---

## 4. Existing Simulation Audit

**Machine-readable companions to this section:**
[`experiments/journal_experiment_registry.yaml`](../experiments/journal_experiment_registry.yaml) ·
[`results/journal/numeric/SEED_LEVEL_RESULTS.csv`](../results/journal/numeric/SEED_LEVEL_RESULTS.csv) ·
[`results/journal/numeric/AGGREGATED_RESULTS.csv`](../results/journal/numeric/AGGREGATED_RESULTS.csv) ·
[`results/journal/numeric/NUMERIC_DATA_MANIFEST.csv`](../results/journal/numeric/NUMERIC_DATA_MANIFEST.csv) ·
[`results/journal/manifests/RUN_MANIFEST.csv`](../results/journal/manifests/RUN_MANIFEST.csv) ·
[`results/journal/manifests/RUN_FAILURES.csv`](../results/journal/manifests/RUN_FAILURES.csv) ·
[`results/journal/manifests/existing_result_audit.csv`](../results/journal/manifests/existing_result_audit.csv) ·
[`results/journal/manifests/result_provenance.json`](../results/journal/manifests/result_provenance.json)

Legend for **Protocol match**: `MATCH` (frozen protocol as defined in Section 3),
`DATE-SEED-PROTOCOL` (valid 3-seed date-based protocol, not directly poolable with new 5-seed
runs per Section 3.1), `PRE-EVALFIX` (predates the 2026-07-14 PLAN-9 eval-bug fix — numbers are
`INVALID`), `PARTIAL-CONFIG` (config/round-budget was wrong at run time, later fixed).

This table covers the three provenance systems found in this repo. It is not yet exhaustive (the
raw-artifact-inventory audit agent has not yet reported — Section 23 change log will note when
this table is extended).

### 4.1 `scripts/journal_experiments/` marker system (seed=0 only, 160.5h wall time, 2026-07-09 to 2026-07-14)

44 experiments attempted: 40 `.done`, 4 `.failed`. **`collect_results.py`'s metric parser was
broken (regex/format mismatch) and has since been fixed by Codex this session** — verified
independently (tests pass, spot-checked against 3 real logs including a known-crashed one, no
false positives). Real metrics are now extracted into `results/journal/summary_*.csv`. A
follow-up refinement is still open: `Federated`/`HierFL`/`HeteroSFL` logs print genuine per-epoch
macro-F1 (`Loss: X F1: Y%`, confirmed real macro-F1 per Section 3.9) that the current parser does
not yet capture (only the final Test Acc line) — low-priority, queued.

**[RESOLVED — root cause found, 2026-07-20]** With real metrics extracted, `results/journal/summary_all.csv`
showed **every one of the 6 ablation presets returning bit-identical accuracy** (exactly 8.1% for
all 6 CIFAR variants, exactly 10.98% for all 6 HAM10000 variants) despite genuinely different
E-HSFP components supposedly being enabled per preset. Dispatched Codex to trace the full config
chain (`main.py` → `ConfigLoader` → `get_ehsfp_config()` → `HierarchicalFL.__init__` → training
loop). **Root cause (>99% confidence, independently reproduced with a CPU-only proof script,
`tests/prove_ehsfp_config_resolution.py`):** this entire 2026-07-09–07-14 journal campaign ran
under the **pre-`cf6454c`** version of `ehsfp/config.py`'s `get_ehsfp_config()`, which applied the
inherited YAML defaults *after* the ablation preset — and since `default.yaml` explicitly sets
every E-HSFP flag to `false`, those defaults silently overwrote every preset's intended settings,
collapsing all 6 presets to identical (all-off) behavior. Only `ablation_mode` (a metadata string,
not a computational flag) survived — exactly matching the observed bit-identical logs. **This bug
is already fixed** in the current tree (commit `cf6454c`, 2026-07-16: defaults load first, preset
applies last) — the proof script confirms the current code resolves genuinely distinct configs
per preset. **Independently confirmed empirically by Claude** via 2-epoch GPU smoke reruns on the
identical config/seed (`outputs/verify/abl_fix_{baseline,memory}`): `baseline_hsfp` got 5.84% Best
Validation Acc, `hsfp_memory` got 5.71% — genuinely different, with differing edge-SSL losses
(1.15 vs 1.05, 1.35 vs 1.03, 1.26 vs 1.03, etc.) and different prediction distributions (PRC-bearing
presets deliberately not touched in this check, to avoid the separately-tracked autograd deadlock).
**This was already partially known**: `docs/optimization_loop/RESULTS_SUMMARY.md`'s
PLAN-0 section already flags "the discovered ablation-collapse and PRC dead-path defects" as the
reason the ~8.10%/~10.98% numbers are "UNVERIFIED and not attributable" — this session's work
independently reconfirms that note with a precise mechanism and a passing regression proof.
**Practical consequence**: every non-baseline-preset H-SFP/E-HSFP run in the entire
`scripts/journal_experiments` 2026-07-09–07-14 campaign (all `abl_*` non-baseline rows, all
`conv_ehsfp_*`/`intv_ehsfp_*` rows) is `INVALID` and must be rerun with current code — reclassified
below. Plain `baseline_hsfp`-only runs and non-H-SFP-method runs (`conv_fed_*`, `conv_hierfl_*`,
etc.) are unaffected by this specific bug (baseline never had any flags to lose) and remain at
whatever validation status they'd otherwise have. Rerun tracked as a new registry task (Section 20).

| Group | Count done | Count failed | Dataset(s) | Methods | Validation status | Reuse decision |
|---|---:|---:|---|---|---|---|
| convergence, non-H-SFP (`conv_fed_*`, `conv_hierfl_*`, `conv_hsfl_*`, `conv_splitfl_*`, `conv_heterosfl_*`) | 16 | 2 | CIFAR(-100), HAM10000 | FedAvg/Prox/Nova/SGD, HierFL, HSFL, SplitFL, HeteroSFL | `COMPLETE_UNVALIDATED` — real metrics now extracted (`collect_results.py` fixed); F1-validity issue (Section 3.9) still applies to HSFL/SplitFL rows | Reuse accuracy values; do not reuse any "F1" value without independent recomputation |
| convergence, H-SFP baseline-only (`conv_hsfp_*` with no ablation, i.e. implicit baseline) | 4 | 0 | CIFAR, HAM10000 | H-SFP baseline | `COMPLETE_UNVALIDATED` — unaffected by the ablation-collapse bug below (baseline had nothing to lose) | Reuse as baseline H-SFP reference points |
| convergence, E-HSFP full preset (`conv_ehsfp_*`) | 4 | 0 | CIFAR, HAM10000 | H-SFP `full_e_hsfp` | **`INVALID`** — ablation-collapse bug (see finding above); these ran with all E-HSFP components silently disabled despite the `full_e_hsfp` label | **Rerun required** with current code (Task in Section 20) |
| ablation (`abl_*`) | 12 | 0 | CIFAR, HAM10000 | H-SFP 6-preset ladder | `COMPLETE_UNVALIDATED` for the `baseline_hsfp` rows (2 of 12); **`INVALID`** for the other 10 (all non-baseline presets collapsed to baseline behavior) | Rerun the 10 non-baseline rows with current code. **CIFAR-100 3-preset rerun COMPLETE** (`docs/optimization_loop/logs/plan37_ablation_rerun/`, all `VERIFIED_NEW`, single seed s0, 60 rounds each):

  | Preset | Best Val/Test Acc | total_comm_MB | Runtime (s) |
  |---|---:|---:|---:|
  | `baseline_hsfp` (unaffected by the collapse bug) | 8.10% | 30414.23 | ~25,000 |
  | `hsfp_memory` | **11.28%** | 1430.47 | 6489 |
  | `hsfp_memory_dropout` | **11.20%** | 1426.39 | 6175 |
  | `hsfp_memory_reliability` | **11.26%** | 1430.47 | 6820 |

  Clean, scientifically sensible pattern: memory alone gives a real +3.18pp jump over baseline
  (not a rounding artifact — this is a completely different training trajectory, not the old
  bit-identical collapse). Dropout and reliability each produce small secondary effects on top of
  memory, in the same *direction* as the earlier (valid, never-collapsed) 10-round proxy ablation
  table in `RESULTS_SUMMARY.md` (memory 10.26%→+dropout 9.89%, a decrease; here 11.28%→11.20%,
  also a decrease) — good cross-protocol consistency evidence that this new data is trustworthy.
  Also note: `total_comm_MB` collapses from ~30,414 (baseline) to ~1,430 (any memory-enabled
  preset) — an artifact of `t1`/`t2` aggregation-interval accounting interacting with the memory
  path, not yet fully explained; flagged for the communication-accounting normalization task
  (Section 3.9) rather than assumed correct.

  **HAM10000 rerun: 2/3 complete, 1 crashed with a new confirmed bug** (`docs/optimization_loop/logs/plan38_ablation_rerun_ham/`):

  | Preset | Result | Notes |
  |---|---|---|
  | `hsfp_memory` | 10.98%, best epoch **1/60** | Accuracy peaked immediately then never improved for the remaining 59 rounds; epoch-57 debug output showed all validation predictions collapsed to class 0 |
  | `hsfp_memory_dropout` | 10.98%, best epoch **1/60** | Identical pattern — **zero** "Saved best model" events after epoch 1 across all 60 rounds, confirming this isn't noise |
  | `hsfp_memory_reliability` | **CRASHED**, runtime 186s (epoch 1) | `ValueError: Out of range float values are not JSON compliant: nan` in the `reliability.weight_variance_sum` runtime counter — a **new, distinct, currently-live bug**, not a recurrence of the ablation-precedence bug or the earlier fp32-SupCon fix |

  The identical 10.98%/epoch-1 result for the first two presets is now well-evidenced as expected
  behavior, not a bug: no E-HSFP component (memory, dropout) has any accumulated history at epoch
  1, so all presets are necessarily identical there by construction — the real issue is a
  HAM10000-specific training collapse after epoch 1 (task: investigate, Section 20), separate from
  the (confirmed-fixed, confirmed-working-on-CIFAR) ablation-precedence bug.

  **[FIXED AND VERIFIED, 2026-07-21]** The `hsfp_memory_reliability` crash was exactly the
  hypothesis: `classification/H-SFP/hierarchy.py`'s client-side prototype extraction
  (`calculate_prototypes_and_distribution` and the client SSL extraction loop) computed
  `mean`/`std` on fp16 autocast output — the identical bug class already fixed in
  `segmentation/H-SFP/hierarchy.py` earlier this session, but never ported to the separate
  classification codebase. Squaring large fp16 activations for the variance overflowed fp16's
  range, producing non-finite prototype sigmas that fed `build_reliability_features()`'s
  `sigma_magnitude` feature, through `Sigmoid` (NaN in → NaN out — the network's bounded-output
  range doesn't help if its *input* is already NaN), into the `weight_variance_sum` runtime
  counter. Codex fixed the actual numerical root cause (`.float()` immediately after the autocast
  forward, matching the segmentation fix's pattern) rather than a `nan_to_num` band-aid.
  **GPU-verified**: the exact previously-crashing command
  (`--ablation hsfp_memory_reliability` on HAM10000) now completes cleanly end-to-end
  (`outputs/verify/reliability_nan_fix/`, 2 epochs, exit 0, Best Validation Acc 8.54%, 6 distinct
  predicted classes — no collapse to a single class in this short run). Full 60-round rerun
  relaunched to complete the ablation campaign.

  **HAM10000 `hsfp_memory_reliability` 60-round rerun complete: 55.82%**, with genuine
  progressive learning — best model updated at epochs 1→15→16→20→29→56
  (8.54%→12.43%→34.80%→42.59%→52.77%→55.82%), no collapse. This is dramatically different from
  `hsfp_memory`/`hsfp_memory_dropout`'s 10.98%-stuck-at-epoch-1 results (Section above). **Because
  the fp16-prototype-std fix touches shared client-side extraction code used by every preset**,
  not just the reliability path, this strongly suggests those two presets' collapse was caused by
  the same silent numerical corruption (degrading training quality without a hard crash) rather
  than a genuine, separate HAM10000-specific pathology — i.e. task #16's "training collapse" and
  task #17's "NaN crash" likely share one root cause. **Rerunning `hsfp_memory` and
  `hsfp_memory_dropout` on HAM10000 with the fix applied** (`docs/optimization_loop/logs/plan39_ablation_rerun_ham_v2/`)
  to confirm before updating their status from `PARTIAL`/stale to `VERIFIED_NEW`.

  **Fp16-fix hypothesis CONFIRMED, HAM10000 3-preset ablation rerun COMPLETE**
  (`docs/optimization_loop/logs/plan39_ablation_rerun_ham_v2/`):

  | Preset | Best Val/Test Acc | Best epoch | total_comm_MB |
  |---|---:|---:|---:|
  | `hsfp_memory` | **54.72%** | 27/60 | 16063.54 |
  | `hsfp_memory_dropout` | **50.77%** | 42/60 | 15473.14 |
  | `hsfp_memory_reliability` | **55.82%** | 56/60 | 16063.54 |

  All three now show genuine progressive learning (multiple best-model updates spread across the
  full 60 rounds, not stuck at epoch 1) — a dramatic, clean contrast with the pre-fix results
  (all three stuck at 10.98%, best epoch 1). The training-collapse finding (task #16) and the
  NaN-crash finding (task #17) are now confirmed to share one root cause — the fp16
  prototype-std overflow bug in `classification/H-SFP/hierarchy.py`, fixed this session — and both
  are resolved by the same fix. Task #14 (CIFAR + HAM10000 3-preset ablation rerun) and task #16
  are complete.

  **Remaining for the ablation study**: the 2 PRC-bearing presets on both datasets, blocked on the
  autograd deadlock (Section 19, still the single highest-priority open item); extending from
  single-seed (s0) to the full 5-seed protocol (Section 3.1) once the deadlock is resolved and
  scope/budget for the larger campaign is confirmed.
  (`hsfp_memory_reliability_prc`, `full_e_hsfp`) for both datasets — blocked on the autograd
  deadlock (Section 19), tracked separately. |
| intervals (`intv_*`) | 6 | 4 | CIFAR, HAM10000 | H-SFP (baseline), E-HSFP(full) x 3 intervals | H-SFP-baseline interval rows: `COMPLETE_UNVALIDATED`; all `_full_e_hsfp` interval rows: **`INVALID`** (same collapse bug); failed rows: `FAILED_RETRYABLE` | Investigate NaN cause (Task in Section 20) for the 4 failures; rerun all `_full_e_hsfp` rows regardless of their prior pass/fail status |
| dropout/staleness | 0 | 0 | — | — | `MISSING` — never run (empty `results/journal/{dropout,staleness}/`) | Run under frozen protocol, with current (post-`cf6454c`) code |

**Confirmed failure causes (from `orchestrator.log`):**

| experiment_id | Failure | Category | Retryable? |
|---|---|---|---|
| `conv_fed_classification_federated_cifar_fednova_resnet50_s0` | `RuntimeError: tensors on cuda:0 and cpu` in FedNova aggregation | implementation_error | **VERIFIED FIXED** — `classification/Federated/server/FedNovaAggregator.py:53-58` has the `.to(device)` + `is_floating_point()` guard (landed `cf6454c`, 2026-07-16, predates this 2026-07-09 failure). Confirmed via a clean 2-epoch smoke rerun this session (`outputs/verify/fednova_smoke`, exit 0, Test Acc 1.09%, no crash). |
| `conv_fed_classification_federated_ham10000_fednova_resnet50_s0` | same | implementation_error | Same fix, same code path — not independently rerun for HAM10000 but the crash site is dataset-independent, treated as fixed |
| `conv_heterosfl_classification_hetero-sfl_cifar_heteroSFL_resnet18_s0` | `RuntimeError: Index put requires the source and destination dtypes match (Long vs Float)` | implementation_error | **VERIFIED FIXED** — `classification/HeteroSFL/runner.py:76-77` has the `is_floating_point()` guard (same `cf6454c` commit). Confirmed via a clean 2-epoch smoke rerun this session (`outputs/verify/heterosfl_smoke`, exit 0, Final Test Acc 7.49%, F1 4.89%, no crash). |
| `conv_heterosfl_classification_hetero-sfl_ham10000_heteroSFL_resnet50_s0` | same | implementation_error | Same fix, same code path — not independently rerun for HAM10000, treated as fixed |
| `intv_hsfp_classification_h-sfp_ham10000_our_resnet50_10_20_s0` | `ValueError: Out of range float values are not JSON compliant` (NaN/Inf accuracy at serialization) | numerical_instability | Likely — may already be fixed by the fp32-SupCon stability patch committed 2026-07-20 (postdates this 2026-07-14 failure); needs verification rerun, not assumed fixed |
| `intv_ehsfp_classification_h-sfp_ham10000_our_resnet50_5_10_full_e_hsfp_s0` | same | numerical_instability | Same caveat, plus depends on `full_e_hsfp` deadlock status |
| `intv_ehsfp_classification_h-sfp_ham10000_our_resnet50_10_20_full_e_hsfp_s0` | same | numerical_instability | Same caveat |
| `intv_ehsfp_classification_h-sfp_ham10000_our_resnet50_25_50_full_e_hsfp_s0` | `Terminated` mid-epoch 11/60 (external kill signal, not a crash) | interrupted_process | Yes — clean rerun |

### 4.2 `docs/optimization_loop/EXPERIMENT_REGISTRY.csv` (date-seeded, hash-tracked, 2026-07-13 to 2026-07-14, classification/CIFAR-100 proxy only)

26 rows, the most rigorous provenance in the repo (git commit + resolved-config hash + partition
hash + architecture id per row). Stops at PLAN-14 (2026-07-14); later work (PLAN-15 onward,
HAM10000/ISIC, camera-ready) is documented narratively in `DECISIONS.md`/`RESULTS_SUMMARY.md`
instead, not added to this CSV.

| Experiment ID | Status | Value | Protocol match | Reuse decision |
|---|---|---:|---|---|
| `plan0_effects`, `plan0_model_audit` | PASS (integrity only) | n/a | MATCH | `VERIFIED_EXISTING` — code-correctness evidence, not a training result |
| `plan1_smoke_baseline` | FAIL (CPU bfloat16/FP32 mismatch) | — | n/a | `INVALID` — smoke-only, superseded |
| `plan1_smoke_full_e`, `plan1d_smoke_*` | PASS (smoke) | n/a | n/a | `VERIFIED_EXISTING` as smoke-gate evidence only, never a reportable accuracy |
| `plan1b_smoke_*` | BLOCKED (no CUDA on that run) | — | n/a | superseded by `plan1d` |
| `plan2_proxy_splitfl` | COMPLETE | 44.57% (test-as-val top1) | `PRE-EVALFIX` + test-leakage | `INVALID` — reused test split as validation (Section 3.9); superseded numbers exist for H-SFP but not confirmed re-run for SplitFL under EVALFIX |
| `plan2_proxy_federated` | COMPLETE | 34.76% (test-as-val top1) | `PRE-EVALFIX` + test-leakage | `INVALID`, same reason |
| `plan2_proxy_hsfp_baseline` | COMPLETE | 3.81% (test-as-val top1) | `PRE-EVALFIX` (frozen client-0 eval bug) | `INVALID` — explicitly superseded by `plan9_raw_evalfix` (7.92%) |
| `plan2_proxy_hsfp_full_e` | PENDING | — | — | `MISSING`, never completed |
| `plan5/6/7_proxy_hsfp_*` (centered-cosine, SSL/supervised client-objective ablations, whitened-cosine) | COMPLETE | 3.48% / 4.14% / 1.31% | `PRE-EVALFIX` | `INVALID` as absolute numbers (pre-evalfix); the *relative* finding (whitening hurts, supervised-objective is neutral) may still be directionally informative but should be re-verified post-evalfix before citing |
| `plan9_raw_evalfix` | COMPLETE | **7.92%** corrected val top1 | MATCH (post-evalfix) | `VERIFIED_EXISTING` — the true corrected raw H-SFP baseline, single seed 20260714 |
| `plan10_centered_corr` | COMPLETE | **9.57%** (+1.65 vs raw) | MATCH | `VERIFIED_EXISTING` — promoted |
| `plan10_whitened_corr` | COMPLETE | 6.32% | MATCH | `VERIFIED_EXISTING` — rejected (below raw) |
| `plan11_freeze_centered` | COMPLETE | **9.76%** | MATCH | `VERIFIED_EXISTING` — best single-seed config, seed 20260714 |
| `plan11_full_e_centered_HANG` | DNF_HANG | — | — | `FAILED_PERMANENT` at time of writing (root cause: `persistent_workers` + many per-client DataLoaders) |
| `plan13_full_e_omp` | DNF_HANG_DEFERRED | — | — | `FAILED_PERMANENT` — futex deadlock persists even after `num_workers=0` + thread pinning; this is a **different, deeper** deadlock than plan11's (later reappears as the autograd-backward deadlock in `DECISIONS.md` 2026-07-16, 5 more failed fix attempts) |
| `plan14_freeze_centered_s20260715` | COMPLETE | 10.55% | MATCH (date-seed protocol) | `VERIFIED_EXISTING` |
| `plan14_freeze_centered_s20260716` | COMPLETE | 10.56% | MATCH (date-seed protocol) | `VERIFIED_EXISTING` |

**3-seed aggregate for H-SFP baseline+centered+freeze, CIFAR-100, 10-round proxy, IID:**
9.76 / 10.55 / 10.56 → **mean 10.29%, std 0.37%** (date-seed protocol, n=3). This is the single
most load-bearing "existing H-SFP number" in the whole repo and is `VERIFIED_EXISTING`.

### 4.3 `results/fair_comparison_{cifar100,ham10000,isic2018}.csv` (date-seeded, 3 seeds: 20260714/15/16)

Already-aggregated 3-seed tables produced by earlier sessions (this session fixed and re-ran the
ISIC-2018 H-SFP row today — see `docs/optimization_loop/DECISIONS.md` 2026-07-20 entry). Schema
note: `fair_comparison_cifar100.csv`/`fair_comparison_ham10000.csv` match `tools/fair_compare/aggregate.py`'s
native `COLUMNS` schema (mergeable); `fair_comparison_isic2018.csv` uses an incompatible ad-hoc
6-column schema and **crashes `aggregate.py` with `KeyError: 'config'`** if a merge is attempted —
confirmed today, fix queued.

| Dataset | Method | Metric | Value (mean±std, n=3) | Status |
|---|---|---|---:|---|
| CIFAR-100 (10-round proxy) | H-SFP (freeze+centered) | top-1 acc | 10.29 ± 0.37% | `VERIFIED_EXISTING` (date-seed protocol) |
| CIFAR-100 (10-round proxy) | SplitFL | top-1 acc | 45.79% | `COMPLETE_UNVALIDATED` — needs post-evalfix confirmation (n=1 in registry; check if this is the same pre-evalfix 44.57 number or a later rerun) |
| ISIC-2018 (10-round proxy) | H-SFP | IoU | 47.27 ± 2.19% | `VERIFIED_EXISTING` — fixed and re-run today, this session |
| ISIC-2018 (10-round proxy) | HierFL | IoU | 63.17 ± 1.56% | `VERIFIED_EXISTING` |
| ISIC-2018 (10-round proxy) | Federated | IoU | 27.76 ± 0.86% | `VERIFIED_EXISTING` |
| ISIC-2018 (10-round proxy) | HeteroSFL | IoU | 25.80 ± 0.26% | `VERIFIED_EXISTING` |
| ISIC-2018 | SplitFL, HSFL | — | — | `MISSING` — never included in the ISIC campaign (no valid config exists, Section 3.9) |

Full detail already in `docs/optimization_loop/RESULTS_SUMMARY.md` and `DECISIONS.md` — linked,
not duplicated here.

### 4.4 Additional findings from the raw-artifact inventory (2026-07-20, folded in post-initial-write)

- **`EXPERIMENT_REGISTRY.csv` is more stale than first noted**: it stops at PLAN-14, but
  `docs/optimization_loop/logs/` contains 41 plan-numbered subfolders running through PLAN-36
  (today's ISIC fix). The registry has not been updated for ~22 later plans. Treat it as a partial
  index of early CIFAR-100 proxy work only, not a complete provenance trail.
- **Full-60-round CIFAR-100 convergence already exists** for all 6 baselines + H-SFP, 3 seeds each
  (`results/fair_comparison_cifar100_full60.csv`, **not committed to git**, local-only):

  | Method | test_top1 (mean, n=3) | Comm (MB) | Runtime (s) | Rounds verified |
  |---|---:|---:|---:|---|
  | SplitFL | ~60.0% | ~222,184 | ~1,300-1,340 | 60/60 confirmed |
  | HSFL | ~57.3% | ~757,374 | ~930-940 | 60/60 confirmed |
  | Federated | ~56.6% | ~565,463 | ~2,710-2,800 | 60/60 confirmed |
  | HierFL | ~54.8% | ~115,663 | ~2,010-2,030 | 60/60 confirmed |
  | HeteroSFL | ~38.4% | ~36,000-37,300 | ~1,130-1,132 | round-count field absent from raw JSON — verify separately before treating as `VERIFIED_EXISTING` |
  | H-SFP (ours) | ~11.5% | ~756 | ~6,010-6,069 | 60/60 confirmed, `runtime_counters.jsonl` confirms these are **baseline H-SFP, not E-HSFP** (all `ehsfp_runtime_counters` zero) |

  Status: `COMPLETE_UNVALIDATED` pending the same F1-validity and communication-accounting checks
  as everywhere else (Section 3.9). Directly usable for Section 5.2's IID full-scale row once
  validated — this is a substantial head start, not `MISSING`.

- **`results/fair_comparison_ham10000.csv` H-SFP row is suspicious and must be investigated before
  reuse**: the reported top-1 (10.983524712930603) is **bit-identical across all 3 different
  seeds** (20260714/15/16). This is not plausible for genuine stochastic training and strongly
  suggests a seed-not-applied bug (e.g. a fixed data order, a cached/reused checkpoint, or an
  eval path that doesn't depend on the trained weights) rather than real convergence stability.
  **Status: `INVALID` pending investigation — do not report this number as-is.** Queued as a new
  Codex investigation task (Section 20).
- **Segmentation SplitFL and HSFL have literally zero artifacts** — not merely missing configs
  (Section 3.9) but no `Figure/` output directory exists at all for either method under
  `segmentation/`. Confirms `MISSING`, not `PARTIAL`.
- A separate, earlier **camera-ready (ECCV reviewer-response) experiment suite**
  (`results/camera_ready/`) exists — covariance/fairness/hetero/inversion/lstat/partial/profiling
  studies. Its own README states numbers are smoke-scale/unexecuted and explicitly warns "no
  numbers in the paper should be taken from here until produced by code." Treat as adjacent,
  not part of this journal campaign, and not a source of reusable numbers without independent
  verification.
- ~280 orphaned TensorBoard event files + 4 dated log subfolders under top-level `logs/` predate
  the July bugfix cycle (some as old as May 2026) — legacy, matches the "stale/mixed-arch" warning
  already in `E-HSFP_STATUS_REPORT.md`. Not a source for this campaign.
- Confirms Section 4.1's HAM10000 interval failure count precisely: **4 of 7** HAM10000 interval
  jobs failed (only `intv_hsfp_ham10000_..._25_50_s0` and `..._5_10_s0` succeeded) — the HAM10000
  interval sweep is more broken than the CIFAR one.

### 4.5 E-HSFP component implementation — corrected/sharpened findings (supersedes Section 3.7's initial pass)

Full component-by-component detail from direct code audit (all line numbers verified):

| # | Component | Real status |
|---|---|---|
| 1 | Episodic prototype memory | **Fully implemented and wired** (`ehsfp/memory.py`), both classification and segmentation. Genuinely exercised. |
| 2 | Prototype dropout | Dropping mechanism **real and wired**. The companion `dropout_consistency_loss` (`ehsfp/losses.py:104`) is imported into both `hierarchy.py` files but **never called** — `use_dropout_consistency`/`lambda_dropout` are dead config keys. |
| 3 | Reliability-aware aggregation | **Fully implemented and wired**, both tasks. Genuinely exercised. |
| 4 | Prototype replay consistency (PRC) | **Fully implemented and wired**, both tasks. This is the component causing the autograd-backward deadlock (Section 19). |
| 5 | Diagonal Gaussian synthesis | **Fully implemented, always-on default path.** Universally exercised whenever no other generator is supplied — which is always, per item 7. |
| 6 | Low-rank covariance synthesis | Implemented in `camera_ready/synthesis.py` but **exists only as a standalone offline script** (`scripts/camera_ready/run_covariance.py`) — not reachable from any live E-HSFP training run, no `synthesis_mode` config key exists. `BLOCKED` for Section 17.11 until wired into `ehsfp/aggregation.py`'s live path. |
| 7 | Residual prototype generation | **Confirmed complete stub.** `ehsfp/generator.py`'s `ResidualPrototypeGenerator` is fully implemented but both `classification/H-SFP/hierarchy.py:467` and `segmentation/H-SFP/hierarchy.py:332` hardcode `self.residual_generator = None` regardless of the `use_residual_generator` flag. **Every stored `full_e_hsfp` run's `run_metadata.json` claims `effective_ehsfp.use_residual_generator: true` — this is false; the flag has zero runtime effect.** This is a provenance-integrity issue, not just a missing feature: existing "full E-HSFP" run metadata actively misrepresents what components ran. |
| 8 | Prototype fidelity regularization | **Not implemented as a training-time regularizer anywhere.** Only an offline measurement exists (`camera_ready/feature_distance.py`: `rbf_mmd` used by the offline covariance script; `gaussian_frechet_distance` implemented but has **zero call sites anywhere in the repo** — fully dead code). No Wasserstein or covariance-reconstruction-error implementation exists at all. |
| 9 | Prototype-induced drift control | **Not implemented as a named component anywhere** (zero grep hits for "drift" in any code path). The closest adjacent code is `ehsfp/prototype_space.py`'s centering/whitening/`recenter_memory` — a different, classification-only ablation (`prototype_space: raw|centered_cosine|whitened_cosine`), not part of `EHSFP_DEFAULTS`/`ABLATION_PRESETS`, not imported by `segmentation/H-SFP/hierarchy.py` at all, and not framed as drift control. |
| — | Serverless stress simulation | **Functionally real**, both tasks: `ServerlessMetricsTracker`'s timeout gating actually skips a client's/edge's entire computation for that round (not just a metric). **Correction to the module's own docstring**, which claims "metrics only, no algorithmic effect" — this is stale/inaccurate. Cold-start is metrics-only (inflates a latency stat, does not delay/skip anything). Event loss and partial-edge-execution are achieved *compositionally* (dropout + timeout together), not as distinct first-class stress primitives — no dedicated "combined stress mode" flag exists (Section 3.8, now resolved from `MISSING` to this description). |
| — | Centralized stability/convergence/drift/fidelity metrics module | **[UPDATED 2026-07-21]** Was confirmed absent at audit time; 5 of 6 now implemented and GPU-verified in `ehsfp/research_metrics.py`, wired into H-SFP (both tasks) — see Section 3.9. Stability-bound term remains genuinely absent (`BLOCKED_NO_THEORY`, not fabricated). |

**Practical consequence:** the paper's "full E-HSFP" configuration, as actually executed by every
run in this repo to date, is missing 4 of its 9 claimed components when checked against the full
component list in Section 2 of the brief (residual generator: complete stub; dropout-consistency
loss: dead code; prototype fidelity regularization: not implemented as a training-time term at
all; prototype-induced drift control: not implemented anywhere under this name), plus low-rank
covariance synthesis being real but offline-only — even setting aside the separate autograd-
deadlock blocker on PRC. **Decision recorded** in `docs/optimization_loop/DECISIONS.md` ("Full
E-HSFP claimed-component scope — DECISION, 2026-07-20"): no further work proceeds as if
`full_e_hsfp` already means what its name implies; either the 4 gaps get implemented for real
before any "full E-HSFP" number is used in the paper, or the journal's empirically-claimed
contribution is explicitly narrowed to the subset that's genuinely implemented (memory + dropout +
reliability + PRC once unblocked). That specific a/b choice is left open as a paper-scope decision.

---

## 5. Main Classification Results

**Status: MOSTLY MISSING.** Only CIFAR-100 H-SFP (Section 4.2/4.3) and fragments of the
`scripts/journal_experiments` convergence sweep (Section 4.1, metrics not yet extracted) exist.
CIFAR-10, ImageNet-1K: no results found for any method. Full 8-method x 4-dataset grid with
5-seed protocol, real macro-F1, and normalized communication accounting is `MISSING` pending the
Section 20 work queue.

### 5.1 CIFAR-10 — `MISSING` (all methods)
### 5.2 CIFAR-100 — `PARTIAL` (H-SFP baseline: VERIFIED_EXISTING date-seed n=3; all other methods + E-HSFP + 5-seed protocol: MISSING)
### 5.3 HAM10000 — `PARTIAL` (raw logs exist in `scripts/journal_experiments`, metrics not extracted: `COMPLETE_UNVALIDATED`)
### 5.4 ImageNet-1K — `BLOCKED` (only H-SFP has a config; local `data/ImageNet` is an empty directory — no data present; config itself untested)

---

## 6. ISIC-2018 Segmentation Results

**Status: PARTIAL, but unblocked.** 4/8 required methods complete and verified (Section 4.3):
H-SFP, HierFL, Federated, HeteroSFL. FedProx/FedNova (strategy flags under Federated, implemented,
never run for ISIC — `MISSING`, not `BLOCKED`).

**[UNBLOCKED AND GPU-VERIFIED, 2026-07-21]** SplitFL and HSFL previously had zero
segmentation-capable model architectures at all (only classification heads) — this was
misdiagnosed earlier as "just needs a config" and corrected mid-session (Section 20 item 8).
Codex implemented genuine client/edge(/server)/decoder segmentation models for both
(`segmentation/SplitFL/models/ISIC_ResNet50.py`, `segmentation/HSFL/models/ISIC_ResNet50.py`,
mirroring H-SFP's working pattern), wired in the existing `DiceFocalLoss`/`compute_iou_and_dice`
helpers, added the missing HSFL ISIC config, and wrote CPU shape tests (independently re-run,
passing: correct intermediate cut shapes and exact `[1,1,224,224]` final output for both). Claude
reviewed both diffs in full and GPU-smoke-verified both end-to-end on real ISIC data (2-epoch
runs):

| Method | Test IoU | Test Dice | Communication routes populated |
|---|---:|---:|---|
| SplitFL | 26.00% | 37.62% | `client_to_server`/`server_to_client` (flat, correct) |
| HSFL | 23.09% | 33.72% | `client_to_edge`/`edge_to_client`/`edge_to_cloud`/`cloud_to_edge` (hierarchical, correct) |

These are 2-epoch smoke numbers, not final campaign results — real numbers require the full
10-round proxy run at the frozen protocol (Section 20 next-steps), but the methods themselves are
now genuinely functional for the first time. `NOT_IMPLEMENTED` → `READY`.

E-HSFP (full): still `BLOCKED` on the `full_e_hsfp` autograd deadlock (itself blocked on missing
`sudo` for `py-spy`/`gdb` — Section 19/20).

---

## 7. Convergence Records

`MISSING` — requires the not-yet-implemented stability/prototype-drift/stability-bound metrics
(Section 3.9). Raw per-round loss/accuracy histories likely exist inside individual run JSONs for
some completed runs (e.g. H-SFP's `metrics_unrounded.jsonl`) but have not yet been inventoried
into a per-round CSV. Queued for the pending raw-artifact audit agent.

### 7.1 IID — `PARTIAL` (H-SFP CIFAR-100 exists; no stability/drift columns)
### 7.2 Mild non-IID (alpha=0.7) — `MISSING` (never run at this alpha)
### 7.3 Severe non-IID (alpha=0.3) — `MISSING` (never run at this alpha)
### 7.4 Prototype drift — `BLOCKED` (metric not implemented anywhere)
### 7.5 Stability-bound term — `BLOCKED` (metric not implemented anywhere)

---

## 8. Prototype-Dropout Results

`MISSING`. `scripts/journal_experiments/run_dropout_staleness.sh` exists and targets dropout
rates `{0.0, 0.1, 0.3, 0.5, 0.7}` — **mismatched with the brief's requested `{0.0, 0.1, 0.2, 0.3, 0.5}`**
(0.2 requested but not swept historically; 0.7 swept historically but not requested). Never
actually executed (empty `results/journal/dropout/`). Frozen decision: **use the brief's values
`{0.0, 0.1, 0.2, 0.3, 0.5}`** for new runs (edit the existing sweep script's rate list rather than
introduce a third convention).

---

## 9. Prototype-Staleness Results

`MISSING`. `run_dropout_staleness.sh`'s staleness sweep (`max_prototype_age ∈ {0,1,3,5,10}`)
**matches the brief exactly**. Never executed. Requires the stability-bound-term metric
(unimplemented) for full Section 17.5 compliance — accuracy/F1/communication/memory/time
sub-metrics can be produced without it.

---

## 10. Serverless-Style Stress Results

`MISSING`. `ServerlessMetricsTracker` exists with cold-start/timeout probability parameters but
exact simulation semantics are unverified (Section 3.8) and partial-execution / missing-edge-update
parameters have not yet been located. Recovery-gap metric unimplemented. Needs the pending
component-audit agent's findings before a registry entry can be written with confidence.

---

## 11. IID and Non-IID Robustness

`MISSING` for the brief's exact 9-cell grid (3 datasets x 3 non-IID levels). Related but
non-comparable data exists: E-HSFP component ablation at alpha=0.1 (single level, CIFAR-100 only,
Section 4.3-adjacent, see `RESULTS_SUMMARY.md`).

---

## 12. Component Ablations

`PARTIAL`. The 6-preset ladder (Section 3.7 table) was run once each (seed=0) for CIFAR-100 and
HAM10000 (`scripts/journal_experiments`, Section 4.1) — metrics not yet extracted, `full_e_hsfp`
preset's actual behavior at proxy scale is suspect given the documented deadlock (it's marked
`.done` in the marker system, meaning it *did* complete for at least one config — worth
investigating why this succeeded when PLAN-11/13's `full_e_hsfp` hung, possibly a different config
scale or a fix landed between 2026-07-14 dates; flagged for the next audit pass, not yet resolved).
5-seed protocol version: `MISSING`.

---

## 13. Memory-Size Ablation

`MISSING`. No `memory_size` sweep found anywhere in the repo (only the fixed default `500` is
used in all located runs).

---

## 14. Aggregation-Rule Comparison

**Status: `READY`** (was `BLOCKED`, unblocked 2026-07-21). `aggregation_mode` now supports all 3
underlying modes needed for the brief's 4-way comparison (`average`, `sample_count_weighted`,
`learnable_reliability` — the 4th, "reliability-aware + memory", is `learnable_reliability` with
`use_episodic_memory=True`, already possible via existing flags). Implemented by Codex in
`ehsfp/aggregation.py`/`ehsfp/config.py`, wired through both classification and segmentation
H-SFP. Reviewed by Claude: hand-verified the weighted-mean math against a synthetic 2-source case
(`(1×0 + 3×10)/4 = 7.5`, matches exactly), confirmed the all-zero-support edge case falls back to
plain averaging without NaN, confirmed `average`/`learnable_reliability` behavior is
byte-identical to before (backward-compatible `aggregation_mode=None` inference retained for any
caller that doesn't pass it explicitly). Independently re-ran all 3 new tests (pass) and
GPU-smoke-verified end-to-end (`--set aggregation_mode=sample_count_weighted`, clean 2-epoch run,
no reliability network instantiated as expected for this mode). No numeric comparison table
produced yet — that requires the actual 4-way campaign run (clean + dropout-stress +
staleness-stress conditions per the brief), still `MISSING` as a *result*, only the
*implementation* is now `READY`.

---

## 15. Prototype-Synthesis Ablation

`BLOCKED`. Diagonal-Gaussian synthesis (`_generate_synthetic_data`) exists and is the default
path. Low-rank covariance synthesis: not yet confirmed present (pending component audit).
Residual generator: present in config but **not wired into either hierarchy** (`residual_generator
= None` regardless of the flag — Section 3.7). Residual generator + consistency loss: doubly
blocked (consistency loss also never invoked). Needs real implementation before this section can
produce anything.

---

## 16. Scalability

`MISSING`. No run at 25/50/100/200/500 explicit client counts was found (200 is the *default*, not
a deliberately-swept scale point). The `[GOTCHA]` in Section 3.3 (mid_server/num_edges must be set
together) must be respected when these configs are authored.

---

## 17. Communication and Tier-Wise Memory

Communication: `BLOCKED_ON_RERUN` — accounting is now normalized across all 6 method families
(Section 3.9), verified correct and GPU-smoke-tested, but no full experiment campaign has been run
with the fixed code yet. Existing per-method communication figures in already-completed runs
predate this fix and are `NOT_COMPARABLE` to each other or to future runs; a fresh cross-method
communication table requires (re)running the campaign, not just recomputing from old logs.

Tier-wise memory: still `PARTIAL`/`NOT_COMPARABLE` — the memory-measurement inconsistencies
(SplitFL's sampled-current-allocation vs. everyone else's true CUDA peak; only HSFL genuinely
separates by tier) are a separate, still-open normalization task (Section 20/21).

---

## 18. Numeric Data Required for Future Figures

All entries: **`NUMERIC_DATA_VERIFIED — RENDERING_DEFERRED`** is the target end-state; current
state is `NUMERIC_DATA_NOT_YET_VERIFIED` for all of the below, since the underlying campaign is
still mostly `MISSING`. Rendering remains deferred regardless per the brief (Section 4/18 rule) —
noted here only for completeness, not as an invitation to start plotting.

| Future figure | Required variables | Existing data | Missing data | Producing experiment ID(s) | Status |
|---|---|---|---|---|---|
| Convergence curves | per-round acc/F1/loss, 3 non-IID levels | H-SFP CIFAR-100 per-round history (unverified extraction) | mild/severe non-IID sweeps, baselines, E-HSFP | TBD (Section 12 registry) | `MISSING` |
| Prototype drift over rounds | per-round drift metric | none | metric itself unimplemented + all runs | TBD | `BLOCKED` |
| Stability-bound vs empirical drift | both metrics per round | none | both metrics unimplemented | TBD | `BLOCKED` |
| Dropout sensitivity | accuracy/F1/stability vs dropout rate | none | full sweep | TBD | `MISSING` |
| Staleness sensitivity | accuracy/F1/stability vs tau | none | full sweep | TBD | `MISSING` |
| Component-ablation trends | per-preset metrics | seed=0 logs exist, unextracted | metric extraction, 5-seed | Section 4.1 `abl_*` rows | `PARTIAL` |
| Reliability-weight analysis | per-prototype reliability fields | none captured in structured form | full instrumentation + runs | TBD | `MISSING` |
| Memory-size effects | metrics vs memory_size | none | full sweep | TBD | `MISSING` |
| Communication–accuracy tradeoff | comm (normalized) vs accuracy, all methods | scattered, non-normalized | accounting normalization + full grid | TBD | `NOT_COMPARABLE` |
| Memory–performance tradeoff | tier memory vs accuracy, all methods | scattered, non-normalized | same | TBD | `NOT_COMPARABLE` |

---

## 19. Failed and Interrupted Runs

See Section 4.1's failure table for full detail (4 crashes + 1 external termination, all diagnosed
with concrete file:line fixes or a "verify, don't assume fixed" note). Machine-readable copy:
`results/journal/manifests/RUN_FAILURES.csv` (to be populated — Section 12 task in progress).

**Additional structural failure, more severe than any single crash:** `full_e_hsfp` (the complete
proposed E-HSFP method) has hung/deadlocked at proxy scale (200 clients) across at least two
distinct root causes and 5+ fix attempts spanning 2026-07-14 (PLAN-11 DataLoader-worker hang,
fixed) through 2026-07-16 (`DECISIONS.md`: torch.autograd backward-pass deadlock at
`hierarchy.py:908`, still unresolved as of that entry — PRC batching, `.item()` sync removal,
TensorBoard writer removal, and graph-bounding via detach were all tried and failed). **This
blocks the paper's headline configuration across nearly every experiment family that requires
"full E-HSFP."** Status: `FAILED_PERMANENT` pending a GPU-attached debugger session (py-spy+sudo
or gdb), explicitly noted in `DECISIONS.md` as needing exactly that. This is the single highest-
priority item in Section 20's queue given how much of the campaign depends on it.

**[2026-07-21] Confirmed genuinely blocked, not just deferred**: tested `py-spy dump --pid` on a
live process this session — fails with "Permission Denied" (`ptrace_scope=1` blocks attaching to
a non-child process without root; confirmed no passwordless `sudo` is configured in this
environment). `gdb` is installed but subject to the same ptrace restriction. This cannot be worked
around autonomously — it requires either the user enabling passwordless `sudo` for `py-spy`/`gdb`
specifically, or the user running an interactive debug session themselves. Not attempting to
bypass this. Pivoting to other Section 20 items in the meantime; this remains the top item to
resume once unblocked.

**Time-budget reality check:** one CIFAR-100 H-SFP convergence run (60 rounds) took ~6.7 hours
wall-clock. The prior `scripts/journal_experiments` pass, covering only convergence+ablation+
intervals at **seed=0 only**, took 160.5 hours. Extrapolating to the full brief (5 seeds x the
complete grid across 17.1-17.13, plus scalability to 500 clients, plus ImageNet-1K) on this single
RTX 3080 Ti implies a multi-month continuous-compute campaign. This is stated plainly so scope
and pacing expectations are shared, not to argue against proceeding — the campaign continues,
prioritized per the brief's own Stage 0→5 ordering (Section 20).

---

## 20. Missing Simulations

Effectively the entire brief beyond Sections 4.2-4.3's existing results. Priority queue (mirrors
the brief's Stage 0→5 + the newly-discovered infrastructure gaps):

1. **[BLOCKING, highest priority]** Debug the `full_e_hsfp` autograd deadlock — without this,
   "full E-HSFP" cannot be reported anywhere in the paper.
2. ~~Investigate the ablation-preset/seed-invariant accuracy finding~~ — **root-caused this
   session** (Codex, >99% confidence, CPU-only proof script): pre-`cf6454c` config-precedence bug,
   already fixed in current code. **New follow-up**: rerun all non-baseline-preset
   `scripts/journal_experiments` H-SFP/E-HSFP runs (10 `abl_*` rows, 4 `conv_ehsfp_*` rows, all
   `intv_*_full_e_hsfp` rows) with current code — this is real GPU campaign work, not a code fix.
3. ~~Fix `collect_results.py` metric-parsing regex~~ — **done this session** (Codex, verified).
   Follow-up: capture the genuine per-epoch macro-F1 already printed by Federated/HierFL/HeteroSFL
   logs (currently unextracted).
4. ~~Verify the classification/HeteroSFL dtype crash and FedNova device crash are fixed~~ — **done
   this session**, confirmed via clean 2-epoch smoke reruns (both exit 0, no crash). Still open:
   verify the 3 HAM10000 interval NaN failures against the fp32-SupCon fix (untested; also
   overlaps with the `full_e_hsfp` rerun above since 2 of the 3 are `_full_e_hsfp` variants).
5. ~~Implement real macro-F1 for H-SFP/HSFL/SplitFL classification~~ — **done 2026-07-21**,
   GPU-verified.
6. ~~Implement a normalized cross-method communication-accounting scheme~~ — **done 2026-07-21**,
   GPU-verified.
7. ~~Implement stability, prototype drift, rounds-to-convergence, prototype fidelity,
   recovery-gap metrics~~ — **5 of 6 done 2026-07-21**, GPU-verified (stability-bound term
   intentionally left undefined, `BLOCKED_NO_THEORY` — no manuscript exists to source the formula
   from; not fabricated).
8. ~~Implement genuine ISIC-2018 segmentation models for `segmentation/SplitFL` and
   `segmentation/HSFL`~~ — **done and GPU-verified 2026-07-21** (Section 6): both now genuinely
   segment (SplitFL 26.00% IoU / HSFL 23.09% IoU on a 2-epoch smoke run), correct communication
   routes populated for each method's topology.
9. ~~Author valid ISIC-2018 configs for SplitFL and HSFL~~ — done, merged into item 8.
10. Wire in the residual prototype generator and dropout-consistency loss (currently no-ops
    despite config flags existing) — or formally reduce "full E-HSFP"'s claimed scope (Section 4.5,
    decision explicitly left to the user/paper-scope).
11. Implement the "sample-count weighted" aggregation rule (currently only average/reliability exist).
12. Then: the full Stage 1→5 numeric campaign per the brief, at the frozen protocol (Section 3).

---

## 21. Protocol Inconsistencies

1. Two incompatible seed conventions (date-based n=3 vs. integer 0-4 n=5) — resolved by decision
   in Section 3.1 (new work uses 0-4; old date-seeded results kept separate, not pooled).
2. Dropout-rate sweep values in the existing script (`{0,0.1,0.3,0.5,0.7}`) vs. the brief's
   requested values (`{0,0.1,0.2,0.3,0.5}`) — resolved in Section 8 (adopt the brief's values).
3. Mild/severe non-IID Dirichlet alphas (0.7/0.3 per the brief) have never actually been run in
   this repo; historical runs used 0.1/0.05/1.0 instead — flagged in Section 3.2, frozen
   provisionally at the brief's values pending manuscript cross-check.
4. ~~Communication accounting is not uniform across methods~~ — **resolved 2026-07-21** (Section
   3.9/17): unified route-based schema across all 6 methods, GPU-verified. Existing/historical
   communication numbers predate the fix and remain not comparable to new runs.
5. Memory-measurement semantics differ between SplitFL (sampled current allocation) and every
   other method (true CUDA peak-allocator API) — not yet resolved, queued.
6. Optimizer mismatch: SplitFL/HeteroSFL classification runners hard-code SGD despite
   `optimizer: adam` in their configs (Section 3.5) — not yet resolved, queued.
7. `configs/*` labeled `resnet50` for CIFAR classification actually instantiate a modified
   ResNet-18 (Section 3.4) — not yet resolved (a manuscript-text fix, not necessarily a code fix);
   flagged prominently.
8. `results/fair_comparison_isic2018.csv` uses an ad-hoc schema incompatible with
   `tools/fair_compare/aggregate.py`'s native columns, causing a `KeyError` crash on merge attempts
   — not yet resolved, queued.
9. Interval-A accuracy vs. interval-C communication/memory figures may have been historically
   combined into one apparent "configuration" (brief's own callout) — not yet traced to specific
   historical numbers; enforced going forward via the registry's explicit interval field.

---

## 22. Final Completeness Checklist

Tracking against the brief's 31-item Section 26 list. All items currently `NOT MET` except:

- [x] Item 2 (audit old artifacts) — in progress, this document; 3/5 audit agents landed, 2 pending.
- [ ] All other items — pending the Section 20 work queue.

This checklist will be updated in place (not appended) as items close.

---

## 23. Change Log

- **2026-07-21 (sample-count-weighted aggregation implemented and GPU-verified; last well-scoped
  Section 20 item closed):** Dispatched Codex to add the missing `sample_count_weighted`
  aggregation mode. Implementation reuses the existing weighted-mean/variance formulas with
  support-count-derived weights instead of network-learned ones, with a proper zero-support
  fallback and full backward compatibility for the two existing modes. Reviewed the diff, hand-
  verified the weighted-mean math on a synthetic case, independently re-ran all 3 new tests, and
  GPU-smoke-verified end-to-end. Section 14 (aggregation-rule comparison) moves from `BLOCKED` to
  `READY` (implementation only — the actual 4-way numeric comparison is still a `MISSING` result
  pending a real campaign run). This closes out every well-scoped, unambiguous item in the Section
  20 priority queue reachable without either the sudo-blocked deadlock or the user's paper-scope
  decision on "full E-HSFP" (Section 4.5/13) — remaining work is the full multi-week numeric
  campaign itself.
- **2026-07-21 (SplitFL and HSFL ISIC segmentation implemented and GPU-verified):** Dispatched
  Codex with the corrected, larger scope (new model architectures, not just configs). It
  implemented genuine client/edge(/server)/decoder segmentation models for both methods mirroring
  H-SFP's working pattern, wired in the existing DiceFocalLoss/compute_iou_and_dice helpers
  (no new loss/metric code), authored the missing HSFL config, and wrote CPU shape tests.
  Reported nothing deferred. Reviewed both diffs in full, independently re-ran the shape tests,
  and GPU-smoke-verified both end-to-end on real ISIC data: SplitFL 26.00% IoU/37.62% Dice
  (correct flat client↔server communication routes), HSFL 23.09% IoU/33.72% Dice (correct
  hierarchical client→edge→cloud routes, all 4 directions populated appropriately). Both methods
  go from `NOT_IMPLEMENTED` to `READY` for the ISIC campaign — this was the last remaining
  `BLOCKED`/`MISSING` method gap for that dataset besides `full_e_hsfp`.
- **2026-07-21 (SplitFL/HSFL ISIC gap re-scoped from config fix to model implementation):**
  Attempted the originally-scoped "just author a config" fix for `segmentation/SplitFL`'s ISIC
  support: wrote `configs/segmentation/splitfl/isic_splitfed_resnet50.yaml` and the matching proxy
  config, smoke-tested it, and found the data loader works correctly (2594 training images
  loaded) but the model factory has zero registered segmentation-capable architectures for this
  method — every model in `segmentation/SplitFL/models/` is classification-only. This is the same
  gap already documented for `segmentation/HSFL`, now confirmed to also apply to SplitFL, and
  merged the two into one corrected, larger task (Section 20 item 8) — new client/edge/decoder
  segmentation model implementation needed for both methods, not just configs. The SplitFL config
  itself is kept (it's correct and will work once a model exists).
- **2026-07-21 (stability/convergence metrics module implemented and verified):** Since no
  manuscript exists in this repo, froze precise operational definitions for 5 of the 6 required
  metrics (stability, rounds-to-convergence, prototype drift, recovery gap, prototype fidelity via
  MMD) directly in Section 3.9, explicitly declining to fabricate a formula for the 6th
  (stability-bound term, which needs the paper's actual theoretical derivation). Dispatched Codex
  to implement the 5 defined metrics as pure, tested functions (`ehsfp/research_metrics.py`) and
  wire stability + rounds-to-convergence into H-SFP's reporting (both tasks). Reviewed the diff in
  full, independently re-ran all 8 new unit tests (including the deliberately tricky "threshold
  crossed then lost before patience completes" case), and GPU-verified with a 3-epoch smoke run —
  correctly reports `None`/`not_converged` rather than fabricating values when too few rounds have
  elapsed to satisfy the frozen window/patience parameters. Confirmed no training-behavior changes
  and no stray stub for the intentionally-omitted stability-bound term.
- **2026-07-21 (communication accounting normalized; full_e_hsfp deadlock confirmed genuinely
  blocked):** Attempted the `full_e_hsfp` autograd-deadlock investigation (top Section-20
  priority); confirmed `py-spy`/`gdb` both require `sudo` to attach to a non-child process
  (`ptrace_scope=1`, no passwordless `sudo` available) — this cannot be worked around
  autonomously, logged as a new blocker (task #18) rather than attempted insecurely. Pivoted to
  communication-accounting normalization: dispatched Codex with a detailed per-method audit
  request; it implemented a unified route-based schema across all 6 method families in both tasks,
  finding and fixing several real bugs beyond the original brief (HeteroSFL's completely
  untracked download/gradient-return cost, SplitFL's wrong gradient-size formula, several methods'
  decimal-MB/binary-MiB inconsistency, "charged all configured clients instead of selected ones"
  bugs). Reviewed the module design and 2 representative diffs in full, ran the new tests
  independently, and GPU-smoke-verified on both a flat method (HeteroSFL) and a hierarchical
  method (H-SFP) — correct schema, no crashes, confirmed no training-behavior changes. Section
  17's communication table is now scientifically valid to produce once a fresh campaign is run
  with this code (existing numbers predate the fix and remain non-comparable).
- **2026-07-21 (ablation rerun campaign complete for 3/6 presets, both datasets):** Real macro-F1
  fix GPU-verified end-to-end (genuinely distinct accuracy/F1 values confirmed). Diagnosed and
  fixed a new, live NaN crash in classification's reliability-aggregation path (same fp16-
  autocast-overflow bug class as the segmentation fix, unported to the separate classification
  codebase) — Codex root-caused and fixed it, Claude reviewed and GPU-verified with the exact
  previously-crashing command. This same fix also resolved the separate-seeming
  "HAM10000 training collapses at epoch 1" finding: reran `hsfp_memory`/`hsfp_memory_dropout` on
  HAM10000 post-fix and both now train genuinely (54.72%/50.77%, multiple best-epoch updates
  across all 60 rounds) instead of collapsing. Final ablation ladder for both datasets (3 of 6
  presets; PRC-bearing presets still blocked on the autograd deadlock): CIFAR-100
  8.10/11.28/11.20/11.26% (baseline/memory/dropout/reliability), HAM10000
  54.72/50.77/55.82% (memory/dropout/reliability). Tasks #14 and #16 closed.
- **2026-07-20 (bounded ablation rerun campaign launched):** Started a full 60-round CIFAR-100
  rerun of the 3 non-PRC ablation presets (`hsfp_memory`, `hsfp_memory_dropout`,
  `hsfp_memory_reliability`) at seed 0, under current (post-`cf6454c`) code — deliberately
  excluding `hsfp_memory_reliability_prc`/`full_e_hsfp` to avoid the unresolved autograd deadlock.
  Running in background (`docs/optimization_loop/logs/plan37_ablation_rerun/`, ~6.7h/run, ~20h
  total for 3 sequential runs). This directly replaces 3 of the 10 `INVALID` ablation rows with
  real `VERIFIED_NEW` data. HAM10000 and the PRC-bearing presets remain queued (Section 20).
- **2026-07-20 (ablation-collapse bug root-caused; crash fixes empirically verified):** Traced the
  bit-identical-across-presets anomaly by hand up through `get_ehsfp_config()`'s call site
  (verified correct), then dispatched Codex with the full evidence chain to finish the trace.
  Codex found the root cause (>99% confidence): the 2026-07-09–07-14 journal campaign ran under a
  pre-`cf6454c` config-precedence bug (YAML defaults applied after, not before, the ablation
  preset — collapsing every preset to baseline). Already fixed in current code; confirmed via a
  new CPU-only proof script (`tests/prove_ehsfp_config_resolution.py`, added and passing). Also
  cross-referenced against `docs/optimization_loop/RESULTS_SUMMARY.md`'s PLAN-0 section, which
  already flagged this class of defect — this session adds a precise mechanism and independent
  reproduction, not a wholly new discovery. Reclassified the affected journal-campaign rows as
  `INVALID` (Section 4.1) and added a rerun task. Separately, empirically verified (not just
  inferred from commit dates) that the classification/HeteroSFL dtype crash and FedNova device
  crash are both fixed, via clean 2-epoch smoke reruns on GPU (both exit 0, no crash).
- **2026-07-20 (collect_results.py fixed by Codex, verified by Claude; new critical finding):**
  Dispatched Codex with a precise, file-specific task to fix `collect_results.py`'s metric parser
  (real log-format regexes for Acc/IoU/Dice/runtime/7 communication fields, crash detection via
  traceback-vs-runtime-line ordering, explicit avoidance of any F1-from-accuracy inference, new
  unit tests). Reviewed the diff, ran the tests independently, and spot-checked 3 real logs
  (FedAvg with no comm-report block, FedNova known-crashed, H-SFP full comm breakdown) — all
  correct, no false positives. Also verified two crash-bug "fixes" I'd queued for Codex were
  already applied in commit `cf6454c` (2026-07-16, predates the 2026-07-09 failures) — caught
  this before wasting a Codex dispatch; downgraded both tasks to "verify via rerun." With real
  metrics now flowing, found a new, high-priority anomaly: all 6 component-ablation presets
  return bit-identical accuracy (8.1% CIFAR / 10.98% HAM10000) despite different components being
  enabled — while the interval sweep (same runner family) shows a believable differentiated
  trend, ruling out a parser artifact. This escalates the earlier "HAM10000 identical across
  seeds" note into a likely systemic eval bug that would invalidate the component-ablation study
  if not fixed. Added as the new #2 priority in Section 20.
- **2026-07-20 (registry + manifests created):** Created
  `experiments/journal_experiment_registry.yaml` (seeded with the 8 verified/known-invalid entries
  from Section 4; full grid enumeration queued as next step) and all 7 required machine-readable
  manifests under `results/journal/{numeric,manifests}/`. Linked from Section 4. Task queue now
  includes: HAM10000 H-SFP identical-across-seeds investigation, full-E-HSFP-scope decision
  (2 of 6 claimed components are confirmed inert stubs), plus the previously-queued
  `collect_results.py` fix and the 4 diagnosed crash-bug fixes (Codex tasks).
- **2026-07-20 (initial audit, this entry):** Repository confirmed as root. Discovered pre-existing
  `scripts/journal_experiments/` campaign harness (160.5h prior wall-clock, 44 experiments, seed=0
  only) and the hash-tracked `docs/optimization_loop/EXPERIMENT_REGISTRY.csv` (26 rows, date-seeded,
  CIFAR-100 proxy only, stops 2026-07-14). Ran `collect_results.py`, found it does not extract real
  metrics (schema/regex mismatch). Diagnosed 4 concrete crash bugs (2x FedNova device mismatch, 2x
  HeteroSFL dtype mismatch) with located fixes. Found 3 critical metric-validity issues via
  sub-agent audit: (a) H-SFP/HSFL/SplitFL classification silently report accuracy as F1, (b)
  segmentation/HSFL performs classification, not segmentation, (c) communication/memory accounting
  is not comparable across methods. Found stability/drift/convergence/fidelity/recovery-gap/
  stability-bound metrics are entirely unimplemented. Froze seed convention (5-seed `[0,1,2,3,4]`
  for new work per the brief's own fallback rule), dropout-rate sweep values (adopt brief's set),
  and non-IID alphas (provisional 0.7/0.3 per brief, unverified against manuscript). Created this
  document, `results/journal/{numeric,manifests,raw}/` skeleton. Registry YAML and CSV manifests
  not yet created — next step. Two audit agents (E-HSFP component implementation status, raw
  artifact inventory) still running at time of writing; will be folded in on next update.
