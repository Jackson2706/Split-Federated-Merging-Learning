# E-HSFP / H-SFP — Status Report & Open Questions

**Date:** 2026-07-13
**Purpose:** Self-contained status for discussion. Covers the core problem found in the
journal results, the root-cause diagnosis, the fixes implemented so far, the evidence,
and the open research questions.

---

## 0. Project context (one paragraph)

**E-HSFP** (Episodic Hierarchical Split-Federated Prototyping) extends **H-SFP**, a
3-tier hierarchical split-federated method: **Client (SSL) → Edge (SSL) → Cloud
(supervised)**. Instead of sending features/gradients, each tier transmits **per-class
prototypes** (a mean + a diagonal std per class); the next tier **resamples synthetic
features** from those statistics (`x = mean + std · N(0,1)`, element-wise) and trains on
them. E-HSFP adds episodic prototype memory, learnable reliability-aware aggregation,
prototype-replay-consistency (PRC) loss, and serverless-aware dropout. Tasks: CIFAR-100 /
HAM10000 classification, ISIC-2018 segmentation. Baselines: Federated (FedAvg/Prox/SGD/Nova),
HierFL, SplitFL, HeteroSFL, HSFL.

---

## 1. The problem we discovered (from the 1-seed, 60-epoch journal grid)

Two findings, both severe for the paper:

### (A) H-SFP / E-HSFP dramatically **underperform** every baseline

| Dataset | H-SFP / E-HSFP (resnet split) | Baselines |
|---|---|---|
| **CIFAR-100** | **8.10%** | Federated 54%, HierFL 47.7%, HSFL 57.4%, SplitFL 58.4% |
| **HAM10000** | **10.98%** | Federated 78.6%, HierFL 78.2%, HSFL 80.3%, SplitFL 83.3% |

The method is ~6–7× worse than methods that train the model normally. Random baseline on
CIFAR-100 (100 classes) is 1%, so 8% is "barely learning."

Notable: H-SFP works **much better with other backbones** at the same 60 epochs:
- CIFAR **alexnet** split: **27.6%**
- HAM **vgg** split: **53.6%**

→ The 8% → 27% → 54% spread is driven by **how semantic the transmitted representation is**,
i.e. **where the model is split**, not by the method's core idea.

### (B) The E-HSFP components have **zero measurable effect**

Every ablation preset returns an **identical** number:
`baseline_hsfp = hsfp_memory = hsfp_memory_reliability = full_e_hsfp = 8.10%` (CIFAR) and
`10.98%` (HAM). The ablation table — the paper's core contribution — shows the novel
parts (memory / reliability / PRC / dropout) change nothing.

---

## 2. Root-cause diagnosis

### 2.1 Why the resnet split collapses to 8% — the prototype pipeline destroys information

Original ResNet-18 split:
- **Client** = `conv1 → bn1 → relu → layer1` → output **`[64, 32, 32]` = 65,536-dim** low-level (edge/texture) feature map.
- The client transmits only a **per-class mean + diagonal std** of that map.
- The edge **resamples** synthetic features `mean + std · N(0,1)` **independently per element**.

Two failures:
1. **Diagonal-Gaussian resampling = white noise around the mean.** Real feature maps have
   strong spatial + cross-channel covariance; the synthetic version keeps only per-element
   marginals, so it lives *off* the real feature manifold. Done **twice** (client→edge→cloud).
2. **Low-level features have no meaningful class mean.** The class-mean of a 32×32 edge/texture
   map is mush — within-class variance dwarfs between-class. The summary is transmitted exactly
   where it is least informative.

This is confirmed by alexnet (27.6%) / vgg (53.6%) beating resnet (8–11%): those split at a
more compact, more semantic point where the class-mean is a good centroid.

### 2.2 The deeper bottleneck — the **synthetic → real gap**

Instrumenting the pipeline exposed the real ceiling:

> The **cloud classifier fits the synthetic training features almost perfectly**
> (cross-entropy 4.6 → ~0.15) **but real-image validation stays near random.**

The cloud is trained **exclusively on Gaussian resamples of prototypes** and evaluated on
**real features**. If real features don't land in the synthetic blobs, accuracy collapses.
Two concrete mechanisms:
- **BatchNorm** in the tiers is calibrated on the synthetic distribution → mis-normalizes real
  features at eval.
- **Feature-scale mismatch**: synthetic-Gaussian vectors and real feature vectors have different
  norms; the classifier boundary doesn't transfer.

### 2.3 Why the ablations are identical — **IID kills the E-HSFP components**

The configs use `iid: true`, `num_users: 200`, `frac: 0.1`. Every round samples 20 clients that
each see **all 100 classes** with near-identical distributions. Therefore:
- **Episodic memory**: past prototypes ≈ current → replay adds nothing.
- **Reliability aggregation**: all clients equally reliable → uniform weights → identical to averaging.
- **PRC loss**: memory vs current consistency ≈ 0 because they're the same.

Under IID the components are **mathematically no-ops** — hence the exactly-identical ablation
numbers. They can only differentiate under **non-IID (Dirichlet), staleness, or client dropout**.

---

## 3. Fixes implemented so far (classification, ResNet-18 / CIFAR-100)

All in `classification/H-SFP/models/cnn_cifar.py` + `classification/H-SFP/hierarchy.py`
(≈88 lines changed). **Verified for shape + train/eval behavior.**

1. **Pooled semantic-vector split.** Client now = `conv1 … layer2 → GAP → 128-d` and transmits a
   *pooled semantic vector prototype* instead of the `[64,32,32]` layer1 map. The edge became a
   1×1-conv MLP refiner (128 → 256), keeping tensors 4D `[B,C,1,1]` so the pipeline plumbing is
   unchanged.
   - **Hard constraint discovered:** the codebase **deep-copies one client model per user (×200)**.
     A full ResNet-18 backbone on the client needs **~27 GB** just for client copies + Adam → OOM.
     `layer2` (0.68M params) is ~1.6 GB and fits. This is why the client can't hold the full backbone.
2. **GroupNorm edge** (replaces BatchNorm). No running stats → **train/eval identical** (verified
   diff = 0.0) → removes the BN synthetic/real mismatch.
3. **L2-normalized cloud input** (NormFace-style, learnable scale). Applied identically to synthetic
   (train) and real (eval) features → removes the feature-scale mismatch, matches the unit-sphere
   geometry SupCon already optimizes.

*Not yet done:* **low-rank covariance resampling** (transmit top-r covariance eigenvectors per class
and sample from `N(μ, Σ_lowrank)` instead of the diagonal Gaussian). This is the principled fix for
2.2's mechanism #1 but touches the entire `(proto, dist)` contract through memory / reliability / PRC
— higher risk, deferred until we see whether the contained fixes suffice.

---

## 4. Evidence

### 4.1 Smoke test (short config) — the gap **responds**

| | Before fixes | After fixes |
|---|---|---|
| Epoch-1 val | 1.07% | **2.90%** |
| Does val track the cloud fit? | ❌ flat at random while cloud loss collapsed | ✅ **climbs with it** |

Qualitatively decisive: validation went from "flat at random regardless of training" to
"moves with training." The fixes engaged the gap.

### 4.2 Full 60-epoch run (real warmup+cosine schedule) — **IN PROGRESS**

New split + GroupNorm + L2-norm, isolated from the grid. Best-val trajectory:

| Epoch | 1 | 3 | 5 | 8 | 14 | 15 | 16 | 20 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|
| Best val % | 2.24 | 6.31 | 6.48 | 7.12 | 7.34 | 8.30 | 8.56 | 9.07 | 9.09 | **9.22** |

- **At epoch ~35/60, best = 9.22%**, already **above the old 8.10%** (which was a *final* 60-epoch
  number), and it plateaus around 9% (cosine tail may add a little).
- **But** cloud training loss is ~0.15 (near-memorizing synthetic) while val is ~9% → **the
  synthetic→real gap is narrowed, not closed.**

---

## 5. Honest bottom line

- The fixes are **sound and produce a real, but modest, gain**: CIFAR H-SFP **8.10% → ~9.2%**,
  and validation now responds to training. The diagnosis (semantic prototype + norm/scale
  consistency) is empirically supported.
- **This is nowhere near the baselines (54%).** The prototype-transfer paradigm with
  **diagonal-Gaussian resampling** appears **fundamentally capped** for 100-class problems: the
  cloud can perfectly fit synthetic blobs that real features never occupy.
- The **ablation problem (B) is unsolved and is arguably more fatal for the paper** than the
  accuracy: under `iid: true` the E-HSFP components *cannot* differentiate by construction.

---

## 6. Open questions for discussion

1. **Is prototype-transfer salvageable for CIFAR-100, or is it the wrong paradigm here?**
   Even the best backbone (vgg on HAM) only reached 53.6%. Should the method target
   *easier / fewer-class* settings where a per-class Gaussian is a reasonable model, and is
   that honest for the paper's claims?

2. **Covariance resampling (Fix 3): worth the risk?** Low-rank `N(μ, Σ)` should shrink the
   synthetic→real gap, but it touches memory/reliability/PRC. Is the likely payoff (few % → ?)
   enough to justify re-plumbing the prototype contract?

3. **The ablation must be run non-IID.** Should we switch the journal experiments to
   Dirichlet(α) partitioning so memory / reliability / PRC have imbalance + drift to correct?
   Without this, the ablation table is a null result. (Partitioner hook `_client_to_edge`
   already exists.)

4. **Framing / positioning.** Given H-SFP loses to baselines on accuracy, should the paper's
   contribution be re-centered on **communication cost / privacy / serverless robustness**
   (where prototype transfer genuinely wins) rather than accuracy parity? What does that require
   from the experiments?

5. **The 200-client deep-copy constraint** limits client capacity (can't hold a full backbone).
   Is refactoring to a *shared* client encoder (with only prototypes differing per client)
   acceptable within the FL/privacy story, and would it change the split-computation narrative?

6. **Grid hygiene.** The new architecture is now in the code, so future H-SFP CIFAR grid runs use
   it; the old 8.10% CIFAR markers are stale/mixed-arch. Reset and regenerate, or freeze the change
   behind a config flag so both variants can be compared cleanly?

---

## 7. Reproduction / where things live

- **Changed files:** `classification/H-SFP/models/cnn_cifar.py`, `classification/H-SFP/hierarchy.py`
- **Full-run config:** `configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml`
  (epochs 60, warmup 5, ssl 10/10/10, t1=5 t2=10, iid=true, 200 users)
- **Baseline results referenced:** 1 seed, 60 epochs, ImageNet-pretrained backbones.
- **Nothing committed yet;** the change is uncommitted in the working tree on branch `DungTT`.
