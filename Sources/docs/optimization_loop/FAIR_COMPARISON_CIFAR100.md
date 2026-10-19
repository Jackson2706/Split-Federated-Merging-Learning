# CIFAR-100 fair-comparison audit (PLAN-2)

## Verdict

**Overall fair match: FLAGGED / not yet suitable for a performance claim.** Through the unified
`main.py` entry point, all six methods can use the same explicit seed and the same IID partition;
the audited seed-42 partition hash is
`6954b58c404551a3fe3034c754f1fa242bf285fe5a2b0c5122de7c5f6e0cacb1` for every method. All six
compute top-1 accuracy as their primary accuracy value, even where variables or log labels say
`f1`. Those two high-priority checks therefore match.

The comparison as a whole does not match. The official 10,000-example CIFAR-100 test split is
returned as both `valid_dataset` and `test_dataset` (or is evaluated directly every round), so no
method currently has a validation-only protocol or an untouched test set. Local training work,
optimizer implementation, LR scheduling, evaluation frequency/path, and effective architectures
also differ. These flags must be resolved or represented by explicit matched controls before a
full baseline reproduction or a `B_d` claim.

## Audited configuration matrix

The matrix resolves the current CIFAR-100 ResNet configurations plus their inherited defaults:
`cifar_our_resnet50_10_20.yaml`, `cifar_splitfed_resnet50.yaml`,
`cifar_fedavg_resnet50.yaml`, `cifar_hierfl_resnet50.yaml`,
`cifar_our_resnet50_10_20.yaml` (HSFL), and `cifar_heteroSFL_resnet18.yaml`.
`heterosfl` below corresponds to the CLI spelling `hetero-sfl`.

| Method | Dataset partition and `partition_hash` | Seed handling | Primary metric and evaluation protocol | Backbone / pretraining | Rounds / clients / local work | Optimizer / scheduler | Fairness flags |
|---|---|---|---|---|---|---|---|
| h-sfp | CIFAR-100 train 50,000; IID `cifar_iid`, 250 disjoint indices for each of 200 clients. Seed 42 hash `6954b58c...cacb1`. Same normalization, no augmentation. | `main.py` seeds Python, NumPy, Torch, and CUDA; deterministic cuDNN. Default is 42; standalone runner has no seed guard. | `accuracy_score` after argmax = top-1. Representative pipeline: client 0 -> its edge -> cloud 0; every `eval_every` (default 1). Best snapshot is then evaluated again. **The validation and test objects are the same official test split.** | Config label `resnet50`, actually modified ImageNet-pretrained ResNet-18 through client layer2; replacement CIFAR stem, edge refiner, and cloud classifier are random-init. | 60 rounds; 20/200 selected. Per round: 10 client SSL epochs, 10 edge synthetic SSL epochs, 10 cloud synthetic supervised epochs; `t1=10`, `t2=20`. | Adam `1e-4`, weight decay `1e-4`; persistent per-tier optimizers; 5-round linear warmup then cosine schedule. | **FLAG:** local work and schedule differ from all baselines. **FLAG:** architecture/training objective is method-specific. **FLAG:** test-as-validation leakage. |
| splitfl | Same IID implementation and seed-42 hash `6954b58c...cacb1`; same normalization/no augmentation. | Same unified seeding; no standalone runner seed guard. | `accuracy_score` after argmax = top-1, despite `eval_f1`/`F1` names. Merged averaged-client + shared-server pipeline every round and at end. **Validation=test object.** | Config label `resnet50`, actually split ImageNet-pretrained ResNet-18: separately instantiated pretrained client layer1-2 and server layer3-4; CIFAR stem and FC random-init. | 60 rounds; 20/200; 5 local epochs per selected client. | **Hard-coded SGD** `1e-4`, momentum `0.9`, no weight decay/scheduler. Config says `optimizer: adam` but runner ignores it. Server optimizer is recreated each round. | **FLAG:** optimizer/config mismatch; not Adam. **FLAG:** 5 local epochs vs H-SFP tier work. **FLAG:** separate pretrained instances and split objective. **FLAG:** test-as-validation leakage. |
| federated | Same IID implementation and seed-42 hash `6954b58c...cacb1`; same normalization/no augmentation. | Same unified seeding; no standalone runner seed guard. | `accuracy_score` after argmax = top-1 on the global model every round and at end. It directly names/uses the official test split; no validation split. | Config label `resnet50`, actually one modified ImageNet-pretrained ResNet-18; CIFAR stem and FC random-init. | 60 rounds; 20/200; 5 local epochs per selected client. | Adam `1e-4`, hard-coded weight decay `1e-4`; optimizer recreated for each client; no scheduler. | **FLAG:** direct repeated test evaluation. **FLAG:** local work/scheduler differ from H-SFP. **FLAG:** full-model architecture differs from split methods. |
| hierfl | Same IID implementation and seed-42 hash `6954b58c...cacb1`; same normalization/no augmentation. | Same unified seeding; no standalone runner seed guard. | `accuracy_score` after argmax = top-1 on the global model every round and at end; direct official-test evaluation. Additional hierarchy diagnostics can also test server models on that split. | Same effective full modified ImageNet-pretrained ResNet-18 as federated; CIFAR stem and FC random-init. | 60 rounds; 20/200; 5 local epochs; five intermediate servers (`mid_server: [5]`). | Adam `1e-4`, hard-coded weight decay `1e-4`; per-client optimizer recreation; no scheduler. | **FLAG:** direct repeated test evaluation. **FLAG:** local work/scheduler differ from H-SFP. **FLAG:** hierarchy and full-model path differ. |
| hsfl | Same IID implementation and seed-42 hash `6954b58c...cacb1`; same normalization/no augmentation. | Same unified seeding; no standalone runner seed guard. | `accuracy_score` after argmax = top-1. Representative client 0 -> its edge -> cloud pipeline only after cloud aggregation (`t2=20`) or final round; best snapshot tested again. **Validation=test object.** | Config label `resnet50`, actually split ImageNet-pretrained ResNet-18: separately instantiated client layer1 and edge layer2-4; CIFAR stem and cloud classifier random-init. | 60 rounds; 20/200; 5 local end-to-end split epochs; `t1=10`, `t2=20`. | Adam `1e-4`, weight decay `1e-4`; persistent server and per-client optimizers; no scheduler. | **FLAG:** evaluation frequency differs. **FLAG:** local work/scheduler differ from H-SFP. **FLAG:** separate pretrained instances. **FLAG:** test-as-validation leakage. |
| heterosfl | Same IID algorithm and seed-42 hash `6954b58c...cacb1`; same normalization/no augmentation. Unlike the other loaders, it has no optional camera-ready Dirichlet hook, but IID is matched. | Same unified seeding; no standalone runner seed guard. NumPy also fixes the high/low client-type shuffle under the explicit seed. | Primary selection uses top-1 `accuracy_score`; macro-F1 is also logged. Evaluation uses only the wide client + adapter/server every `print_every` (1), then final test; low-client paths are not evaluated. **Validation=test object.** | Explicit `resnet18`; separately instantiated ImageNet-pretrained client layer1-2 and server layer3-4, plus random compression/decoder/adapter/FC. | 60 rounds; 20/200; 5 local epochs; 30% high-end and 70% narrow clients. | **Hard-coded SGD** `1e-4`, momentum `0.9`, no weight decay/scheduler, despite config saying Adam. Server optimizer recreated each round. | **FLAG:** optimizer/config mismatch. **FLAG:** wide-only evaluation does not cover low clients. **FLAG:** heterogeneous architecture/objective. **FLAG:** test-as-validation leakage. |

## Exact match checks

| Check | Verdict | Evidence / consequence |
|---|---|---|
| Identical CIFAR-100 partition with identical seed? | **MATCH, conditional on unified explicit-seed launch.** | Resetting NumPy to seed 42 before each method's actual `cifar_iid` implementation gives the identical full-coverage 200-client hash `6954b58c...cacb1`. The PLAN-2 proxy seed 20260714 similarly gives `93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1`. Running method runners outside `main.py`, omitting a controlled seed in another entry point, or consuming NumPy RNG before partitioning invalidates the match. |
| Is the primary metric top-1 accuracy? | **MATCH in computation; FLAG in naming.** | All primary accuracy paths use argmax plus `sklearn.metrics.accuracy_score`. Several fields/logs are misleadingly called `f1`; HeteroSFL separately reports genuine macro-F1 as a secondary metric. |
| Same evaluation data and protocol? | **MISMATCH / fairness blocker.** | All use the same official CIFAR-100 test images, but they reuse them for checkpoint selection or every-round reporting. Representative split paths, global models, evaluation frequency, and HeteroSFL's wide-only path differ. There is no untouched test set. |
| Same backbone/pretraining? | **MISMATCH.** | All draw some weights from ImageNet-pretrained ResNet-18, but the split points, independently instantiated pretrained pieces, random heads/adapters, and trained objectives differ. The `resnet50` label is inaccurate for five methods. |
| Same rounds/clients/local epochs? | **PARTIAL.** | Historical defaults match at 60 rounds and 20/200 clients, but baseline local work is 5 epochs while H-SFP performs three distinct 10-epoch tier phases. |
| Same optimizer/scheduler? | **MISMATCH.** | H-SFP uses Adam plus warmup/cosine; Federated/HierFL/HSFL use Adam without a scheduler; SplitFL/HeteroSFL hard-code SGD while declaring Adam. |

## PLAN-2 bounded proxy controls

The four proxy configs use full CIFAR-100, IID, seed `20260714`, 200 total clients, 20 selected per
round, and 10 global rounds. The resulting common partition hash is
`93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1`.
Only global rounds are reduced; method-specific local work and optimizers are preserved so no
baseline is weakened. Consequently these are plumbing/runtime/convergence proxies, not a fully
matched performance comparison.

| Proxy | Config-prep resolved hash | Declared unique output directory |
|---|---|---|
| SplitFL | `cb01b9aac887ae57e9e364c95f6b25b5deac8cc25c5177f6472d0224dcec21cc` | `outputs/proxy/plan2_cifar100_splitfl_seed20260714` |
| Federated/FedAvg | `da304a30a69430804f638514b4c2c3a3fe49609e9b658a46a8ca2c343a97f2c1` | `outputs/proxy/plan2_cifar100_federated_seed20260714` |
| H-SFP baseline | `817f3265d5bec70c72df50cb009c9de5770c8cc660e00418c49d30ca77a7555b` | `outputs/proxy/plan2_cifar100_hsfp_baseline_seed20260714` |
| Full E-HSFP | `0018e9c95ab3de020785d6fe91940fbd4ddf4268c2ba66765e10a743be4d43bb` | `outputs/proxy/plan2_cifar100_hsfp_full_e_seed20260714` |
| HierFL | PLAN-15 resolved config; seed 20260714 partition `93e52364...dcb1` | `outputs/proxy/plan15_hierfl_seed20260714` |
| HSFL | PLAN-15 resolved config; seed 20260714 partition `93e52364...dcb1` | `outputs/proxy/plan15_hsfl_seed20260714` |
| HeteroSFL | PLAN-15 resolved config; seed 20260714 partition `93e52364...dcb1` | `outputs/proxy/plan15_heterosfl_seed20260714` |

### PLAN-15 baseline controls

These additional proxy rows use the same full CIFAR-100 IID partition, explicit seed `20260714`,
200 clients, 20 selected clients, normalization, and 10-round budget as SplitFL and Federated.
At that seed all three resolve to partition hash
`93e523640b9820c12f7cbf570ecb253a8610531942569577831a8b34d822dcb1`.
The primary metric is top-1 accuracy on the shared test-as-validation proxy protocol. Method-local
architectures and optimization are deliberately retained rather than weakened.

| Method | Partition / metric | Backbone | Rounds / local work | Optimizer / scheduler |
|---|---|---|---|---|
| HierFL | CIFAR-100 IID, 200 clients, seed-controlled hash above; top-1 | Modified ImageNet-pretrained ResNet-18 (`resnet50` legacy label) | 10 rounds, 20 selected, 5 local epochs; 5 intermediate servers | Adam `1e-4`, weight decay `1e-4`; no scheduler |
| HSFL | Same partition/hash; top-1 | Split ImageNet-pretrained ResNet-18 (`resnet50` legacy label) | 10 rounds, 20 selected, 5 end-to-end split epochs; `t1=10`, `t2=20` | Adam `1e-4`, weight decay `1e-4`; no scheduler |
| HeteroSFL | Same partition/hash; top-1 (macro-F1 secondary) | Heterogeneous split ResNet-18, 30% wide / 70% narrow | 10 rounds, 20 selected, 5 local epochs | Hard-coded SGD `1e-4`, momentum `0.9`; no scheduler |

The hashes above apply the repository's canonical resolved-config hashing helper to all four
resolved YAMLs; the H-SFP runner enforces this hash at runtime. SplitFL and Federated do not yet
persist that hash in their result JSON, which remains an integrity flag. Their runners now consume
the optional proxy `output_dir` and reserve it with `exist_ok=False`; configs without that key keep
the legacy method-local path. H-SFP uses each declared directory as the root for its existing
architecture/hash-addressed run path. Thus all four proxy launches are isolated and fail rather
than overwrite an existing proxy directory.

## Exact host launch commands

Run sequentially from the repository root; no command enables W&B.

```bash
python main.py --task classification --method splitfl --cfg configs/proxy/plan2_cifar100_splitfl.yaml --seed 20260714
python main.py --task classification --method federated --cfg configs/proxy/plan2_cifar100_federated.yaml --seed 20260714
python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_baseline.yaml --ablation baseline_hsfp --seed 20260714
python main.py --task classification --method h-sfp --cfg configs/proxy/plan2_cifar100_hsfp_full_e.yaml --ablation full_e_hsfp --seed 20260714
```
