# Optimization Loop Project State

## PLAN-0 status

- Outcome: PASS (Phase-0 integrity criteria only; no accuracy claim and no training grid run).
- Authoritative tree: `/home/jackson/Desktop/Split-Federated-Merging-Learning`, branch `DungTT`.
- Audited commit: `e74388c084984e7eaeb3625ce46bff5e37051dac`.
- Initial dirty-worktree snapshot is preserved verbatim in `logs/codex_plan0_20260713_225256.log`; no pre-existing edits or results were reverted, deleted, or overwritten.
- Deterministic seed: CLI seed, with local default `42` when omitted.

## Real entry points and precedence

`main.py` is the unified entry point. Its registry dynamically loads `classification/H-SFP/runner.py` or `segmentation/H-SFP/runner.py`. Each runner calls its task-specific `ConfigLoader`, dataset loader, model registry, and `HierarchicalFL.train_end_to_end`; evaluation is performed in the same hierarchy class and a final task-specific test is performed by the runner.

Resolved precedence is: recursive YAML `base` (lowest) -> child YAML -> CLI `--ablation`/`--seed`/`--set` temporary YAML -> E-HSFP defaults -> resolved YAML E-HSFP values -> named ablation preset (highest for component flags). The final resolved YAML plus effective E-HSFP configuration is SHA-256 hashed. Every run records the resolved configuration, seed, commit, partition hash, architecture manifest/hash, and output path in `run_metadata.json`.

## Resolved model and training facts

The facts below were obtained by constructing the registered models locally with cached pretrained weights and running one no-gradient forward pass; see `logs/plan0_model_audit_20260713.log`.

| Task/config label | Actual tiers and pretrained state | Tensor path | Params/trainable by tier |
|---|---|---|---|
| classification CIFAR-100 `model: resnet50` | `ClientModel` is a torchvision ResNet-18 stem+layer1+layer2+GAP (ImageNet DEFAULT weights, but CIFAR conv1 is newly replaced); custom EdgeModel and CloudModel are random-init. This is a confirmed config-name mismatch. | `[1,3,32,32] -> [1,128,1,1] -> [1,256,1,1] -> [1,256] -> [1,100]` | client 675,392; edge 198,016; cloud 157,285; all trainable unless `HSFP_FREEZE_BACKBONE=1` |
| classification HAM10000 `model: resnet50` | ResNet-50 DEFAULT client stem+layer1; separately instantiated ResNet-50 DEFAULT edge layer2-4+GAP; random classifier head. | `[1,3,224,224] -> [1,256,56,56] -> [1,2048,1,1] -> [1,2048] -> [1,7]` | 225,344; 23,282,688; 1,182,215; all trainable |
| segmentation ISIC `model: resnet50` | ResNet-50 DEFAULT client stem+layer1; separately instantiated ResNet-50 DEFAULT edge layer2-4; random prototype classifier and random decoder. | `[1,3,224,224] -> [1,256,56,56] -> [1,2048,7,7] -> [1,2048] -> [1,2]` | client 225,344; edge 23,282,688; prototype cloud 1,050,114; decoder 11,085,697; all trainable |

Current defaults use 60 global rounds. Classification selects 20/200 clients per round; segmentation selects 5/50. Selected clients perform 10 client SSL epochs, active edges 10 synthetic SSL epochs, and cloud 10 synthetic supervised epochs; segmentation additionally performs 5 decoder epochs. Client-model aggregation occurs every `t1` rounds and edge-model aggregation every `t2` rounds. Adam uses LR `1e-4` and weight decay `1e-4`; classification enables 5-round linear warmup then cosine scheduling, while segmentation has no scheduler. Reliability bootstrap uses Adam LR `1e-3`, weight decay `1e-4` when enabled.

Classification evaluation runs every `eval_every` (default 1) in `eval()`/`torch.no_grad()` using client 0, its connected edge, and cloud 0, with top-1 accuracy. Segmentation similarly uses client 0, its connected edge, and the decoder, reporting IoU/Dice; its hierarchy currently evaluates on the runner's `test_dataset` argument.

## Component wiring

- Memory: client/edge records are stored each round, mixed by `mix_current_and_memory`, then aged.
- Reliability: `PrototypeReliabilityNetwork` supplies nonuniform aggregation weights and is bootstrapped from memory metadata.
- PRC: class-keyed current prototypes are now compared with the matching tier memory and the loss is added before optimizer backpropagation at edge/cloud.
- Dropout: `PrototypeDropout.apply_to_source_outputs` filters active client or edge IDs before aggregation.
- Runtime effects are persisted through `ehsfp_runtime_counters`, dropout stats, and unrounded JSON metrics.

## Unresolved anomalies

- `use_residual_generator` is present in the full preset, but both hierarchy constructors still leave `residual_generator = None`; this optional component was outside PLAN-0's four mandated effect tests.
- `use_dropout_consistency`/`dropout_consistency_loss` is defined but is not invoked by the main hierarchy training loops; PLAN-0 verified source-ID dropout itself, not the optional consistency regularizer.
- Runtime counters are unit-verified but have not yet been observed in a bounded real-data smoke run.
- Segmentation selects its best checkpoint on the runner-provided `test_dataset`, so split semantics/data leakage need review before scientific comparison.

## Output identity

H-SFP output is now reserved under `Figure/data/runs/<architecture_id>/<resolved_config_hash>/`. Existing directories cause a hard refusal. Checkpoints embed the resolved-config hash and the validator rejects mismatched hashes.
