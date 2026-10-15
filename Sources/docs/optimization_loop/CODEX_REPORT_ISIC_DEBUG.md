The runs expose two independent failures: the reported IoU is invalid due to double sigmoid, while the code=1 failures occur before any model executes.

## (a) Why IoU reaches 0

The segmentation models already return one-channel probabilities:

- [CNN_ISIC.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/models/CNN_ISIC.py:63) applies `torch.sigmoid(out)`.
- HierFL uses the same architecture.
- HeteroSFL and H-SFP models also end in sigmoid.

But federated/HierFL evaluation applies sigmoid again in [test.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/clients/test.py:59), then thresholds at `> 0.5`.

For a model probability `p ∈ (0,1)`, `sigmoid(p) ∈ (0.5,0.731)`. Thus almost every finite pixel becomes foreground. The early reported IoUs around 23–28% are therefore approximately lesion-area prevalence, not learned segmentation quality.

Once model probabilities saturate to exactly zero—or invalid values are converted to zero—the second sigmoid produces exactly `0.5`, and strict `> 0.5` produces all-background. That explains the abrupt transition to IoU 0. It is a metric/prediction-path bug, not a class-dimension or `argmax` issue.

There is also a training-stability problem currently hidden by uncommitted loss changes. [DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/clients/DiceFocalLoss.py:28) converts NaN predictions to zero with `nan_to_num`. That silently turns numerical failure into all-background and supplies no useful gradient at invalid elements. A loss around 1.25 is consistent with near-zero predictions: Dice loss near 1 plus foreground focal penalty.

Additional loss concerns:

- The focal implementation multiplies every pixel by the same `alpha=0.25`; it does not use separate foreground/background weights. Therefore it does not actually correct lesion/background imbalance.
- Dice and focal weights are fixed 1:1 and cannot be configured.
- Independent unsorted image/mask `glob()` lists in the ISIC datasets are unsafe. They should be paired by ISIC identifier, although the repeatable collapse does not prove that mismatching occurred in these runs.

## Configuration assessment

The proxy YAMLs contain only `base` plus `output_dir`; the effective settings come from `configs/segmentation/*`.

For federated and HierFL:

- Adam, LR `1e-4`, local epochs 5 and batch size 16 match the repository’s existing segmentation defaults and are reasonable starting values.
- 60 global epochs is not intrinsically unreasonable.
- Binary segmentation correctly uses a one-channel sigmoid output. A `num_classes` value is unnecessary for these models.

For HeteroSFL:

- `num_classes: 2` is conceptually inconsistent with its one-channel sigmoid/BCE-Dice decoder, although it appears unused in the segmentation output path. Prefer `num_classes: 1` or remove it.
- More importantly, [runner.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HeteroSFL/runner.py:135) ignores `optimizer: adam` and hard-codes SGD at `1e-4` for both client and server. That LR is extremely conservative for SGD and is not comparable to the working Adam defaults.

## (b) code=1 crashes

### HeteroSFL

All three logs fail while importing the runner:

```text
main.py:314  main()
main.py:304  run(...)
main.py:244  runner = _load_runner(method_dir)
main.py:106  spec.loader.exec_module(module)
segmentation/HeteroSFL/runner.py:10
    from config import ConfigLoader
ImportError: cannot import name 'ConfigLoader' from 'config' (unknown location)
```

Root cause: `segmentation/HeteroSFL/config/` has `config_loader.py` but no `__init__.py`, so `from config import ConfigLoader` cannot work. Other runners either export `ConfigLoader` from `config/__init__.py` or import it explicitly.

This is an import/package bug, not an ISIC tensor-shape or model bug. Training never began, so the suspected HeteroSFL model shape path was not exercised.

### H-SFP

There is no traceback because runner loading never starts:

```text
Error: config not found:
.../configs/proxy/plan_isic2018_h-sfp.yaml
Tip: python main.py --list-configs --task segmentation --method h-sfp
EXIT ... code=1
```

The actual file is [plan_isic2018_hsfp.yaml](/home/jackson/Desktop/Split-Federated-Merging-Learning/configs/proxy/plan_isic2018_hsfp.yaml:1), without the hyphen between `h` and `sfp`. The checked-in fair-comparison script already references the correct filename, so the plan-26 launcher/invocation used an incorrect method-to-config filename. No H-SFP shape/model code ran.

## Minimal proposed fixes

1. Establish one consistent output contract:

   - Minimal: keep model sigmoid outputs and remove sigmoid from every segmentation evaluation path; threshold probabilities directly with `outputs >= 0.5`.
   - Cleaner: return logits, use `BCEWithLogitsLoss` plus a logits-compatible soft-Dice loss, and apply sigmoid exactly once for metrics.

2. Replace `nan_to_num` on predictions with an explicit finite-value assertion/failure. Numerical corruption should stop the run rather than be reported as all-background.

3. Correct focal class weighting, e.g. foreground `alpha` and background `1-alpha`, or use configurable `pos_weight` with BCE. Make Dice/BCE weights configurable.

4. Fix HeteroSFL import with:

   ```python
   from config.config_loader import ConfigLoader
   ```

   or add `config/__init__.py` exporting `ConfigLoader`.

5. Make HeteroSFL honor `config["optimizer"]`; for the existing fair settings, use Adam at `1e-4`. If retaining SGD, give it a separately tuned LR.

6. Change/remove HeteroSFL and H-SFP `num_classes: 2` for the one-channel binary decoder.

7. Fix the plan-26 H-SFP config mapping to `plan_isic2018_hsfp.yaml`.

8. Pair ISIC images and masks by filename stem and use nearest-neighbor mask resizing.

No files were modified and no GPU/training commands were run. Existing uncommitted changes were preserved.