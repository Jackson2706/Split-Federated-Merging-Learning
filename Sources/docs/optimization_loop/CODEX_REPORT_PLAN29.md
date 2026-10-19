## EXECUTION REPORT

PLAN-29 implementation is complete. No GPU/training runs were performed.

New work:

- Foreground-aware focal weighting (`alpha_fg=0.75`, `alpha_bg=0.25`) applied per pixel; Dice math unchanged:
  - [Federated DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/clients/DiceFocalLoss.py:59)
  - [HierFL DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/clients/DiceFocalLoss.py:59)
  - [HeteroSFL DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HeteroSFL/clients/DiceFocalLoss.py:54)
  - [H-SFP DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/clients/DiceFocalLoss.py:54)

- Federated best-round tracking:
  - Import/state: [runner.py:15](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/runner.py:15), [runner.py:53](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/runner.py:53)
  - Update/snapshot: [runner.py:113](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/runner.py:113)
  - Restore/report and JSON fields: [runner.py:136](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/runner.py:136), [runner.py:161](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/runner.py:161)

- HierFL best-round tracking:
  - Import/state: [runner.py:15](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/runner.py:15), [runner.py:48](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/runner.py:48)
  - Update/snapshot: [runner.py:101](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/runner.py:101)
  - Restore/report and JSON fields: [runner.py:125](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/runner.py:125), [runner.py:150](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/runner.py:150)

- Aggregator:
  - `best_iou` remains preferred over `test_iou`: [aggregate.py:126](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/aggregate.py:126)
  - `best_dice` now populates the paired segmentation metric instead of last-round Dice: [aggregate.py:142](/home/jackson/Desktop/Split-Federated-Merging-Learning/tools/fair_compare/aggregate.py:142)

CPU tests in [test_plan29_segmentation.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/tests/test_plan29_segmentation.py:38):

- Foreground-weighted loss exceeds old uniform-alpha loss for an all-background prediction: passed.
- Best metrics remain monotonic through a later collapse: passed.
- Prediction/mask shape alignment and subsequent loss computation: passed.
- Result: **3/3 passed**.

Validation:

- Aggregator synthetic best-round preference: passed.
- All four proxy configs resolve to `epochs: 10`.
- `git diff --check`: passed.
- AST/YAML parsing: **18 modified Python files and 5 modified YAML files passed cleanly**.
- All pre-existing uncommitted changes were preserved.