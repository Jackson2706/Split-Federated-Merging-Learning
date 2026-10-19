Implemented the same robust Dice+Focal numerical guards across all four segmentation methods:

- [Federated DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/clients/DiceFocalLoss.py:30)
  - Non-finite sanitization and clamping: lines 30–33
  - Stabilized Dice numerator/denominator: lines 42–44
  - Empty-tensor focal guard and probability clamp: lines 53–56
  - Autocast-disabled float32 computation: lines 64–66

- [HierFL DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/clients/DiceFocalLoss.py:30)
  - Same changes at lines 30–33, 42–44, 53–56, and 64–66

- [HeteroSFL DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HeteroSFL/clients/DiceFocalLoss.py:28)
  - Same changes at lines 28–31, 38–40, 48–51, and 59–61

- [H-SFP DiceFocalLoss.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/clients/DiceFocalLoss.py:28)
  - Same changes at lines 28–31, 38–40, 48–51, and 59–61

Guards added:

- Loss runs in float32 with autocast disabled.
- NaN/Inf predictions and targets are converted to valid binary-range values.
- Probabilities are clamped to `[eps, 1-eps]` before BCE/focal logarithms.
- Dice retains existing smoothing and adds epsilon to numerator and denominator.
- Empty tensors return a finite, gradient-connected zero focal loss.
- All-zero masks remain finite.

Validation passed:

- `python -m py_compile` on all four files
- `git diff --check`
- CPU-only checks for float16 inputs, all-zero masks, empty tensors, and NaN/±Inf inputs
- Existing uncommitted documentation changes were preserved.