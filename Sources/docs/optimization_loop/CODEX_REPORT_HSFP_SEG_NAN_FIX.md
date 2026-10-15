Implemented locally while preserving all existing uncommitted changes and diagnostics.

- [segmentation/H-SFP/hierarchy.py:145](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/hierarchy.py:145): SupCon computation now disables autocast and casts features to float32. This protects both client and edge SSL loss calls while retaining autocast for model forwards.
- [segmentation/H-SFP/hierarchy.py:530](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/H-SFP/hierarchy.py:530): added a clear warning listing any classes whose client prototypes contain NaNs.
- [classification/H-SFP/hierarchy.py:215](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/hierarchy.py:215): classification had no existing fp32 protection, so its shared client/edge SupCon loss received the same fix.
- Existing `HSFP_SEG_DIAG` instrumentation remains intact.

Validation:

- `python -m unittest discover -s tests -v`: **27/27 passed**
- `make check`: **passed**—compilation, registry, and partition self-test
- `git diff --check`: **passed**
- No GPU or training runs performed.