Fixed the dtype crash generically.

Changes:

- [segmentation/HeteroSFL/runner.py:68](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HeteroSFL/runner.py:68)
  - Skips accumulation and averaging for non-floating tensors.
  - Integer/boolean buffers retain the existing global value and dtype.

- [segmentation/Federated/server/FedAvgAggregator.py:21](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/Federated/server/FedAvgAggregator.py:21)
  - Fixed the analogous FedAvg bug by preserving the first-client value for non-floating buffers.

- [segmentation/HierFL/servers/FedAvgAggregator.py:15](/home/jackson/Desktop/Split-Federated-Merging-Learning/segmentation/HierFL/servers/FedAvgAggregator.py:15)
  - Replaced the limited float dtype whitelist with generic `is_floating_point()` handling.

Root cause confirmed: float division of integer state entries such as BatchNorm `num_batches_tracked` produced a float source that could not be assigned into the original `torch.long` destination.

CPU-only validation passed:

- Floating weights were averaged correctly.
- `torch.long` and `torch.bool` buffers retained their original dtype and value.
- HeteroSFL, Federated, and HierFL paths were checked.
- Python compilation and `git diff --check` passed.
- No GPU or training was run.
- Existing uncommitted changes were preserved.