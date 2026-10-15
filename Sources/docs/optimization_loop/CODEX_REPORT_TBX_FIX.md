Eliminated the E-HSFP TensorBoard deadlock source.

- [classification/H-SFP/runner.py:13](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/runner.py:13): removed `tensorboardX.SummaryWriter` import.
- [classification/H-SFP/runner.py:25](/home/jackson/Desktop/Split-Federated-Merging-Learning/classification/H-SFP/runner.py:25): removed the former `SummaryWriter("./logs")` construction before training.

That unused construction was the only TensorBoard writer in classification H-SFP. Although no events were logged through it, initialization launched tensorboardX’s asynchronous `EventFileWriter` thread, allowing it to contend with PyTorch autograd. H-SFP training now creates no TensorBoard writer or background writer thread.

JSONL logging is unchanged:

- [ehsfp/metrics_logger.py:94](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/metrics_logger.py:94) still combines every current E-HSFP metric with the epoch’s extra metrics.
- [ehsfp/metrics_logger.py:118](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/metrics_logger.py:118) still appends the complete record to `metrics_unrounded.jsonl`.
- No metric collection or `append_jsonl` call was modified.

Validation:

- Python compilation passed.
- JSONL round-trip preservation check passed, including task, PRC, dropout, contrastive, and extra epoch fields.
- AST check confirmed classification H-SFP contains no `tensorboardX`, `SummaryWriter`, or `FileWriter`.
- `git diff --check` passed.
- `pytest` is unavailable in the environment.
- `unittest discover` completed but found no registered tests.
- No GPU run was performed, as requested.
- Existing uncommitted changes were preserved.