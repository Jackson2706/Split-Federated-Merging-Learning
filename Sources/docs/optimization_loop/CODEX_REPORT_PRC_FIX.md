Fixed locally in [ehsfp/losses.py](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/losses.py:45).

Cause: PRC performed two model forwards per shared class on every minibatch—200 tiny CUDA/BLAS dispatches for 100 CIFAR classes. At proxy scale, this saturated/stalled the worker pool and appeared as `futex_wait`. No unreleased lock, multiprocessing primitive, per-call DataLoader, or unbounded loop was found. Reliability works because it batches class processing.

Exact change:

- Lines 45–64: stack all shared-class current/memory distributions and sample them together.
- Lines 66–71: flatten class/sample dimensions and perform only two model forwards total.
- The global MSE is mathematically identical to averaging equal-sized per-class MSE losses.
- Existing PRC counters remain intact.
- No changes were needed in `hierarchy.py`; its existing uncommitted edits were preserved.

Validation:

- `python -m unittest -q tests.test_phase0_effects`: all 9 tests passed.
- CPU 100-class micro-test: loss `1650.3855`, gradient norm `5788.3008`, with a nonzero trainable gradient.
- `git diff --check` passed.
- No GPU/training was run.