Implemented a bounded PRC replay graph while preserving trainable PRC gradients.

- [ehsfp/losses.py:14](/home/jackson/Desktop/Split-Federated-Merging-Learning/ehsfp/losses.py:14)
  - Added `max_replay_batch=128`.
  - CIFAR-100 now uses one replay pair per shared class instead of 16, reducing each PRC branch from 1,600 to 100 edge-model examples.
  - Detached current/memory prototype statistics before sampling, preventing accidental retention of upstream extraction graphs.
  - Both current and memory representations still pass through the trainable model and contribute MSE gradients.

- [tests/test_phase0_effects.py:168](/home/jackson/Desktop/Split-Federated-Merging-Learning/tests/test_phase0_effects.py:168)
  - Added a 100-class regression test confirming bounded forward batches, nonzero PRC loss, and nonzero model parameter gradients.

Identified mechanism: an oversized combined autograd graph, not an autocast/scaler mismatch. Both edge PRC and SSL already execute under the same autocast context, and the summed loss is scaled once. Multiple forwards through the same module are valid, and no `retain_graph` or relevant in-place operation exists. The problematic scale was two retained 1,600-example edge-model PRC graphs plus the SSL graphs in every minibatch.

Validation:

```text
Ran 24 tests in 1.795s
OK
```

`git diff --check` passed. The full proxy verification requires CUDA (`require_cuda: true`), so it was not run locally. Unrelated uncommitted changes were preserved.