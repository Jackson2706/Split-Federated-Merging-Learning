import importlib.util
import random
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from ehsfp.integrity import index_batch_hash, partition_hash, state_dict_hash


ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _short_smoke(seed):
    """Cheap proxy for init/partition/sampling and a seeded training metric."""
    _seed(seed)
    sampling = _load("classification/H-SFP/data/sampling.py", f"sampling_{seed}")
    partition = sampling.ham10000_iid(list(range(80)), 8)
    model = nn.Linear(4, 3)
    initial_hash = state_dict_hash(model.state_dict())
    sampled_clients = np.random.choice(range(8), 3, replace=False)

    features = torch.randn(24, 4)
    labels = torch.arange(24) % 3
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(features), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    fingerprints = {
        "partition": partition_hash(partition),
        "initial_model": initial_hash,
        "first_sampled_clients": index_batch_hash(sampled_clients),
    }
    return fingerprints, losses


class HSFPSeedAndCollapseTests(unittest.TestCase):
    def test_state_dict_hash_handles_batch_norm_scalar_buffer(self):
        model = nn.BatchNorm2d(3)
        state = model.state_dict()

        self.assertEqual(state["num_batches_tracked"].dim(), 0)
        initial_hash = state_dict_hash(state)
        self.assertEqual(initial_hash, state_dict_hash(state))

        changed_state = {name: value.clone() for name, value in state.items()}
        changed_state["num_batches_tracked"].add_(1)
        self.assertNotEqual(initial_hash, state_dict_hash(changed_state))

    def test_same_seed_reproduces_fingerprints_and_short_smoke_metrics(self):
        first_fingerprints, first_losses = _short_smoke(7)
        second_fingerprints, second_losses = _short_smoke(7)

        self.assertEqual(first_fingerprints, second_fingerprints)
        np.testing.assert_allclose(first_losses, second_losses, rtol=0, atol=1e-8)

    def test_different_seeds_change_partition_init_and_client_batch(self):
        seed_zero, _ = _short_smoke(0)
        seed_one, _ = _short_smoke(1)

        self.assertNotEqual(seed_zero["partition"], seed_one["partition"])
        self.assertNotEqual(seed_zero["initial_model"], seed_one["initial_model"])
        self.assertNotEqual(
            seed_zero["first_sampled_clients"], seed_one["first_sampled_clients"]
        )

    def test_prototype_statistics_promote_fp16_before_variance(self):
        hierarchy = _load("classification/H-SFP/hierarchy.py", "hsfp_hierarchy_seed_test")
        # 400**2 exceeds fp16's finite range. The pre-fix fp16 std reduction
        # returned inf here and poisoned edge/cloud losses with NaN.
        features = torch.tensor(
            [[400.0, -400.0], [-400.0, 400.0], [300.0, -300.0]],
            dtype=torch.float16,
        )
        labels = torch.tensor([0, 0, 1])

        prototypes, sigmas = hierarchy.calculate_prototypes_and_distribution(
            features, labels
        )

        self.assertEqual(sigmas[0].dtype, torch.float32)
        self.assertTrue(torch.isfinite(prototypes[0]).all())
        self.assertTrue(torch.isfinite(sigmas[0]).all())
        self.assertGreater(sigmas[0].min().item(), 300.0)


if __name__ == "__main__":
    unittest.main()
