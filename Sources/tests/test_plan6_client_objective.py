import inspect
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ehsfp import get_ehsfp_config, resolved_config_hash
from ehsfp.client_objective import (
    compose_client_objective_loss,
    local_supervised_contrastive_loss,
    validate_client_objective,
)


class Plan6ClientObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "classification_hsfp_hierarchy_plan6",
            root / "classification/H-SFP/hierarchy.py",
        )
        cls.hierarchy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.hierarchy)

    def test_ssl_regression_returns_bitwise_identical_loss_tensor(self):
        ssl_loss = torch.tensor(1.234567, requires_grad=True)
        actual = compose_client_objective_loss(ssl_loss, "ssl")
        self.assertIs(actual, ssl_loss)
        self.assertTrue(torch.equal(actual, ssl_loss))
        self.assertEqual(validate_client_objective(None), "ssl")

    def test_ssl_client_step_is_bitwise_identical_to_legacy_formula(self):
        torch.manual_seed(61)
        legacy_model = nn.Conv2d(2, 3, kernel_size=1)
        actual_model = nn.Conv2d(2, 3, kernel_size=1)
        actual_model.load_state_dict(legacy_model.state_dict())
        data = torch.randn(4, 2, 1, 1)
        labels = torch.tensor([0, 0, 1, 1])

        legacy_optimizer = torch.optim.Adam(legacy_model.parameters(), lr=1e-3)
        legacy_optimizer.zero_grad(set_to_none=True)
        z1, z2 = legacy_model(data), legacy_model(data)
        z1 = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z1), 1)
        z2 = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(z2), 1)
        legacy_loss = self.hierarchy.supervised_contrastive_loss(
            torch.cat([z1, z2]), torch.cat([labels, labels]), temperature=0.1
        )
        legacy_loss.backward()
        legacy_optimizer.step()

        trainer = self.hierarchy.HierarchicalFL.__new__(self.hierarchy.HierarchicalFL)
        trainer.device = torch.device("cpu")
        trainer.structure = {-1: {0: actual_model}}
        trainer.optimizers = {-1: {0: {
            "optimizer": torch.optim.Adam(actual_model.parameters(), lr=1e-3),
            "scaler": torch.amp.GradScaler(enabled=False),
        }}}
        trainer.ssl_transforms = nn.Identity()
        trainer.supcon_temp = 0.1
        trainer.client_objective = "ssl"
        trainer.client_objective_weight = 0.1
        trainer.client_supervised_heads = {}
        trainer.client_supervised_optimizers = {}
        loader = DataLoader(TensorDataset(data, labels), batch_size=4, shuffle=False)
        trainer._client_ssl_extraction_phase(0, loader, ssl_epochs=1)

        for key, expected in legacy_model.state_dict().items():
            self.assertTrue(torch.equal(actual_model.state_dict()[key], expected), key)

    def test_ssl_supcon_is_finite_nonzero_and_reaches_encoder(self):
        torch.manual_seed(6)
        encoder = nn.Linear(5, 4)
        features = encoder(torch.randn(6, 5))
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        auxiliary = local_supervised_contrastive_loss(features, labels)
        self.assertTrue(torch.isfinite(auxiliary))
        self.assertGreater(auxiliary.item(), 0)
        auxiliary.backward()
        self.assertGreater(encoder.weight.grad.norm().item(), 0)

    def test_supervised_is_finite_nonzero_and_reaches_encoder(self):
        torch.manual_seed(7)
        encoder, head = nn.Linear(5, 4), nn.Linear(4, 3)
        features = encoder(torch.randn(6, 5))
        labels = torch.tensor([0, 1, 2, 0, 1, 2])
        loss = compose_client_objective_loss(
            features.sum() * 0, "supervised", features, labels,
            linear_head=head, weight=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(encoder.weight.grad.norm().item(), 0)

    def test_auxiliary_api_has_no_cross_tier_inputs(self):
        for function in (local_supervised_contrastive_loss, compose_client_objective_loss):
            parameters = set(inspect.signature(function).parameters)
            self.assertFalse(parameters & {
                "edge_features", "cloud_features", "edge_outputs", "prototypes",
                "memory", "client_outputs",
            })

    def test_invalid_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_client_objective("cross_tier")

    def test_proxy_objectives_resolve_to_distinct_hashes(self):
        base = {
            "dataset": "cifar100",
            "seed": 20260714,
            "epochs": 10,
            "ablation_mode": "baseline_hsfp",
        }
        hashes = {
            objective: resolved_config_hash(
                {**base, "client_objective": objective},
                get_ehsfp_config({**base, "client_objective": objective}),
            )
            for objective in ("ssl", "supervised")
        }
        self.assertNotEqual(hashes["ssl"], hashes["supervised"])


if __name__ == "__main__":
    unittest.main()
