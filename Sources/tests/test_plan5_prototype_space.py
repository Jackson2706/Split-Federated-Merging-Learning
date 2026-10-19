import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from ehsfp.prototype_space import (
    CosineClassifier,
    center_features,
    center_source_outputs,
    derive_global_feature_mean,
    validate_prototype_space,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_hierarchy():
    spec = importlib.util.spec_from_file_location(
        "classification_hsfp_hierarchy", ROOT / "classification/H-SFP/hierarchy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cifar_models():
    spec = importlib.util.spec_from_file_location(
        "classification_hsfp_cnn", ROOT / "classification/H-SFP/models/cnn_cifar.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Client(nn.Module):
    def forward(self, x):
        return x


class _Edge(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return 2.0 * x


class _Cloud(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, x):
        self.last_input = x.detach().clone()
        return x


class Plan5PrototypeSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hierarchy = _load_hierarchy()
        cls.cifar_models = _load_cifar_models()

    def test_raw_regression_is_numerically_identical_to_legacy_pipeline(self):
        torch.manual_seed(7)
        client, edge, cloud = _Client(), _Edge(), nn.Linear(3, 4)
        x = torch.randn(5, 3, 1, 1)

        legacy = cloud(torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(edge(client(x))), 1))
        raw = self.hierarchy.FullPipelineModel(client, edge, cloud)(x)

        self.assertEqual(validate_prototype_space(None), "raw")
        self.assertTrue(torch.equal(raw, legacy))

    def test_raw_cloud_head_matches_current_hsfp_formula(self):
        torch.manual_seed(11)
        model = self.cifar_models.CloudModel({"num_classes": 100}).eval()
        features = torch.randn(8, 256)
        expected = model.classifier(F.normalize(features, dim=1) * model.feat_scale)
        actual = model(features)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(hasattr(model, "cosine_head"))

    def test_centered_prototypes_have_zero_support_weighted_global_mean(self):
        outputs = {
            "a": ({0: torch.tensor([1.0, 3.0]), 1: torch.tensor([5.0, 7.0])},
                  {0: torch.ones(2), 1: torch.ones(2)}),
            "b": ({0: torch.tensor([9.0, 11.0])}, {0: torch.ones(2)}),
        }
        support = {"a": {0: 2, 1: 1}, "b": {0: 3}}
        mean = derive_global_feature_mean(outputs, support)
        centered = center_source_outputs(outputs, mean)
        post_mean = derive_global_feature_mean(centered, support)

        self.assertTrue(torch.allclose(post_mean, torch.zeros_like(post_mean), atol=1e-6))
        self.assertEqual(
            sum(t.numel() for pair in outputs.values() for d in pair for t in d.values()),
            sum(t.numel() for pair in centered.values() for d in pair for t in d.values()),
        )
        self.assertEqual(len(centered["a"]), 2)  # no transmitted mean tensor

    def test_cosine_head_is_scale_times_cosine(self):
        head = CosineClassifier(3, 2, initial_scale=7.0)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 2.0, 0.0], [-1.0, 0.0, 1.0]]))
        features = torch.tensor([[2.0, 1.0, 2.0], [0.0, 3.0, 4.0]])
        expected = 7.0 * F.normalize(features, dim=1) @ F.normalize(head.weight, dim=1).T
        self.assertTrue(torch.allclose(head(features), expected, atol=1e-7))
        self.assertTrue(head.scale.requires_grad)

    def test_train_and_eval_use_the_same_boundary_centering(self):
        client, edge, cloud = _Client(), _Edge(), _Cloud()
        mu_l1 = torch.tensor([[[1.0]], [[-2.0]], [[0.5]]])
        mu_l2 = torch.tensor([2.0, -4.0, 1.0])
        real = torch.tensor([[[[3.0]], [[1.0]], [[2.5]]]])
        pipeline = self.hierarchy.FullPipelineModel(
            client, edge, cloud, prototype_space="centered_cosine",
            client_to_edge_mean=mu_l1, edge_to_cloud_mean=mu_l2,
        )

        result = pipeline(real)
        training_l1 = center_features(real, mu_l1)
        training_l2 = center_features(2.0 * training_l1.flatten(1), mu_l2)

        self.assertTrue(torch.equal(edge.last_input, training_l1))
        self.assertTrue(torch.equal(cloud.last_input, training_l2))
        self.assertTrue(torch.equal(result, training_l2))

    def test_invalid_space_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_prototype_space("bogus_space")


if __name__ == "__main__":
    unittest.main()
