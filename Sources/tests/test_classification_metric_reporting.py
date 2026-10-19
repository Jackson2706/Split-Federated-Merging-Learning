import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from classification.training_metrics import (
    save_prediction_artifact,
    validate_prediction_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
METRIC_FILES = (
    ROOT / "classification/H-SFP/hierarchy.py",
    ROOT / "classification/HSFL/hierarchy.py",
    ROOT / "classification/SplitFL/runner.py",
    ROOT / "classification/HeteroSFL/runner.py",
)


class ClassificationMetricReportingTests(unittest.TestCase):
    def test_imbalanced_outputs_report_distinct_accuracy_and_macro_f1(self):
        targets = [0] * 90 + [1] * 5 + [2] * 5
        predictions = [0] * 100
        output = {
            "final_test_accuracy": accuracy_score(targets, predictions),
            "final_test_f1": f1_score(
                targets, predictions, average="macro", zero_division=0
            ),
        }

        self.assertAlmostEqual(output["final_test_accuracy"], 0.9)
        self.assertAlmostEqual(output["final_test_f1"], 6 / 19)
        self.assertGreater(
            output["final_test_accuracy"] - output["final_test_f1"], 0.5
        )

    def test_all_fixed_methods_compute_zero_safe_macro_f1(self):
        for path in METRIC_FILES:
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                calls = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "f1_score"
                ]
                self.assertTrue(calls)
                self.assertTrue(any(
                    any(k.arg == "average" and isinstance(k.value, ast.Constant)
                        and k.value.value == "macro" for k in call.keywords)
                    and any(k.arg == "zero_division" and isinstance(k.value, ast.Constant)
                            and k.value.value == 0 for k in call.keywords)
                    for call in calls
                ))

    def test_independent_validator_matches_hand_computed_case(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        predictions = np.array([0, 1, 1, 1, 0, 2])
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "predictions.npz"
            save_prediction_artifact(artifact, predictions, labels)
            metrics = validate_prediction_artifact(artifact)

        self.assertEqual(metrics["classes"], [0, 1, 2])
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 59 / 90)
        self.assertEqual(metrics["per_class_recall"], [0.5, 1.0, 0.5])
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 1, 0], [0, 2, 0], [1, 0, 1]],
        )

    def test_all_validation_methods_name_best_and_last_artifacts(self):
        for path in METRIC_FILES:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("best_val_predictions.npz", source)
                self.assertIn("last_round_predictions.npz", source)


if __name__ == "__main__":
    unittest.main()
