import ast
from pathlib import Path
import unittest

from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
METRIC_FILES = (
    ROOT / "classification/H-SFP/hierarchy.py",
    ROOT / "classification/HSFL/hierarchy.py",
    ROOT / "classification/SplitFL/runner.py",
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


if __name__ == "__main__":
    unittest.main()
