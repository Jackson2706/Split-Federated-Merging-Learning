import ast
from pathlib import Path
import unittest

import torch

from segmentation.training_metrics import compute_iou_and_dice


ROOT = Path(__file__).resolve().parents[1]
CALL_SITES = (
    ROOT / "segmentation/HierFL/clients/test.py",
    ROOT / "segmentation/Federated/clients/Client.py",
    ROOT / "segmentation/Federated/clients/test.py",
    ROOT / "segmentation/HeteroSFL/clients/test.py",
    ROOT / "segmentation/H-SFP/clients/test.py",
)


class SegmentationMetricProtocolTests(unittest.TestCase):
    def test_all_five_call_sites_use_the_canonical_function(self):
        for path in CALL_SITES:
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                local_definitions = [
                    node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "compute_iou_and_dice"
                ]
                canonical_imports = [
                    node for node in tree.body
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "segmentation.training_metrics"
                    and any(
                        alias.name == "compute_iou_and_dice"
                        for alias in node.names
                    )
                ]
                self.assertFalse(local_definitions)
                self.assertEqual(len(canonical_imports), 1)

    def test_empty_masks_return_exactly_zero_zero(self):
        empty = torch.zeros((2, 1, 3, 3))
        self.assertEqual(compute_iou_and_dice(empty, empty), (0, 0))

    def test_whole_batch_flattened_protocol_and_strict_threshold(self):
        predictions = torch.tensor([[[[0.5, 0.9]]], [[[0.9, 0.1]]]])
        labels = torch.tensor([[[[1.0, 1.0]]], [[[0.0, 0.0]]]])
        iou, dice = compute_iou_and_dice(predictions, labels)
        self.assertAlmostEqual(iou, 1 / 3)
        self.assertAlmostEqual(dice, 1 / 2)


if __name__ == "__main__":
    unittest.main()
