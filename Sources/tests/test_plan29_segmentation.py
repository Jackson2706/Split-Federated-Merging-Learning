import ast
import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from segmentation.training_metrics import BestSegmentationMetrics


ROOT = Path(__file__).resolve().parents[1]
LOSS_PATHS = [
    ROOT / "segmentation" / method / "clients" / "DiceFocalLoss.py"
    for method in ("Federated", "HierFL", "HeteroSFL", "H-SFP")
]


def load_loss(path):
    spec = importlib.util.spec_from_file_location(f"plan29_loss_{path.parent.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DiceFocalLoss


def load_alignment_helper():
    path = ROOT / "segmentation" / "HeteroSFL" / "runner.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "align_prediction_to_mask"
    )
    namespace = {"F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["align_prediction_to_mask"]


class Plan29SegmentationTests(unittest.TestCase):
    def test_foreground_weighting_increases_all_background_penalty(self):
        focal_functions = []
        for path in LOSS_PATHS:
            tree = ast.parse(path.read_text())
            loss_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
            focal_functions.append(next(
                node for node in loss_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "focal_loss"
            ))
        focal_dumps = [ast.dump(node, include_attributes=False) for node in focal_functions]
        self.assertTrue(all(dump == focal_dumps[0] for dump in focal_dumps[1:]))

        loss = load_loss(LOSS_PATHS[0])()
        target = torch.zeros(1, 1, 20, 20)
        target[..., 0, 0] = 1.0
        prediction = torch.full_like(target, 1e-4)

        new_focal = loss.focal_loss(prediction, target)
        bce = F.binary_cross_entropy(prediction, target, reduction="none")
        old_uniform_focal = (loss.alpha * (1 - torch.exp(-bce)) ** loss.gamma * bce).mean()
        self.assertGreater(new_focal.item(), old_uniform_focal.item())

    def test_best_metrics_survive_later_collapse(self):
        best = BestSegmentationMetrics()
        history = [(0.12, 0.20), (0.31, 0.47), (0.25, 0.40), (0.0, 0.0)]
        observed_best = []
        for round_number, (iou, dice) in enumerate(history, 1):
            best.update(iou, dice, round_number)
            observed_best.append(best.iou)

        self.assertEqual(observed_best, sorted(observed_best))
        self.assertEqual((best.iou, best.dice, best.round), (0.31, 0.47, 2))

    def test_align_prediction_to_mask_smoke(self):
        align = load_alignment_helper()
        prediction = torch.rand(2, 1, 8, 8)
        mask = torch.zeros(2, 1, 16, 16)
        aligned = align(prediction, mask)

        self.assertEqual(aligned.shape, mask.shape)
        value = load_loss(LOSS_PATHS[0])()(aligned, mask)
        self.assertTrue(torch.isfinite(value))


if __name__ == "__main__":
    unittest.main()
