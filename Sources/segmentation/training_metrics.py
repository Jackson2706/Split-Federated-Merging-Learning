from dataclasses import dataclass

import numpy as np


def compute_iou_and_dice(preds, labels):
    """Compute the frozen journal binary-segmentation IoU/Dice protocol.

    Predictions are binarized with the strict threshold ``preds > 0.5``.
    Predictions and labels are flattened across the whole batch and scored as
    one global pixel-level aggregate, not as a mean of per-image metrics. An
    empty union produces IoU 0, and an empty Dice denominator produces Dice 0;
    empty samples are neither treated as perfect nor excluded.
    """
    preds = preds.detach().cpu().numpy().flatten()
    labels = labels.detach().cpu().numpy().flatten()
    preds_binary = (preds > 0.5).astype(np.int32)

    intersection = np.sum((preds_binary == 1) & (labels == 1))
    union = np.sum((preds_binary == 1) | (labels == 1))
    iou = intersection / union if union != 0 else 0

    denominator = np.sum(preds_binary == 1) + np.sum(labels == 1)
    dice = 2 * intersection / denominator if denominator != 0 else 0
    return iou, dice


@dataclass
class BestSegmentationMetrics:
    """Best validation metrics, with Dice tied to the best-IoU round."""

    iou: float = float("-inf")
    dice: float = float("-inf")
    round: int = 0

    def update(self, iou: float, dice: float, round_number: int) -> bool:
        if iou <= self.iou:
            return False
        self.iou = float(iou)
        self.dice = float(dice)
        self.round = int(round_number)
        return True
