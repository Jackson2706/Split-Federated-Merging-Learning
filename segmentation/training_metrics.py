from dataclasses import dataclass


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
