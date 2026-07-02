import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceFocalLoss(nn.Module):
    def __init__(
        self, alpha=0.25, gamma=2.0, smooth=1.0, eps=1e-7, debug=False
    ):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps
        self.debug = debug

    def _sanity_check(self, preds, targets):
        if targets.dtype != torch.float32:
            targets = targets.float()
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if preds.shape != targets.shape:
            raise ValueError(
                f"[DiceFocalLoss] Shape mismatch: preds {preds.shape}, targets {targets.shape}"
            )
        preds = torch.clamp(preds, min=self.eps, max=1 - self.eps)
        return preds, targets

    def dice_loss(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

    def focal_loss(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

    def forward(self, preds, targets):
        # F.binary_cross_entropy (used by focal_loss) is unsafe under autocast,
        # and this loss operates on probabilities. Force a float32, autocast-off
        # region so it is safe whether or not the caller is inside autocast.
        with torch.autocast(device_type=preds.device.type, enabled=False):
            preds = preds.float()
            targets = targets.float()
            preds, targets = self._sanity_check(preds, targets)
            loss_dice = self.dice_loss(preds, targets)
            loss_focal = self.focal_loss(preds, targets)
            return loss_dice + loss_focal
