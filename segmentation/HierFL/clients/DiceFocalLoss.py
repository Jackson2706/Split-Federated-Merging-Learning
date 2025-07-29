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
        self.eps = eps  # for clamping
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

        # Clamp predictions to avoid NaN/Inf in log and BCE
        preds = torch.clamp(preds, min=self.eps, max=1 - self.eps)

        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print("❌ NaN or Inf found in predictions")
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("❌ NaN or Inf found in targets")

        return preds, targets

    def dice_loss(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        if torch.isnan(dice) or torch.isinf(dice):
            print("❌ NaN/Inf in Dice loss computation")

        return 1 - dice

    def focal_loss(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

    def forward(self, preds, targets):
        preds, targets = self._sanity_check(preds, targets)

        loss_dice = self.dice_loss(preds, targets)
        loss_focal = self.focal_loss(preds, targets)
        total_loss = loss_dice + loss_focal

        if self.debug:
            print(
                f"Dice Loss: {loss_dice.item():.6f}, Focal Loss: {loss_focal.item():.6f}, Total: {total_loss.item():.6f}"
            )

        return total_loss
