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
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        if preds.shape != targets.shape:
            raise ValueError(
                f"[DiceFocalLoss] Shape mismatch: preds {preds.shape}, targets {targets.shape}"
            )

        # Invalid values cannot be made finite by clamp alone. Map them to the
        # nearest valid binary value, then keep probabilities away from log(0).
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
        preds = preds.clamp(min=self.eps, max=1.0 - self.eps)
        targets = targets.clamp(min=0.0, max=1.0)

        return preds, targets

    def dice_loss(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth + self.eps) / (
            union + self.smooth + self.eps
        )

        return 1 - dice

    def focal_loss(self, preds, targets):
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        if preds.numel() == 0:
            return preds.sum()

        preds = preds.clamp(min=self.eps, max=1.0 - self.eps)
        bce = F.binary_cross_entropy(preds, targets, reduction="none")
        pt = torch.exp(-bce)
        # ``alpha`` weights background; foreground receives the complementary
        # (larger, with the default) weight to counter lesion sparsity.
        alpha_t = targets * (1.0 - self.alpha) + (1.0 - targets) * self.alpha
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()

    def forward(self, preds, targets):
        # Keep reductions and BCE in float32 even when the caller uses autocast.
        with torch.autocast(device_type=preds.device.type, enabled=False):
            preds = preds.float()
            targets = targets.float()
            preds, targets = self._sanity_check(preds, targets)

            loss_dice = self.dice_loss(preds, targets)
            loss_focal = self.focal_loss(preds, targets)
            total_loss = loss_dice + loss_focal

        if self.debug:
            print(
                f"Dice Loss: {loss_dice.item():.6f}, Focal Loss: {loss_focal.item():.6f}, Total: {total_loss.item():.6f}"
            )

        return total_loss
