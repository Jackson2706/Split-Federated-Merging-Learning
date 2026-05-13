import numpy as np
import torch
from torch.utils.data import DataLoader

from .DiceFocalLoss import DiceFocalLoss


def compute_iou_and_dice(preds, labels):
    preds = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    preds_binary = (preds > 0.5).astype(np.int32)

    intersection = np.sum((preds_binary == 1) & (labels == 1))
    union = np.sum((preds_binary == 1) | (labels == 1))
    iou = intersection / union if union != 0 else 0

    denom = np.sum(preds_binary == 1) + np.sum(labels == 1)
    dice = 2 * intersection / denom if denom != 0 else 0

    return iou, dice


def test_inference(args, model, test_dataset):
    """Returns test IoU, Dice, and loss."""
    model.eval()
    device = "cuda" if args["is_gpu"] else "cpu"
    criterion = DiceFocalLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = model.to(device)

    total_loss, test_iou, test_dice, total_samples = 0.0, 0.0, 0.0, 0
    torch.cuda.empty_cache()

    with torch.no_grad():
        for inputs, masks in testloader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            size = inputs.size(0)

            total_loss += criterion(outputs, masks).item()
            outputs = (outputs > 0.5).float()
            iou, dice = compute_iou_and_dice(outputs, masks)
            test_iou += iou * size
            test_dice += dice * size
            total_samples += size

    test_iou /= total_samples
    test_dice /= total_samples
    return test_iou, test_dice, total_loss / len(testloader)
