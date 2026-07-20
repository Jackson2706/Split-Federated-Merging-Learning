import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import numpy as np
from .DiceFocalLoss import DiceFocalLoss


def compute_iou_and_dice(preds, labels):
    # Convert tensors to numpy arrays
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # Flatten arrays
    preds = preds.flatten()
    labels = labels.flatten()

    # Convert to binary predictions (if needed)
    preds_binary = (preds > 0.5).astype(np.int32)

    # Compute Intersection and Union for IoU
    intersection = np.sum((preds_binary == 1) & (labels == 1))
    union = np.sum((preds_binary == 1) | (labels == 1))
    iou = intersection / union if union != 0 else 0

    # Compute Dice Coefficient
    dice = (
        2 * intersection / (np.sum(preds_binary == 1) + np.sum(labels == 1))
        if (np.sum(preds_binary == 1) + np.sum(labels == 1)) != 0
        else 0
    )

    return iou, dice


def test_inference(args, model, test_dataset):
    """Returns the test F1 score (macro) and loss."""
    model.eval()
    loss = 0.0

    device = "cuda" if args["gpu"] else "cpu"
    criterion = DiceFocalLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = model.to(device)
    test_iou = 0.0
    test_dice = 0.0
    total_samples = 0
    torch.cuda.empty_cache()  # Clear GPU memory
    for batch_idx, (inputs, masks) in enumerate(testloader):
        torch.cuda.empty_cache()
        inputs, masks = inputs.to(device), masks.to(device)

        outputs = model(inputs)
        size = inputs.size(0)
        del inputs

        batch_loss = criterion(outputs, masks)
        loss += batch_loss.item()
        del batch_loss
        outputs = (outputs > 0.5).float()
        iou, dice = compute_iou_and_dice(outputs, masks)
        del outputs, masks
        test_iou += iou * size  # Multiply by batch size
        test_dice += dice * size
        total_samples += (
            size  # Update total_samples to use size instead of inputs.size(0)
        )
    test_iou /= total_samples
    test_dice /= total_samples
    return test_iou, test_dice, loss / len(testloader)
