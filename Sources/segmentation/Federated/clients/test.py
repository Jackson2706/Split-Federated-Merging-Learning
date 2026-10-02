import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from .DiceFocalLoss import DiceFocalLoss
from segmentation.training_metrics import compute_iou_and_dice


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
