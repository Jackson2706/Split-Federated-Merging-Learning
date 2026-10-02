"""
Model evaluation utilities.

Provides functions for evaluating trained models on test datasets,
computing metrics like macro-F1 score.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def evaluate_model(model, test_dataset, device, batch_size=1):
    """
    Evaluate a model on a test dataset and compute macro-F1 score.

    Args:
        model: Trained nn.Module (e.g., FullPipelineModel).
        test_dataset: Test dataset.
        device: torch.device for inference.
        batch_size: Batch size for the test DataLoader.

    Returns:
        Dict with 'f1_macro', 'predictions', and 'targets'.
    """
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average="macro")

    return {
        "f1_macro": f1,
        "predictions": all_preds,
        "targets": all_targets,
    }
