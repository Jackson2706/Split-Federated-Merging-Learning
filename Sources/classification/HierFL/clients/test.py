import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader


def test_inference(args, model, test_dataset):
    """ Returns the test F1 score and loss.
    """

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    device = 'cuda' if args["is_gpu"] else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model = model.to(device)

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Predictions
        _, pred_labels = torch.max(outputs, 1)
        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Primary metric: top-1 accuracy
    acc = accuracy_score(all_labels, all_preds)
    return acc, total_loss
