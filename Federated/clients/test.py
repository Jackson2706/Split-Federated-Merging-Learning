import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

def test_inference(args, model, test_dataset):
    """Returns the test F1 score (macro) and loss."""
    model.eval()
    loss = 0.0

    device = 'cuda' if args["gpu"] else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    model = model.to(device)

    all_preds = []
    all_labels = []
    torch.cuda.empty_cache()  # Clear GPU memory
    for batch_idx, (images, labels) in enumerate(testloader):
        torch.cuda.empty_cache()
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')  # 'macro', 'micro', or 'weighted'
    return f1, loss
