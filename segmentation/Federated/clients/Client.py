from abc import ABC, abstractmethod

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from .DiceFocalLoss import DiceFocalLoss
import numpy as np


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


class DatasetSplit(Dataset):
    """A dataset wrapper for splitting data by index."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # Avoid warning: use clone for tensors, otherwise convert
        image = image.clone()
        label = (
            label.clone()
            if isinstance(label, torch.Tensor)
            else torch.tensor(label)
        )
        return image, label


class Client(ABC):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = (
            self.train_val_test(dataset, list(idxs))
        )
        self.device = "cuda" if args["is_gpu"] else "cpu"

    def train_val_test(self, dataset, idxs):
        # Split dataset
        idxs_train = idxs[: int(0.8 * len(idxs))]
        idxs_val = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)) :]
        from torch.utils.data import DataLoader

        trainloader = DataLoader(
            DatasetSplit(dataset, idxs_train),
            batch_size=self.args["local_bs"],
            shuffle=True,
        )
        validloader = DataLoader(
            DatasetSplit(dataset, idxs_val),
            batch_size=max(int(len(idxs_val) / 10), 1),
            shuffle=False,
        )
        testloader = DataLoader(
            DatasetSplit(dataset, idxs_test),
            batch_size=max(int(len(idxs_test) / 10), 1),
            shuffle=False,
        )
        return trainloader, validloader, testloader

    @abstractmethod
    def update_weights(self, model, global_round):
        pass

    def inference(self, model):
        model.eval()
        model = model.to(self.device)
        criterion = DiceFocalLoss().to(self.device)
        loss = 0.0
        test_iou = 0.0
        test_dice = 0.0
        total_samples = 0

        for inputs, masks in self.testloader:
            inputs, masks = inputs.to(self.device), masks.to(self.device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, masks)
            loss += batch_loss.item()
            outputs = torch.sigmoid(
                outputs
            )  # Apply sigmoid if the output is logits
            outputs = (outputs > 0.5).float()
            iou, dice = compute_iou_and_dice(outputs, masks)
            test_iou += iou * inputs.size(0)  # Multiply by batch size
            test_dice += dice * inputs.size(0)
            total_samples += inputs.size(0)
        test_iou /= total_samples
        test_dice /= total_samples
        return test_iou, test_dice, loss / len(self.testloader)
