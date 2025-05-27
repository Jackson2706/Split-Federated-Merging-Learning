from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone(), torch.tensor(label)

class Client(ABC):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args["is_gpu"] else 'cpu'

    def train_val_test(self, dataset, idxs):
        # Split dataset
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        from torch.utils.data import DataLoader
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args["local_bs"], shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=max(int(len(idxs_val)/10), 1), shuffle=False)
        testloader  = DataLoader(DatasetSplit(dataset, idxs_test),  batch_size=max(int(len(idxs_test)/10), 1), shuffle=False)
        return trainloader, validloader, testloader

    @abstractmethod
    def update_weights(self, model, global_round):
        pass

    def inference(self, model):
        model.eval()
        criterion = nn.NLLLoss().to(self.device)
        loss, total, correct = 0.0, 0.0, 0.0
        for images, labels in self.testloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
        return correct / total, loss
