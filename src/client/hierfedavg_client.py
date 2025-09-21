import copy
import time

import torch

from src.utils.logger import Logger

from .base_client import BaseClient


class HierFedAvgClient(BaseClient):
    def __init__(self, client_id, data_loader, model, config):
        super().__init__(client_id, data_loader, model, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger(log_dir=f"logs/HierFedAvg", name=f"client_{client_id}")

    def local_train(self, epochs=1, lr=0.01):
        """Perform local training on client dataset"""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.model.train()
        self.model.to(self.device)
        torch.cuda.empty_cache()
        for e in range(epochs):
            torch.cuda.empty_cache()
            epoch_start = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for x, y in self.data_loader:
                torch.cuda.empty_cache()
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            avg_loss = total_loss / total
            acc = 100.0 * correct / total

            self.logger.log(step=f"epoch_{e+1}", loss=avg_loss, acc=acc, step_start=epoch_start)

        return copy.deepcopy(self.model.state_dict())

    def send_update(self):
        """Send local weights to cloud"""
        return copy.deepcopy(self.model.state_dict())

    def send_smashed_data(self, batch):
        """Not used in FedAvg (SplitNN only)"""
        return None

    def receive_gradients(self, grad):
        """Not used in FedAvg (SplitNN only)"""
        return None
