from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import torch
class BaseCloud(ABC):
    def __init__(self, global_model, clients, edges=None, config=None, test_dataset=None):
        self.global_model = global_model
        self.clients = clients
        self.edges = edges or 0
        self.config = config
        self.test_dataset=test_dataset

    @abstractmethod
    def run(self, rounds, **kwargs):
        """Run the FL simulation for a number of rounds"""
        pass

    @abstractmethod
    def aggregate(self, updates):
        """Aggregate client/edge updates into new global model"""
        pass

    @abstractmethod
    def distribute(self):
        """Send updated global model to clients/edges"""
        pass

    @abstractmethod
    def first_distribute(self):
        """Send global model to clients at the first round"""
        pass
        
    def evaluate(self, batch_size=128):
        """Evaluate global model on test dataset"""
        testloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss()
        self.global_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.global_model(x)
                loss = criterion(out, y)

                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / total
        acc = 100.0 * correct / total
        return avg_loss, acc
