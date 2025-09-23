from abc import ABC, abstractmethod

import torch


class BaseClient(ABC):
    def __init__(self, client_id, data_loader, model=None, config=None):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def local_train(self, epochs, lr):
        """Perform local training and return update"""
        pass

    @abstractmethod
    def send_update(self):
        """Send local model update"""
        pass

    @abstractmethod
    def send_smashed_data(self, batch):
        """Send smashed data (SplitNN style)"""
        pass

    @abstractmethod
    def receive_gradients(self, grad):
        """Receive gradients from cloud/edge"""
        pass