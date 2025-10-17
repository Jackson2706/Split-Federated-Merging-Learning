from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load_data(self):
        """Load raw dataset"""
        pass

    @abstractmethod
    def partition_data(self, dataset):
        """Partition dataset (IID / Non-IID)"""
        pass

    @abstractmethod
    def prepare(self):
        """
        Return (global_model, clients, edges) 
        - global_model: initialized model
        - clients: list of client objects
        - edges: list of edge objects (may be empty if 2-tier)
        """
        pass