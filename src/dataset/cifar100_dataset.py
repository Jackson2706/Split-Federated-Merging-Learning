import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.baseline_factory import get_baseline
from src.models.model_factory import get_model

from .base_dataset import BaseDataset


class CIFAR100Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self.num_clients = config["experiment"]["clients"]
        self.batch_size = config["training"]["batch_size"]
        self.scenario = config["experiment"]["scenario"]  # iid | non-iid
        self.noniid_type = config["experiment"].get("noniid_type", None)
        self.alpha = config["experiment"].get("alpha", 0.5)  # dirichlet concentration
        self.classes_per_client = config["experiment"].get("classes_per_client", None)

    # ✅ implements BaseDataset.load_data()
    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.CIFAR100(
            root="dataset/",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="dataset/",
            train=False,
            download=True,
            transform=transform
        )
        return train_dataset, test_dataset

    # ✅ implements BaseDataset.partition_data()
    def partition_data(self, dataset):
        num_samples = len(dataset)
        indices = np.arange(num_samples)

        if self.scenario == "iid":
            np.random.shuffle(indices)
            split_indices = np.array_split(indices, self.num_clients)
            return [Subset(dataset, idxs) for idxs in split_indices]

        elif self.scenario == "non-iid":
            targets = np.array(dataset.targets)
            num_classes = len(np.unique(targets))

            if self.noniid_type == "dirichlet":
                class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
                client_indices = [[] for _ in range(self.num_clients)]
                for c, idxs in enumerate(class_indices):
                    np.random.shuffle(idxs)
                    proportions = np.random.dirichlet(self.alpha * np.ones(self.num_clients))
                    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
                    split = np.split(idxs, proportions)
                    for cid, part in enumerate(split):
                        client_indices[cid].extend(part)
                return [Subset(dataset, idxs) for idxs in client_indices]

            elif self.noniid_type == "label-split":
                classes = np.arange(num_classes)
                np.random.shuffle(classes)

                # assign classes_per_client
                split_classes = np.array_split(classes, self.num_clients)
                client_indices = []
                for cid, cls in enumerate(split_classes):
                    cls = cls[: self.classes_per_client]
                    idxs = np.where(np.isin(targets, cls))[0]
                    client_indices.append(idxs)
                return [Subset(dataset, idxs) for idxs in client_indices]

        raise ValueError(f"Unknown scenario: {self.scenario}")

    # ✅ implements BaseDataset.prepare()
    def prepare(self):
        train_dataset, test_dataset = self.load_data()
        client_datasets = self.partition_data(train_dataset)


        # Get baseline client class
        baseline_cfg = get_baseline(self.config["experiment"]["baseline"])
        ClientClass = baseline_cfg["client"]

        # Create clients
        clients = []
        for cid, subset in enumerate(client_datasets):
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            clients.append(ClientClass(
                client_id=cid,
                data_loader=loader,
                model=None,
                config=self.config
            ))
        return clients, test_dataset