import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


def get_cifar(data_path: str = "/mnt/Data/cifar/"):
    tr = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(data_path, train=True, download=False, transform=tr)
    testset = CIFAR10(data_path, train=False, download=False, transform=tr)
    return trainset, testset


def prepare_dataset(
    num_partitions: int, batch_size: int, val_ratio: float = 0.1
):
    trainset, testset = get_cifar()

    # Split train dataset into "num_partitions" trainsets
    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(8386)
    )

    # Create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(8386)
        )
        trainloaders.append(
            DataLoader(
                for_train, batch_size=batch_size, shuffle=True, num_workers=2
            )
        )

        valloaders.append(
            DataLoader(
                for_val, batch_size=batch_size, shuffle=False, num_workers=2
            )
        )
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader