import random
from torchvision.datasets import CIFAR100
from torchvision import transforms
from PIL import Image
import torch

# Define data transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


class CIFAR100PairDataset(CIFAR100):
    def __init__(self, data_root, phase=True, num_pairs=70000):
        """
        phase=True for train, False for val/test
        num_pairs: number of unique (unordered) pairs to generate
        """
        super().__init__(root=data_root, train=phase, download=True, transform=None)
        self.num_pairs = num_pairs
        self.pair_list = self._generate_unique_pairs(len(self.data), num_pairs)
        self.transform = train_transform if phase else valid_transform

    def _generate_unique_pairs(self, dataset_size, num_pairs):
        seen = set()
        pairs = []

        while len(pairs) < num_pairs:
            i = random.randint(0, dataset_size - 1)
            j = random.randint(0, dataset_size - 1)
            if i == j:
                continue
            a, b = sorted((i, j))  # ensure (a, b) == (min(i,j), max(i,j))
            if (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b))

        return pairs

    def __getitem__(self, index):
        i, j = self.pair_list[index]

        img1, label1 = self.data[i], self.targets[i]
        img2, label2 = self.data[j], self.targets[j]

        # Convert from numpy array to PIL Image
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), (torch.tensor(label1), torch.tensor(label2))

    def __len__(self):
        return self.num_pairs