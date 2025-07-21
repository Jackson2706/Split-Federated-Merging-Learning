import torch
import torch.nn as nn
from torchvision import models


class ResNet18_CIFAR100(nn.Module):
    def __init__(self, args):
        super(ResNet18_CIFAR100, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapt for small CIFAR-100 images (32x32)
        self.model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.model.maxpool = nn.Identity()  # Remove 3x3 maxpool
        self.model.fc = nn.Linear(
            self.model.fc.in_features, args["num_classes"]
        )

    def forward(self, x):
        return self.model(x)
