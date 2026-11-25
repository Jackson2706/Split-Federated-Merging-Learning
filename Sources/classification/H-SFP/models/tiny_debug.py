"""Tiny split model for the CPU debug configuration (configs/debug.yaml)."""
from torch import nn


class TinyClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        return self.net(x)


class TinyEdge(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        return self.net(x)


class TinyCloud(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(16, int(config["num_classes"]))

    def forward(self, x):
        return self.fc(x.flatten(1))
