import torch.nn as nn
from torchvision import models


def set_requires_grad(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze


class ISICClientModelResNet50(nn.Module):
    """ResNet50 stem through layer1: [B, 3, 224, 224] -> [B, 256, 56, 56]."""

    def __init__(self, freeze_backbone=False, weights=models.ResNet50_Weights.DEFAULT):
        super().__init__()
        resnet = models.resnet50(weights=weights)
        self.part = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

    def forward(self, x):
        return self.part(x)


class ISICEdgeModelResNet50(nn.Module):
    """ResNet50 layer2/layer3: [B, 256, 56, 56] -> [B, 1024, 14, 14]."""

    def __init__(self, freeze_backbone=False, weights=models.ResNet50_Weights.DEFAULT):
        super().__init__()
        resnet = models.resnet50(weights=weights)
        self.part = nn.Sequential(resnet.layer2, resnet.layer3)
        set_requires_grad(self.part, freeze=freeze_backbone)

    def forward(self, x):
        return self.part(x)


class ISICCloudModelResNet50(nn.Module):
    """ResNet50 layer4 plus 7x7-to-224x224 segmentation decoder."""

    def __init__(
        self, args=None, freeze_backbone=False, weights=models.ResNet50_Weights.DEFAULT
    ):
        super().__init__()
        resnet = models.resnet50(weights=weights)
        self.feature_part = resnet.layer4
        set_requires_grad(self.feature_part, freeze=freeze_backbone)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.feature_part(x))
