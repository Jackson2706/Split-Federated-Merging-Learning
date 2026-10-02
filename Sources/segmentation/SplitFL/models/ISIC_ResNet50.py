import torch.nn as nn
from torchvision import models


class ISICClientModelResNet50(nn.Module):
    """ResNet50 stem through layer2: [B, 3, 224, 224] -> [B, 512, 28, 28]."""

    def __init__(self, weights=models.ResNet50_Weights.DEFAULT):
        super().__init__()
        resnet = models.resnet50(weights=weights)
        self.client_part = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )

    def forward(self, x):
        return self.client_part(x)


class ISICServerModelResNet50(nn.Module):
    """Remaining encoder and decoder: [B, 512, 28, 28] -> [B, 1, 224, 224]."""

    def __init__(self, args=None, weights=models.ResNet50_Weights.DEFAULT):
        super().__init__()
        resnet = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(resnet.layer3, resnet.layer4)
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
        return self.decoder(self.encoder(x))


class ISICMergedModelResNet50(nn.Module):
    def __init__(self, args=None, weights=models.ResNet50_Weights.DEFAULT):
        super().__init__()
        self.client_side_model = ISICClientModelResNet50(weights=weights)
        self.server_side_model = ISICServerModelResNet50(args, weights=weights)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)

    def forward(self, x):
        return self.server_side_model(self.client_side_model(x))
