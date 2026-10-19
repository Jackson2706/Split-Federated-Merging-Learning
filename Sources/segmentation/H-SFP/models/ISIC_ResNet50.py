import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def set_requires_grad(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze


class ISICClientModel(nn.Module):
    """Client tier: ResNet50 stem + layer1 (encoder1 + encoder2).

    Output: [B, 256, 56, 56] for 224x224 input.
    """

    def __init__(self, freeze_backbone=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.part = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,  # -> [B, 64, 56, 56]
            resnet.layer1,  # -> [B, 256, 56, 56]
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

    def forward(self, x):
        return self.part(x)


class ISICEdgeModel(nn.Module):
    """Edge tier: ResNet50 layer2 + layer3 + layer4.

    Input:  [B, 256, 56, 56]
    Output: [B, 2048, 7, 7] for 224x224 input.
    """

    def __init__(self, freeze_backbone=False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.part = nn.Sequential(
            resnet.layer2,  # -> [B, 512, 28, 28]
            resnet.layer3,  # -> [B, 1024, 14, 14]
            resnet.layer4,  # -> [B, 2048, 7, 7]
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

    def forward(self, x):
        return self.part(x)


class ISICCloudModel(nn.Module):
    """Cloud tier: Decoder for segmentation.

    Takes pooled 2D features from edge prototypes and produces segmentation map.
    Input for supervised training: [B, D] (2D feature vectors from prototypes).
    For full-pipeline inference, use ISICFullPipelineModel instead.

    Since prototypes are 2D (after GAP at edge), the cloud trains a classifier
    on (fg=1, bg=0) class prototypes. During inference, the full pipeline
    reconstructs spatial predictions via the decoder path.
    """

    def __init__(self, args=None):
        super().__init__()
        # Cloud receives 2D feature vectors (after GAP from edge output)
        # Edge output is [B, 2048] after GAP
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),  # 2 classes: background, foreground
        )

    def forward(self, x):
        return self.classifier(x)


class ISICCloudDecoder(nn.Module):
    """Full decoder for inference: takes encoder bottleneck [B, 2048, 7, 7]
    and produces segmentation mask [B, 1, 224, 224].
    """

    def __init__(self):
        super().__init__()
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, 2048, 7, 7]
        d5 = self.upconv5(x)   # [B, 1024, 14, 14]
        d4 = self.upconv4(d5)  # [B, 512, 28, 28]
        d3 = self.upconv3(d4)  # [B, 256, 56, 56]
        d2 = self.upconv2(d3)  # [B, 64, 112, 112]
        d1 = self.upconv1(d2)  # [B, 32, 224, 224]
        out = self.final_conv(d1)  # [B, 1, 224, 224]
        out = torch.sigmoid(out)
        return out
