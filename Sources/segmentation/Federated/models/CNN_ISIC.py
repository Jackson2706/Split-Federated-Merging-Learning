import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ISIC_UNET_2D(nn.Module):
    def __init__(self, args=None):
        super(ISIC_UNET_2D, self).__init__()

        # Load pretrained ResNet-50
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.encoder1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
        )
        self.encoder2 = self.encoder.layer1
        self.encoder3 = self.encoder.layer2
        self.encoder4 = self.encoder.layer3
        self.encoder5 = self.encoder.layer4

        # Decoder (upsampling path)
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)

        # Final conv layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Downsample (encode)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # Upsample (decode) with skip connections
        d5 = self.upconv5(x5)
        # d5 = F.interpolate(d5, size=(x4.size(2), x4.size(3)), mode='bilinear', align_corners=False) + x4

        d4 = self.upconv4(d5)
        # d4 = F.interpolate(d4, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False) + x3

        d3 = self.upconv3(d4)
        # d3 = F.interpolate(d3, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False) + x2

        d2 = self.upconv2(d3)
        # d2 = F.interpolate(d2, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False) + x1

        # Final layer
        out = self.final_conv(d2)
        out = F.interpolate(
            out,
            size=(x.size(2), x.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        out = torch.sigmoid(out)  # Apply sigmoid for binary segmentation
        return out
