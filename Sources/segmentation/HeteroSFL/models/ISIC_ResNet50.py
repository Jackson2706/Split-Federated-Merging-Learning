import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ISICClientModelResNet50(nn.Module):
    """Client-side encoder for ISIC segmentation with HeteroSFL.

    ResNet50 stem + layer1 + layer2 -> compression encoder (wide/narrow channels).
    """

    def __init__(self, args):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.client_part = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,  # Output: [B, 512, H/8, W/8]
        )
        self.compression_channels = args.get("client_out_channels", 512)
        self.bl_encoder = nn.Conv2d(512, self.compression_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.client_part(x)
        x = self.bl_encoder(x)
        return x


class ISICServerModelResNet50(nn.Module):
    """Server-side decoder for ISIC segmentation with HeteroSFL.

    Decoder maps received channels -> 512, then continues ResNet50 encoder
    layers 3-4, followed by transposed-conv decoder for dense prediction.
    """

    def __init__(self, args):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        input_channels = args.get("server_in_channels", 512)
        self.bl_decoder = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)

        self.encoder3 = resnet.layer3   # 512 -> 1024
        self.encoder4 = resnet.layer4   # 1024 -> 2048

        # Decoder (upsampling path)
        self.upconv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.bl_decoder(x)
        x3 = self.encoder3(x)
        x4 = self.encoder4(x3)

        d5 = self.upconv5(x4)
        d4 = self.upconv4(d5)
        d3 = self.upconv3(d4)
        d2 = self.upconv2(d3)
        out = self.final_conv(d2)
        out = torch.sigmoid(out)
        return out


class ISICMergedModelResNet50(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.client_side_model = ISICClientModelResNet50(args)
        self.server_side_model = ISICServerModelResNet50(args)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)

    def forward(self, x):
        out = self.client_side_model(x)
        return self.server_side_model(out)
