import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HAM10000ClientModelResNet50(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Load Pretrained Weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Standard ResNet Stem & Layers 1-2
        self.client_part = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,  # Output shape: [Batch, 512, H/8, W/8]
        )
        
        # --- HeteroSFL Encoder ---
        # Compresses 512 channels -> client_out_channels (e.g., Wide=76, Narrow=4)
        self.compression_channels = args.get("client_out_channels", 512)
        self.bl_encoder = nn.Conv2d(512, self.compression_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.client_part(x)
        x = self.bl_encoder(x)
        return x


class HAM10000ServerModelResNet50(nn.Module):
    def __init__(self, args):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # --- HeteroSFL Decoder ---
        # Maps received channels -> 512 (Expected input for ResNet50 Layer 3)
        input_channels = args.get("server_in_channels", 512)
        self.bl_decoder = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)

        self.server_part = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, args["num_classes"])

    def forward(self, x):
        # 1. Decode back to 512 channels
        x = self.bl_decoder(x)
        
        # 2. Standard Server Processing
        x = self.server_part(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=1)


class HAM10000MergedModelResNet50(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.client_side_model = HAM10000ClientModelResNet50(args)
        self.server_side_model = HAM10000ServerModelResNet50(args)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)

    def forward(self, x):
        out = self.client_side_model(x)
        return self.server_side_model(out)