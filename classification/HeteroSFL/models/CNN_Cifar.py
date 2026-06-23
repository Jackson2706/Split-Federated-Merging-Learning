import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CifarClientModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Standard ResNet18 initialization
        resnet = models.resnet18(weights=None)
        
        # Modify for CIFAR input (32x32)
        # Keeps spatial dimensions larger for small images
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1, # Output: 64 channels
            resnet.layer2, # Output: 128 channels
        )
        
        # --- HeteroSFL Encoder ---
        # Maps 128 (ResNet layer2 output) -> client_out_channels (Wide=76 / Narrow=4)
        self.compression_channels = args.get("client_out_channels", 128) 
        
        # We use a 3x3 Conv for the encoder to allow spatial mixing before transmission
        self.bl_encoder = nn.Conv2d(128, self.compression_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.features(x)
        x = self.bl_encoder(x)
        return x


class CifarServerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        # --- HeteroSFL Decoder ---
        # Maps received channels -> 128 (What ResNet layer3 expects)
        input_channels = args.get("server_in_channels", 128) # Default 128 if standard split
        
        # Standard ResNet18 flow:
        # layer1: 64ch -> layer2: 128ch -> [SPLIT] -> layer3: 256ch (but expects 128 input)
        self.bl_decoder = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        
        self.server_part = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, args["num_classes"])

    def forward(self, x):
        # 1. Decode / Restore channels to 128
        x = self.bl_decoder(x)
        
        # 2. Standard ResNet processing
        x = self.server_part(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        # Return raw logits: the HeteroSFL runner applies CrossEntropyLoss / KD
        # divergence, which expect logits (not log-probabilities).
        return logits


class MergedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Ensure we pass args so client/server know the channel configuration
        self.client_side_model = CifarClientModel(args)
        self.server_side_model = CifarServerModel(args)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)

    def forward(self, x):
        out = self.client_side_model(x)
        return self.server_side_model(out)