import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HAM10000ClientModelResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.client_part = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,  # CUT here
        )
        
    def forward(self, x):
        return self.client_part(x)

class HAM10000ServerModelResNet50(nn.Module):
    def __init__(self, args):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.server_part = nn.Sequential(
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.fc = nn.Linear(resnet.fc.in_features, args["num_classes"])
        
    def forward(self, x):
        x = self.server_part(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=1)


class HAM10000MergedModelResNet50(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.client_side_model = HAM10000ClientModelResNet50()
        self.server_side_model = HAM10000ServerModelResNet50(args)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)
    
    def forward(self, x):
        out = self.client_side_model(x)
        return self.server_side_model(out)