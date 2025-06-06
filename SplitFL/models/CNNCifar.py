import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F

class CifarClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(pretrained=True)
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

class CifarServerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        resnet = resnet50(pretrained=True)
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
