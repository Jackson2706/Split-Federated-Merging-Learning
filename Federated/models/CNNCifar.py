# import torch.nn.functional as F
# from torch import nn


# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args["num_classes"])

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet18(pretrained=True)

        # Remove the original classification head (fc layer)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # Exclude the final fc layer
        
        # Add custom classifier
        self.classifier = nn.Linear(resnet.fc.in_features, args["num_classes"])

    def forward(self, x):
        # Input shape: (B, 3, 224, 224)
        features = self.feature_extractor(x)  # Output shape: (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (B, 2048)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=1)
