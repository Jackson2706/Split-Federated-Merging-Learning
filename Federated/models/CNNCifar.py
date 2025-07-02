import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the original classification head (fc layer)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # Exclude the final fc layer

        # Freeze feature extractor if desired
        if args.get("freeze_backbone", False):
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Add custom classifier
        self.classifier = nn.Linear(resnet.fc.in_features, args["num_classes"])

    def forward(self, x):
        features = self.feature_extractor(x)  # Output shape: (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=1)
