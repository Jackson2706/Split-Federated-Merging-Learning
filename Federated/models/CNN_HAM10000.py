import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNNHAM10000(nn.Module):
    def __init__(self, args):
        super(CNNHAM10000, self).__init__()

        # Load pretrained ResNet-152 from torchvision
        self.resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

        # Replace the classifier head with custom number of classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, args["num_classes"])

    def forward(self, x):
        logits = self.resnet(x)  # Output shape: (B, num_classes)
        return F.log_softmax(logits, dim=1)  # For NLLLoss
