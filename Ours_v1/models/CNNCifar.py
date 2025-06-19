import torch
import torch.nn as nn
from torchvision import models


def set_requires_grad(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze


class ClientModel(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(ClientModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.part = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

    def forward(self, x):
        return self.part(x)


class EdgeModel(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(EdgeModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.part = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

    def forward(self, x):
        return self.part(x)


class CloudModel(nn.Module):
    def __init__(self, args, freeze_backbone=False):
        super(CloudModel, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.feature_part = nn.Sequential(
            resnet.layer4,
            resnet.avgpool
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, args["num_classes"])
        )
        set_requires_grad(self.feature_part, freeze=freeze_backbone)

    def forward(self, x):
        x = self.feature_part(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits



# Example usage
if __name__ == "__main__":
    client = ClientModel()
    edge = EdgeModel()
    cloud = CloudModel(num_classes=10, freeze_backbone=False)

    x = torch.randn(4, 3, 224, 224)  # Example batch of 4 images
    x = client(x)
    x = edge(x)
    output = cloud(x)

    print(output.shape)  # Should print: torch.Size([4, 10])
