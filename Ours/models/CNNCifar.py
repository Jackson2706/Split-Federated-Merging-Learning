import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.part = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        )

    def forward(self, x):
        return self.part(x)


class EdgeModel(nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.part = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
        )

    def forward(self, x):
        return self.part(x)


class CloudModel(nn.Module):
    def __init__(self, args):
        super(CloudModel, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.feature_part = nn.Sequential(
            resnet.layer4,
            resnet.avgpool
        )
        self.classifier = nn.Linear(resnet.fc.in_features, args["num_classes"])


    def forward(self, x):
        x = self.feature_part(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)


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
