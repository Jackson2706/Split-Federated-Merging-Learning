
import torch
import torch.nn as nn
from torchvision import models


def set_requires_grad(module, freeze=False):
    for param in module.parameters():
        param.requires_grad = not freeze


class ClientModel(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(ClientModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

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
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        self.part = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        set_requires_grad(self.part, freeze=freeze_backbone)
        self.fc = nn.Sequential(torch.nn.Linear(2048, 1000),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(1000, 256))

    def forward(self, x):
        return self.part(x)
    
    def forward_contrastive(self, out):
        out = out.view(out.size(0), -1)
        return self.fc(out)


class CloudModel(nn.Module):
    def __init__(self, args, freeze_backbone=False):
        super(CloudModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, args["num_classes"]),
        )
        # set_requires_grad(self.feature_part, freeze=freeze_backbone)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Optional: test forward and gradient flow
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = ClientModel().to(device)
    edge = EdgeModel().to(device)
    cloud = CloudModel(num_classes=10).to(device)

    x = torch.randn(4, 3, 224, 224).to(device)
    x = client(x)
    x = edge(x)
    output = cloud(x)

    y = torch.randint(0, 10, (4,)).to(device)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()

    # Check gradient norms
    def print_gradients(model, name):
        print(f"\nGradient norms in {name}:")
        for pname, param in model.named_parameters():
            if "bn" in pname.lower():
                continue
            if param.grad is not None:
                print(f"{pname}: grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"{pname}: ❌ No gradient")

    print_gradients(client, "ClientModel")
    print_gradients(edge, "EdgeModel")
    print_gradients(cloud, "CloudModel")

if __name__ == "__main__":
    test()
