import torch
import torch.nn as nn
from torchvision import models


def set_requires_grad(module, freeze=False):
    for param in module.parameters():
        param.requires_grad = not freeze


class ClientModel(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(ClientModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify for CIFAR input (32x32)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()

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
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.part = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

        # ResNet-18 final feature dim is 512 (not 2048 like ResNet-50)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        return self.part(x)

    def forward_contrastive(self, out):
        out = out.view(out.size(0), -1)  # Flatten
        return self.fc(out)


class CloudModel(nn.Module):
    def __init__(self, args):
        super(CloudModel, self).__init__()
        # Input dim adjusted to 512 from ResNet-18
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, args["num_classes"]),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# === Optional: test forward and gradient flow ===
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = ClientModel().to(device)
    edge = EdgeModel().to(device)
    cloud = CloudModel({"num_classes": 100}).to(device)

    x = torch.randn(4, 3, 32, 32).to(device)  # CIFAR-100 input
    x = client(x)
    x = edge(x)
    output = cloud(x)

    y = torch.randint(0, 100, (4,)).to(device)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()

    # === Print gradient norms for debugging ===
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
