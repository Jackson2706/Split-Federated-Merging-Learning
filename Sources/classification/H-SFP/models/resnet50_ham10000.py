import torch.nn as nn
from torchvision import models
from ehsfp.prototype_space import CosineClassifier, validate_prototype_space


def set_requires_grad(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze


class HAM10000ClientModelResNet50(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(HAM10000ClientModelResNet50, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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


class HAM10000EdgeModelResNet50(nn.Module):
    def __init__(self, freeze_backbone=False):
        super(HAM10000EdgeModelResNet50, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.part = nn.Sequential(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4, 
            resnet.avgpool
        )
        set_requires_grad(self.part, freeze=freeze_backbone)

    def forward(self, x):
        return self.part(x)


class HAM10000CloudModelResNet50(nn.Module):
    def __init__(self, args, freeze_backbone=False):
        super(HAM10000CloudModelResNet50, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.prototype_space = validate_prototype_space(args.get("prototype_space"))
        if self.prototype_space == "centered_cosine":
            self.cosine_head = CosineClassifier(resnet.fc.in_features, args["num_classes"])
            return

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet.fc.in_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, args["num_classes"]),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.prototype_space == "centered_cosine":
            return self.cosine_head(x)
        logits = self.classifier(x)
        return logits
