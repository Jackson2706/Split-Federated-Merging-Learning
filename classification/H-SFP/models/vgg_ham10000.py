from torch import nn
from ehsfp.prototype_space import CosineClassifier, validate_prototype_space


class VGGClient_Ours(nn.Module):
    def __init__(self):
        super().__init__()
        self.client_part = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
        )

    def forward(self, x):
        return self.client_part(x)

class VGGEedge_Ours(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_part = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
        )

    def forward(self, x):
        return self.edge_part(x)

class VGGCloud_Ours(nn.Module):
    """
    Cloud head for VGG split architecture.

    Receives 2D flattened features (256-dim) from the edge's prototype
    extraction phase (AdaptiveAvgPool2d + flatten) and classifies them.
    """

    def __init__(self, args):
        super().__init__()
        self.prototype_space = validate_prototype_space(args.get("prototype_space"))
        if self.prototype_space == "centered_cosine":
            self.cosine_head = CosineClassifier(256, args["num_classes"])
            return
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, args["num_classes"]),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.prototype_space == "centered_cosine":
            return self.cosine_head(x)
        return self.classifier(x)
