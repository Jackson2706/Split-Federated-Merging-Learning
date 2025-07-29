from torch import nn


class AlexNetClient_Ours(nn.Module):
    def __init__(self):
        super().__init__()
        self.client_part = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x16x16
        )

    def forward(self, x):
        return self.client_part(x)

class AlexNetEdge_Ours(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_part = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 192x8x8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.edge_part(x)

class AlexNetCloud_Ours(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_classes = args["num_classes"]
        self.cloud_part = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256x4x4
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.cloud_part(x)
