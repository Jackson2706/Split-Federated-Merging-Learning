from torch import nn
import torch.nn.functional as F


class AlexNetClient_SplitFed(nn.Module):
    def __init__(self):
        super().__init__()
        self.client_part = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 64x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x16x16
        )

    def forward(self, x):
        return self.client_part(x)


class AlexNetServer_SplitFed(nn.Module):
    def __init__(self, args):
        num_classes = args["num_classes"]
        super().__init__()
        self.server_part = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 192x8x8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
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
        x = self.server_part(x)
        return F.log_softmax(x, dim=1)

class AlexNetMergedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.client_side_model = AlexNetClient_SplitFed()
        self.server_side_model = AlexNetServer_SplitFed(args)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)
    
    def forward(self, x):
        out = self.client_side_model(x)
        return self.server_side_model(out)