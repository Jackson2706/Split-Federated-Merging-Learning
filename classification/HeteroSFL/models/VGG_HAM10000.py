import torch.nn.functional as F
from torch import nn


class VGGClient_SplitFed(nn.Module):
    def __init__(self):
        super().__init__()
        self.client_part = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.client_part(x)

class VGGServer_SplitFed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.server_part = nn.Sequential(
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, args["num_classes"])
)

    def forward(self, x):
        x = self.server_part(x)
        # Return raw logits: the HeteroSFL runner applies CrossEntropyLoss / KD
        # divergence, which expect logits (not log-probabilities).
        return x

class HAM10000MergedModelVGG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.client_side_model = VGGClient_SplitFed()
        self.server_side_model = VGGServer_SplitFed(args)

    def load_weight(self, client_model_weight, server_model_weight):
        self.client_side_model.load_state_dict(client_model_weight)
        self.server_side_model.load_state_dict(server_model_weight)
    
    def forward(self, x):
        out = self.client_side_model(x)
        return self.server_side_model(out)