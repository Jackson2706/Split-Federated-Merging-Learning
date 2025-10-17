# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ------------------ Client Model ------------------
# class AlexnetClientModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.client_part = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 64x16x16
#         )

#     def forward(self, x):
#         return self.client_part(x)

# # ------------------ Edge Model ------------------
# class AlexnetEdgeModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.edge_part = nn.Sequential(
#             nn.Conv2d(64, 192, kernel_size=3, padding=1),
#             nn.BatchNorm2d(192),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 192x8x8

#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.BatchNorm2d(384),
#             nn.ReLU(),

#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),

#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # 256x4x4

#         )
#         self.fc = nn.Sequential(torch.nn.Linear(256*4*4, 1000),
#                                 torch.nn.ReLU(),
#                                 torch.nn.Dropout(0.1),
#                                 torch.nn.Linear(1000, 256))
#     def forward(self, x):
#         return self.edge_part(x)

#     def forward_contrastive(self, out):
#         out = out.view(out.size(0), -1)
#         return self.fc(out)
# # ------------------ Cloud Head (Task-Specific) ------------------
# class ALexnetCloudHead(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.LayerNorm(256 * 4 * 4),

#             nn.Dropout(0.3),
#             nn.Linear(256 * 4 * 4, 1024),
#             nn.ReLU(),

#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.ReLU(),

#             nn.Linear(512, args["num_classes"]),
#         )

#     def forward(self, x):
#         return self.head(x)

# # ------------------ Split Forward Function ------------------
# def split_forward(x, client, edge, cloud):
#     out_client = client(x)
#     out_edge = edge(out_client)
#     out_cloud = cloud(out_edge)
#     return out_cloud

# # ------------------ Demo Train Step ------------------
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_classes = 10

#     # Instantiate models
#     client = ClientModel().to(device)
#     edge = EdgeModel().to(device)
#     cloud = CloudHead(num_classes=num_classes).to(device)

#     # Optimizer and loss
#     optimizer = torch.optim.Adam(cloud.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()

#     # Dummy input (CIFAR-10 shape)
#     x = torch.randn(8, 3, 32, 32).to(device)  # batch_size=8
#     y = torch.randint(0, num_classes, (8,)).to(device)

#     # Forward pass
#     pred = split_forward(x, client, edge, cloud)

#     # Compute loss and backprop
#     loss = criterion(pred, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print("Logits:", pred)
#     print("Softmax:", torch.softmax(pred, dim=1))
#     print("True labels:", y)
#     print("Loss:", loss.item())


from torch import nn
import torch

class AlexnetClientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.client_part = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x16x16
        )

    def forward(self, x):
        return self.client_part(x)


class AlexnetEdgeModel(nn.Module):
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
        self.fc = nn.Sequential(
            torch.nn.Linear(16384, 1000),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1000, 256),
        )

    def forward(self, x):
        return self.edge_part(x)

    def forward_contrastive(self, out):
        out = out.view(out.size(0), -1)
        return self.fc(out)


class ALexnetCloudHead(nn.Module):
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
