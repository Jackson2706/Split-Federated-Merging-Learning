from torch import nn

class FullPipelineModel(nn.Module):
    """Mô hình đầy đủ (Client + Edge + Cloud) để test."""
    def __init__(self, client_model, edge_model, cloud_model):
        super().__init__()
        self.client = client_model
        self.edge = edge_model
        self.cloud = cloud_model

    def forward(self, x):
        x = self.client(x)
        x = self.edge(x)
        x = self.cloud(x)
        return x