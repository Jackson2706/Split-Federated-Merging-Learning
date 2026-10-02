"""
Full pipeline model for end-to-end inference.

Composes client, edge, and cloud sub-models into a single nn.Module
for validation and testing.
"""

import torch
from torch import nn


class FullPipelineModel(nn.Module):
    """
    End-to-end model composing Client → Edge → Cloud sub-models.

    Used for validation/testing where the full forward pass
    needs to run through all three tiers.
    """

    def __init__(self, client_model, edge_model, cloud_model):
        super().__init__()
        self.client = client_model
        self.edge = edge_model
        self.cloud = cloud_model

    def forward(self, x):
        with torch.amp.autocast(device_type='cuda', enabled=(x.device.type == 'cuda')):
            x = self.client(x)
            x = self.edge(x)
            # Apply GAP + flatten to match the prototype extraction pipeline
            # (edge extraction does AdaptiveAvgPool2d + flatten before cloud)
            x = torch.flatten(nn.AdaptiveAvgPool2d((1, 1))(x), start_dim=1)
            x = self.cloud(x)
        return x
