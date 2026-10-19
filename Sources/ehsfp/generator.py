"""
Residual Prototype Generator for E-HSFP.

Adds a learnable residual to the standard Gaussian prototype synthesis:
    z_syn = mu + sigma * epsilon + G_theta(mu, sigma, epsilon)

This improves synthetic feature diversity while preserving the same
communication payload (only mu, sigma sent from clients).
"""

import torch
import torch.nn as nn


class ResidualPrototypeGenerator(nn.Module):
    """Lightweight residual generator for synthetic prototype features.

    z_syn = mu + sigma * epsilon + scale * G_theta(cat(mu, sigma, epsilon))
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mu: [N, D] or [N, *shape] class means (broadcast from [C, *shape]).
            sigma: [N, D] or [N, *shape] class stds.
            epsilon: [N, D] or [N, *shape] random noise.

        Returns:
            [N, D] synthetic features with learned residual.
        """
        flat_dim = mu.shape[1:]
        mu_flat = mu.reshape(mu.shape[0], -1)
        sigma_flat = sigma.reshape(sigma.shape[0], -1)
        eps_flat = epsilon.reshape(epsilon.shape[0], -1)

        # Standard synthesis
        z_base = mu_flat + sigma_flat * eps_flat

        # Learned residual
        inp = torch.cat([mu_flat, sigma_flat, eps_flat], dim=-1)
        residual = self.net(inp)

        z_syn = z_base + self.scale * residual
        return z_syn.reshape(mu.shape)

    @torch.no_grad()
    def generate(
        self,
        prototypes: torch.Tensor,
        distributions_std: torch.Tensor,
        num_samples_per_class: int,
    ) -> torch.Tensor:
        """Generate synthetic data matching the original API.

        Args:
            prototypes: [C, *shape]
            distributions_std: [C, *shape]
            num_samples_per_class: N

        Returns:
            [C*N, *shape] synthetic features.
        """
        num_classes = prototypes.shape[0]
        shape = prototypes.shape[1:]
        device = prototypes.device

        means = prototypes.unsqueeze(1).expand(-1, num_samples_per_class, *shape)
        stds = distributions_std.unsqueeze(1).expand(-1, num_samples_per_class, *shape)
        epsilon = torch.randn_like(means)

        # Flatten class*sample dims for forward pass
        means_flat = means.reshape(-1, *shape)
        stds_flat = stds.reshape(-1, *shape)
        eps_flat = epsilon.reshape(-1, *shape)

        z_syn = self.forward(means_flat, stds_flat, eps_flat)
        return z_syn
