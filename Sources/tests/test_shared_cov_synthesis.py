import torch
from ehsfp.synthesis import SharedCovarianceGenerator, pooled_covariance, within_class_scatter


def test_generator_matches_target_covariance_and_means():
    torch.manual_seed(0)
    A = torch.randn(6, 6); cov = A @ A.T + 0.1 * torch.eye(6)
    g = SharedCovarianceGenerator(); g.set_covariance(cov)
    mu = torch.stack([torch.zeros(6), torch.ones(6) * 5])
    x = g.generate(mu, torch.ones(2, 6), 20000)
    a, b = x[:20000], x[20000:]
    assert torch.allclose(a.mean(0), mu[0], atol=0.1) and torch.allclose(b.mean(0), mu[1], atol=0.1)
    assert torch.allclose(torch.cov(a.T), cov, atol=0.25 * cov.abs().max())


def test_pooled_covariance_from_scatter():
    torch.manual_seed(1)
    feats = {c: list(torch.randn(500, 4) * 2 + c) for c in range(3)}
    cov = pooled_covariance([within_class_scatter(feats)])
    assert torch.allclose(cov, 4 * torch.eye(4), atol=0.5)


def test_fallback_to_diagonal_without_covariance_keeps_shape():
    g = SharedCovarianceGenerator()
    x = g.generate(torch.zeros(3, 8, 1, 1), torch.ones(3, 8, 1, 1), 5)
    assert x.shape == (15, 8, 1, 1)
