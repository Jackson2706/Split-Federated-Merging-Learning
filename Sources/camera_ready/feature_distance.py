"""
Feature-distribution distance metrics: RBF-kernel MMD (and an optional FID-like
Gaussian Frechet distance) between real and synthesized feature sets.

Used by the covariance-ablation experiment to quantify how well each synthesis
mode reproduces the true client feature distribution.
"""

import torch


@torch.no_grad()
def rbf_mmd(X, Y, sigmas=None, max_samples=2000):
    """Unbiased RBF-kernel Maximum Mean Discrepancy^2 between X [n,D] and Y [m,D].

    Uses a sum of Gaussian kernels at multiple bandwidths (median heuristic by
    default) for robustness. Returns a non-negative float (MMD^2).
    """
    X = X.reshape(X.shape[0], -1).float()
    Y = Y.reshape(Y.shape[0], -1).float()
    if X.shape[0] > max_samples:
        X = X[torch.randperm(X.shape[0])[:max_samples]]
    if Y.shape[0] > max_samples:
        Y = Y[torch.randperm(Y.shape[0])[:max_samples]]

    device = X.device
    Y = Y.to(device)

    xx = torch.cdist(X, X) ** 2
    yy = torch.cdist(Y, Y) ** 2
    xy = torch.cdist(X, Y) ** 2

    if sigmas is None:
        med = torch.median(xy.detach())
        med = med if med > 0 else torch.tensor(1.0, device=device)
        base = med
        sigmas = [base * s for s in (0.25, 0.5, 1.0, 2.0, 4.0)]

    n, m = X.shape[0], Y.shape[0]
    mmd2 = torch.tensor(0.0, device=device)
    for s in sigmas:
        g = 1.0 / (2.0 * s + 1e-12)
        k_xx = torch.exp(-g * xx)
        k_yy = torch.exp(-g * yy)
        k_xy = torch.exp(-g * xy)
        # Unbiased estimator: exclude diagonal of k_xx, k_yy.
        sum_xx = (k_xx.sum() - k_xx.diagonal().sum()) / max(n * (n - 1), 1)
        sum_yy = (k_yy.sum() - k_yy.diagonal().sum()) / max(m * (m - 1), 1)
        sum_xy = k_xy.mean()
        mmd2 = mmd2 + (sum_xx + sum_yy - 2.0 * sum_xy)
    return float((mmd2 / len(sigmas)).clamp_min(0.0).item())


@torch.no_grad()
def gaussian_frechet_distance(X, Y, eps=1e-6):
    """FID-style Frechet distance between two feature sets assuming Gaussianity.

    FD^2 = ||mu_x - mu_y||^2 + Tr(Cx + Cy - 2 (Cx Cy)^{1/2}).
    Returns a float. Heavy for large D; intended for modest feature dims.
    """
    X = X.reshape(X.shape[0], -1).float()
    Y = Y.reshape(Y.shape[0], -1).float().to(X.device)
    mu_x, mu_y = X.mean(0), Y.mean(0)
    Cx = torch.cov(X.T) + eps * torch.eye(X.shape[1], device=X.device)
    Cy = torch.cov(Y.T) + eps * torch.eye(Y.shape[1], device=X.device)
    diff = (mu_x - mu_y).dot(mu_x - mu_y)
    # sqrt(Cx Cy) via eigendecomposition of the symmetric product surrogate.
    prod = Cx @ Cy
    evals = torch.linalg.eigvals(prod).real.clamp_min(0.0)
    covmean = evals.sqrt().sum()
    fd = diff + torch.trace(Cx) + torch.trace(Cy) - 2.0 * covmean
    return float(fd.clamp_min(0.0).item())


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.randn(1000, 16)
    B = torch.randn(1000, 16)
    C = torch.randn(1000, 16) * 2 + 3
    print("MMD(A,A~):", rbf_mmd(A, torch.randn(1000, 16)))
    print("MMD(A,B):  ", rbf_mmd(A, B))
    print("MMD(A,C):  ", rbf_mmd(A, C))
