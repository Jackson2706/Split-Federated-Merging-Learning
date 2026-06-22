"""
Prototype-synthesis covariance modes and communication-cost accounting.

H-SFP's default synthesis is a diagonal Gaussian per class (mu + sigma * eps).
This module generalizes the class-conditional generative model so we can ablate:

    mu_only              : transmit class mean only; synth = mu (no spread).
    diag_sigma           : H-SFP default; transmit mu + diagonal sigma.
    full_covariance      : transmit mu + full covariance (Cholesky sampling).
    low_rank_covariance  : transmit mu + low-rank factor (rank r) + residual diag.
    mixture_gaussian     : transmit K diagonal Gaussians + mixing weights.

For each mode we provide:
    compute_class_stats(features, mode, ...)  -> stats dict for one class.
    synthesize_from_stats(stats, mode, n, ...)-> [n, D] synthetic features.
    stat_num_floats(mode, D, ...)             -> #float32 transmitted per class.

These are used by the covariance-ablation experiment and (optionally) wired
into the H-SFP pipeline via the `synthesis_mode` config key.
"""

import math

import torch

MODES = ("mu_only", "diag_sigma", "full_covariance",
         "low_rank_covariance", "mixture_gaussian")

TRANSMITTED = {
    "mu_only": r"$\mu$",
    "diag_sigma": r"$\mu,\sigma$",
    "full_covariance": r"$\mu,\Sigma$",
    "low_rank_covariance": r"$\mu,U_r,d$",
    "mixture_gaussian": r"$\{\pi_k,\mu_k,\sigma_k\}$",
}


@torch.no_grad()
def compute_class_stats(features, mode="diag_sigma", rank=8, k=3, eps=1e-5):
    """Compute the statistics for one class from features [N, D]."""
    feats = features.reshape(features.shape[0], -1).float()
    n, d = feats.shape
    mu = feats.mean(0)

    if mode == "mu_only":
        return {"mu": mu, "dim": d}

    if mode == "diag_sigma":
        sigma = feats.std(0, unbiased=False)
        return {"mu": mu, "sigma": sigma, "dim": d}

    if mode == "full_covariance":
        if n > 1:
            cov = torch.cov(feats.T)
        else:
            cov = torch.zeros(d, d, device=feats.device)
        cov = cov + eps * torch.eye(d, device=feats.device)
        return {"mu": mu, "cov": cov, "dim": d}

    if mode == "low_rank_covariance":
        centered = feats - mu
        r = int(min(rank, max(1, min(n - 1, d))))
        # Truncated SVD of centered data: columns of V are principal directions.
        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        except Exception:
            U, S, Vh = torch.svd(centered)
            Vh = Vh.T
        denom = max(n - 1, 1)
        factor = (Vh[:r].T * (S[:r] / math.sqrt(denom)))  # [D, r]
        total_var = centered.pow(2).sum(0) / denom         # [D]
        resid = (total_var - factor.pow(2).sum(1)).clamp_min(eps)  # [D]
        return {"mu": mu, "factor": factor, "resid": resid, "rank": r, "dim": d}

    if mode == "mixture_gaussian":
        comps = _fit_gmm_diag(feats, k=k, eps=eps)
        comps["dim"] = d
        return comps

    raise ValueError(f"unknown synthesis mode: {mode}")


@torch.no_grad()
def synthesize_from_stats(stats, mode, n, device=None, generator=None):
    """Generate [n, D] synthetic features for one class."""
    ref = stats["mu"] if "mu" in stats else stats["mus"]
    if device is None:
        device = ref.device

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    if mode == "mixture_gaussian":
        weights = stats["weights"].to(device)
        mus = stats["mus"].to(device)         # [K, D]
        sigmas = stats["sigmas"].to(device)   # [K, D]
        comp = torch.multinomial(weights, n, replacement=True, generator=generator)
        eps = randn(n, mus.shape[1])
        return mus[comp] + eps * sigmas[comp]

    mu = stats["mu"].to(device)
    d = mu.numel()

    if mode == "mu_only":
        return mu.unsqueeze(0).expand(n, d).clone()

    if mode == "diag_sigma":
        eps = randn(n, d)
        return mu + eps * stats["sigma"].to(device)

    if mode == "full_covariance":
        cov = stats["cov"].to(device)
        L = torch.linalg.cholesky(cov)
        eps = randn(n, d)
        return mu + eps @ L.T

    if mode == "low_rank_covariance":
        factor = stats["factor"].to(device)   # [D, r]
        resid = stats["resid"].to(device)     # [D]
        r = factor.shape[1]
        z = randn(n, r)
        e = randn(n, d)
        return mu + z @ factor.T + e * resid.sqrt()

    raise ValueError(f"unknown synthesis mode: {mode}")


def stat_num_floats(mode, dim, rank=8, k=3):
    """Number of float32 values transmitted per class for a mode."""
    d = dim
    if mode == "mu_only":
        return d
    if mode == "diag_sigma":
        return 2 * d
    if mode == "full_covariance":
        return d + d * (d + 1) // 2          # mu + symmetric covariance
    if mode == "low_rank_covariance":
        return d + d * rank + d              # mu + factor + residual diag
    if mode == "mixture_gaussian":
        return k * (2 * d) + k               # K diag Gaussians + weights
    raise ValueError(f"unknown synthesis mode: {mode}")


def asymptotic_cost(mode, rank=None, k=None):
    """Big-O per class in feature dimension D, as a LaTeX string."""
    return {
        "mu_only": r"$O(D)$",
        "diag_sigma": r"$O(D)$",
        "full_covariance": r"$O(D^2)$",
        "low_rank_covariance": (r"$O(rD)$" if rank is None else fr"$O({rank}D)$"),
        "mixture_gaussian": (r"$O(KD)$" if k is None else fr"$O({k}D)$"),
    }[mode]


def stat_bytes(mode, dim, rank=8, k=3, bytes_per_float=4):
    return stat_num_floats(mode, dim, rank=rank, k=k) * bytes_per_float


@torch.no_grad()
def _fit_gmm_diag(feats, k=3, iters=10, eps=1e-5):
    """Tiny diagonal-covariance GMM via k-means init + a few EM steps.

    Uses sklearn if available, otherwise a minimal torch implementation.
    Returns {weights:[K], mus:[K,D], sigmas:[K,D]}.
    """
    n, d = feats.shape
    k = int(min(k, max(1, n)))

    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=k, covariance_type="diag",
                             max_iter=iters, reg_covar=eps)
        gm.fit(feats.cpu().numpy())
        return {
            "weights": torch.tensor(gm.weights_, dtype=torch.float32),
            "mus": torch.tensor(gm.means_, dtype=torch.float32),
            "sigmas": torch.tensor(gm.covariances_, dtype=torch.float32).clamp_min(eps).sqrt(),
        }
    except Exception:
        pass

    # Fallback: k-means then per-cluster diagonal stats.
    g = torch.Generator(device=feats.device)
    perm = torch.randperm(n, generator=g, device=feats.device)[:k]
    centers = feats[perm].clone()
    assign = torch.zeros(n, dtype=torch.long, device=feats.device)
    for _ in range(iters):
        dists = torch.cdist(feats, centers)
        assign = dists.argmin(1)
        for c in range(k):
            mask = assign == c
            if mask.any():
                centers[c] = feats[mask].mean(0)
    weights, mus, sigmas = [], [], []
    for c in range(k):
        mask = assign == c
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        cf = feats[mask]
        weights.append(cnt / n)
        mus.append(cf.mean(0))
        sigmas.append(cf.std(0, unbiased=False).clamp_min(math.sqrt(eps)))
    w = torch.tensor(weights, device=feats.device)
    w = w / w.sum()
    return {"weights": w, "mus": torch.stack(mus), "sigmas": torch.stack(sigmas)}


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(500, 32) * 2.0 + 1.0
    for m in MODES:
        s = compute_class_stats(X, mode=m, rank=4, k=3)
        y = synthesize_from_stats(s, m, n=100)
        nf = stat_num_floats(m, 32, rank=4, k=3)
        print(f"{m:22s} synth={tuple(y.shape)} floats/class={nf} "
              f"mean_err={(y.mean(0)-X.mean(0)).norm():.3f}")
