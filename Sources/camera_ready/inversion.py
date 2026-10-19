"""
Feature-inversion / exposure analysis (informal, NOT formal privacy).

We compare how much input information can be reconstructed from what each method
transmits:

    SplitFed  : sample-wise smashed activations f(x)   (per-sample).
    H-SFP     : aggregated class-wise statistics (mu_only or mu+sigma).

Attack: optimization-based inversion. We optimize a synthetic input x' so that the
client feature extractor maps it close to a target representation, with a small
total-variation prior for image plausibility.

    SplitFed target : a single sample's activation  -> reconstruct that sample.
    H-SFP target    : a class prototype (mu), optionally with sigma-jitter
                      -> reconstruct a class-representative (no specific sample).

Language for the paper: H-SFP reduces exposure of sample-wise activations because
it transmits aggregated class-wise statistics rather than per-sample smashed data.
This is an empirical exposure comparison, NOT a differential-privacy guarantee.
"""

import torch
import torch.nn.functional as F


def _tv(x):
    """Total-variation regularizer for [B,C,H,W]."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def invert_to_match(extractor, target_feat, input_shape, device="cuda",
                    iters=300, lr=0.1, tv_weight=1e-2, seed=0, init=None):
    """Reconstruct an input whose features match `target_feat`.

    Args:
        extractor: client feature extractor (frozen).
        target_feat: target feature tensor [*feat_shape] (single) — broadcast over batch.
        input_shape: tuple (B, C, H, W) for the reconstruction.
        init: optional initial input tensor.

    Returns:
        reconstructed input tensor [B, C, H, W] (detached, on CPU).
    """
    extractor = extractor.to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)

    g = torch.Generator(device=device).manual_seed(seed)
    if init is None:
        x = torch.randn(input_shape, device=device, generator=g) * 0.1
    else:
        x = init.clone().to(device)
    x.requires_grad_(True)

    target = target_feat.to(device).detach()
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        feat = extractor(x)
        # Match the (batch-averaged) feature to the target representation.
        feat_m = feat.mean(0) if feat.shape[0] != target.shape[0] else feat
        loss = F.mse_loss(feat_m, target.expand_as(feat_m)) + tv_weight * _tv(x)
        loss.backward()
        opt.step()
    return x.detach().cpu()


@torch.no_grad()
def recon_metrics(recon, reference):
    """Reconstruction quality metrics. recon/reference: [B,C,H,W] in same range.

    Returns dict with mse, psnr, and (if available) ssim, lpips.
    """
    recon = recon.float()
    reference = reference.float().to(recon.device)
    mse = F.mse_loss(recon, reference).item()
    # PSNR assuming data range estimated from the reference span.
    rng = (reference.max() - reference.min()).clamp_min(1e-8).item()
    psnr = 10.0 * torch.log10(torch.tensor(rng ** 2 / max(mse, 1e-12))).item()
    out = {"mse": mse, "psnr": psnr}

    try:  # optional SSIM
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        a = recon.cpu().numpy(); b = reference.cpu().numpy()
        vals = []
        for i in range(a.shape[0]):
            vals.append(ssim(a[i].transpose(1, 2, 0), b[i].transpose(1, 2, 0),
                             channel_axis=2, data_range=rng))
        out["ssim"] = float(np.mean(vals))
    except Exception:
        out["ssim"] = None

    try:  # optional LPIPS
        import lpips as _lpips
        net = _lpips.LPIPS(net="alex").to(recon.device)
        out["lpips"] = float(net(recon.to(recon.device), reference).mean().item())
    except Exception:
        out["lpips"] = None
    return out


def save_grid(tensors, titles, path, nrow=None):
    """Save a comparison grid PNG/PDF. tensors: list of [C,H,W] (or [1,C,H,W])."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[save_grid] matplotlib unavailable; skipping.")
        return []
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    imgs = []
    for t in tensors:
        t = t.detach().cpu().float()
        if t.dim() == 4:
            t = t[0]
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        imgs.append(t.permute(1, 2, 0).numpy())

    n = len(imgs)
    nrow = nrow or n
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(2.2 * nrow, 2.2 * ncol))
    axes = [axes] if n == 1 else (axes.flatten() if hasattr(axes, "flatten") else axes)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i].clip(0, 1))
            if i < len(titles):
                ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    saved = []
    for ext in ("png", "pdf"):
        p = path if path.endswith(ext) else f"{path}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        saved.append(p)
    plt.close(fig)
    return saved
