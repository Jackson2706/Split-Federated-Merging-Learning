"""
Empirical Lipschitz-sensitivity estimation for prototype statistics.

For a class-conditional batch we compute clean statistics (mu, sigma), perturb
the *inputs* with Gaussian noise of scale rho, recompute (mu', sigma'), and report

    Lstat = (||mu' - mu||_2 + ||sigma' - sigma||_2) / ||x' - x||_2.

IMPORTANT (paper language): this measures *bounded empirical sensitivity* of the
prototype statistics to input perturbations. It does NOT prove representation
preservation or any formal privacy/robustness guarantee.
"""

import torch


@torch.no_grad()
def _class_stats(feats, labels, classes):
    """Return {class: (mu, sigma)} from features [N,D] and labels [N]."""
    out = {}
    for c in classes:
        mask = labels == c
        if mask.sum() < 2:
            continue
        cf = feats[mask].reshape(int(mask.sum()), -1).float()
        out[int(c)] = (cf.mean(0), cf.std(0, unbiased=False))
    return out


@torch.no_grad()
def estimate_lstat(extractor, images, labels, rhos=(0.01, 0.10, 0.20),
                   device="cuda", seed=0, classes=None):
    """Estimate Lstat for each (rho, class).

    Args:
        extractor: client feature extractor (nn.Module) mapping images -> features.
        images: input tensor [N, C, H, W] (already normalized as in training).
        labels: int tensor [N].
        rhos: perturbation scales (std of additive Gaussian noise on inputs).
        classes: iterable of class ids to evaluate (default: all present).

    Returns:
        list of per-(rho, class) record dicts:
          {rho, class, n, delta_mu, delta_sigma, delta_x, lstat}
    """
    extractor = extractor.to(device).eval()
    images = images.to(device)
    labels = labels.to(device)
    if classes is None:
        classes = [int(c) for c in torch.unique(labels)]

    clean_feats = extractor(images)
    clean_stats = _class_stats(clean_feats, labels, classes)

    g = torch.Generator(device=device).manual_seed(seed)
    records = []
    for rho in rhos:
        noise = torch.randn(images.shape, device=device, generator=g) * rho
        pert_images = images + noise
        pert_feats = extractor(pert_images)
        pert_stats = _class_stats(pert_feats, labels, classes)

        for c in classes:
            if c not in clean_stats or c not in pert_stats:
                continue
            mu0, sg0 = clean_stats[c]
            mu1, sg1 = pert_stats[c]
            mask = labels == c
            dx = (noise[mask]).reshape(int(mask.sum()), -1).norm(dim=1).mean()
            dx = float(dx.clamp_min(1e-12))
            dmu = float((mu1 - mu0).norm())
            dsg = float((sg1 - sg0).norm())
            records.append({
                "rho": float(rho),
                "class": int(c),
                "n": int(mask.sum()),
                "delta_mu": dmu,
                "delta_sigma": dsg,
                "delta_x": dx,
                "lstat": (dmu + dsg) / dx,
            })
    return records


def aggregate_lstat(records):
    """Aggregate per-(rho) mean/std across classes/clients/seeds."""
    from collections import defaultdict
    import math
    buckets = defaultdict(list)
    for r in records:
        buckets[r["rho"]].append(r)
    out = []
    for rho in sorted(buckets):
        grp = buckets[rho]
        agg = {"rho": rho, "n_records": len(grp)}
        for key in ("delta_mu", "delta_sigma", "delta_x", "lstat"):
            vals = [g[key] for g in grp]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
        out.append(agg)
    return out
