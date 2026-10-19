"""MiSFP synthetic failure-mode diagnostic (seeded).

One class = equal-weight N(-2, 0.1^2) and N(2, 0.1^2) in the first coordinate
(second coordinate N(0, 0.5^2) for the 2-D view). Four simulated clients each
hold part of the class; compare synthetic samples from

  single   : moment-matched single Gaussian (what one mean/std packet carries)
  retained : clients' local K=2 components pooled at the edge, no merging
  merged   : retained pool greedily merged (W2 cost) to 2 components

against true samples on: held-out log-likelihood (1-D and 2-D) and the
fraction of synthetic points in the predefined low-density gap |x1| < 1.
This demonstrates a failure mode; it is not evidence of dataset-level gains.

    python scripts/misfp/synthetic_diagnostic.py [--out results/misfp/synthetic]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from misfp import (FitConfig, fit_class_mixture, make_generator, mixture_log_prob,  # noqa: E402
                   sample_mixture)
from misfp.merge import greedy_merge_to_cap  # noqa: E402
from misfp.packets import concat_mixtures  # noqa: E402

SEED = 20261002
GAP = 1.0  # |x1| < GAP is the predefined low-density region
N_CLIENT, N_PER_CLIENT, N_HELDOUT, N_SYN = 4, 100, 4000, 4000


def true_samples(n, gen, dims):
    mode = torch.randint(0, 2, (n,), generator=gen)
    x1 = torch.where(mode == 0, -2.0, 2.0) + 0.1 * torch.randn(n, generator=gen, dtype=torch.float64)
    if dims == 1:
        return x1[:, None]
    x2 = 0.5 * torch.randn(n, generator=gen, dtype=torch.float64)
    return torch.stack([x1, x2], 1)


def true_log_prob(X):
    lp = []
    for m in (-2.0, 2.0):
        l1 = -0.5 * (np.log(2 * np.pi * 0.01) + (X[:, 0] - m) ** 2 / 0.01)
        if X.shape[1] == 2:
            l1 = l1 - 0.5 * (np.log(2 * np.pi * 0.25) + X[:, 1] ** 2 / 0.25)
        lp.append(l1 + np.log(0.5))
    return torch.logsumexp(torch.stack(lp, 1), 1)


def run(dims):
    gen = make_generator(SEED, "data", dims)
    clients = [true_samples(N_PER_CLIENT, gen, dims) for _ in range(N_CLIENT)]
    heldout = true_samples(N_HELDOUT, gen, dims)
    cfg = FitConfig(k=2)
    local = [fit_class_mixture(X, 0, cfg, make_generator(SEED, "fit", dims, i))[0]
             for i, X in enumerate(clients)]
    retained = concat_mixtures(local)
    models = {"single": retained.collapse(), "retained": retained,
              "merged": greedy_merge_to_cap(retained, 2)}
    out, samples = {"dims": dims}, {"true": true_samples(N_SYN, gen, dims)}
    out["true"] = {"heldout_loglik": float(true_log_prob(heldout).mean()),
                   "gap_fraction": float((samples["true"][:, 0].abs() < GAP).double().mean())}
    for name, cm in models.items():
        Z, _ = sample_mixture(cm, N_SYN, make_generator(SEED, "syn", dims, name))
        samples[name] = Z
        out[name] = {"components": cm.R,
                     "heldout_loglik": float(mixture_log_prob(heldout, cm, 1e-12).mean()),
                     "gap_fraction": float((Z[:, 0].abs() < GAP).double().mean()),
                     "weights": [round(float(w), 4) for w in cm.weights]}
    # Exactness check: every model has the same total count/mean/variance.
    ref = models["retained"].collapse()
    for name, cm in models.items():
        c = cm.collapse()
        assert abs(c.total - ref.total) < 1e-9
        assert torch.allclose(c.means, ref.means) and torch.allclose(c.variances, ref.variances)
    return out, samples


def plot(results, samples, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, series, ref = "#0b0b0b", "#52514e", "#2a78d6", "#8a8984"
    names = [("true", "True samples"), ("single", "Moment-matched single Gaussian"),
             ("retained", "Retained mixture (8 comps)"), ("merged", "Merged mixture (2 comps)")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), constrained_layout=True)
    grid = np.linspace(-3.5, 3.5, 400)
    true_pdf = 0.5 * (np.exp(-(grid + 2) ** 2 / 0.02) + np.exp(-(grid - 2) ** 2 / 0.02)) / np.sqrt(2 * np.pi * 0.01)
    r1, s1 = results[1], samples[1]
    r2, s2 = results[2], samples[2]
    for j, (key, title) in enumerate(names):
        ax = axes[0, j]
        ax.axvspan(-GAP, GAP, color="#e9e8e4", lw=0, zorder=0)
        ax.hist(s1[key][:, 0].numpy(), bins=120, range=(-3.5, 3.5), density=True, color=series, zorder=2)
        ax.plot(grid, true_pdf, color=ref, lw=1.2, ls="--", zorder=3)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ll = r1[key]["heldout_loglik"]
        ax.text(0.02, 0.97, f"held-out LL {ll:.2f}\ngap frac {r1[key]['gap_fraction']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=muted)
        ax.set_ylim(0, 2.3)
        ax.set_yticks([])
        ax = axes[1, j]
        ax.axvspan(-GAP, GAP, color="#e9e8e4", lw=0, zorder=0)
        Z = s2[key].numpy()[:1500]
        ax.scatter(Z[:, 0], Z[:, 1], s=4, color=series, alpha=0.35, lw=0, zorder=2)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-2, 2)
        ax.text(0.02, 0.97, f"held-out LL {r2[key]['heldout_loglik']:.2f}\ngap frac {r2[key]['gap_fraction']:.3f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=muted)
        for a in axes[:, j]:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color("#c9c8c3")
            a.tick_params(colors=muted, labelsize=8)
    axes[0, 0].set_ylabel("1-D density (dashed: true)", color=muted, fontsize=9)
    axes[1, 0].set_ylabel("2-D: x2", color=muted, fontsize=9)
    for a in axes[1]:
        a.set_xlabel("x1  (shaded: low-density gap |x1|<1)", color=muted, fontsize=8.5)
    fig.suptitle("MiSFP synthetic diagnostic: one class, two narrow modes, 4 clients (seed %d)" % SEED,
                 fontsize=11, color=ink, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "misfp", "synthetic"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res, smp = {}, {}
    for dims in (1, 2):
        res[dims], smp[dims] = run(dims)
    with open(os.path.join(args.out, "synthetic_diagnostic.json"), "w") as fh:
        json.dump({"seed": SEED, "gap": f"|x1|<{GAP}", "n_heldout": N_HELDOUT, "n_synthetic": N_SYN,
                   "clients": N_CLIENT, "per_client": N_PER_CLIENT,
                   "results": {str(k): v for k, v in res.items()}}, fh, indent=2)
    plot(res, smp, os.path.join(args.out, "synthetic_diagnostic.png"))
    for dims in (1, 2):
        print(f"--- {dims}-D")
        for k in ("true", "single", "retained", "merged"):
            r = res[dims][k]
            print(f"{k:9s} comps={r.get('components', '-')!s:3s} heldout_LL={r['heldout_loglik']:8.3f} "
                  f"gap_frac={r['gap_fraction']:.4f}")


if __name__ == "__main__":
    main()
