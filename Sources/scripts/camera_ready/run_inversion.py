#!/usr/bin/env python3
"""
Experiment 6: Feature-exposure / inversion analysis (informal, NOT formal privacy).

We compare how much a specific input sample can be reconstructed from what each
method transmits:

  - SplitFed : the sample's smashed activation f(x)         (per-sample).
  - H-SFP    : the class-wise prototype mu (mu_only).
  - H-SFP    : the class-wise prototype mu + sigma jitter.

Optimization-based inversion reconstructs an input whose client-extractor features
match the target, then we measure reconstruction quality against the *specific*
sample (MSE/PSNR, + SSIM/LPIPS if installed). SplitFed (per-sample) is expected to
reconstruct the sample far better than H-SFP's aggregated statistics, supporting:
"H-SFP reduces exposure of sample-wise activations because it transmits aggregated
class-wise statistics." This is an empirical exposure comparison, not a DP claim.

Smoke:
  python scripts/camera_ready/run_inversion.py \
      --cfg configs/camera_ready/smoke/hsfp_smoke.yaml \
      --num-samples 3 --iters 100 --tag smoke
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hsfp_common as H

ROOT = H.ROOT
sys.path.insert(0, ROOT)
from camera_ready.inversion import invert_to_match, recon_metrics, save_grid
from camera_ready.io_utils import cr_dir, write_csv, write_json
from camera_ready.latex import write_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/camera_ready/smoke/hsfp_smoke.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--target-class", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--class-probe", type=int, default=64, help="samples to estimate mu/sigma")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--tv", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="cifar100")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device

    cfg = H.load_config(os.path.join(ROOT, args.cfg) if not os.path.isabs(args.cfg) else args.cfg)
    client, edge, cloud = H.build_models(cfg)
    if args.ckpt:
        print("[inversion] checkpoint loaded:", H.load_checkpoint_into(client, edge, cloud, args.ckpt))
    extractor = H.ClientExtractor(client).to(device).eval()

    train_ds, valid_ds, test_ds, _ = H.build_dataset(cfg)
    targets = np.asarray(train_ds.targets)
    cls_idx = np.where(targets == args.target_class)[0]
    rng.shuffle(cls_idx)
    probe_idx = cls_idx[:args.class_probe]
    sample_idx = cls_idx[args.class_probe:args.class_probe + args.num_samples]

    # Class prototype statistics from the client activations.
    probe_imgs = torch.stack([train_ds[i][0] for i in probe_idx]).to(device)
    with torch.no_grad():
        probe_feats = extractor(probe_imgs)             # [P, C, H, W]
    mu = probe_feats.mean(0)                              # [C, H, W]
    sigma = probe_feats.std(0, unbiased=False)           # [C, H, W]

    input_shape = (1,) + tuple(probe_imgs.shape[1:])
    rows = []
    grid_tensors, grid_titles = [], []

    for k, si in enumerate(sample_idx):
        x = train_ds[si][0].unsqueeze(0).to(device)      # [1,C,H,W]
        with torch.no_grad():
            smashed = extractor(x)[0]                      # [C,H,W]

        targets_map = {
            "splitfed_smashed": smashed,
            "hsfp_mu": mu,
            "hsfp_mu_sigma": mu + sigma * torch.randn_like(sigma),
        }
        recons = {}
        for name, tgt in targets_map.items():
            recon = invert_to_match(extractor, tgt, input_shape, device=device,
                                    iters=args.iters, lr=args.lr, tv_weight=args.tv,
                                    seed=args.seed + k)
            recons[name] = recon
            m = recon_metrics(recon, x.cpu())
            rows.append({"sample": int(si), "method": name, **m})

        if k == 0:  # qualitative grid for the first sample
            grid_tensors = [x.cpu(), recons["splitfed_smashed"],
                            recons["hsfp_mu"], recons["hsfp_mu_sigma"]]
            grid_titles = ["original", "SplitFed recon",
                           "H-SFP mu recon", "H-SFP mu+sigma recon"]
        print(f"  sample {si}: "
              + " | ".join(f"{r['method']} PSNR={r['psnr']:.2f}"
                           for r in rows[-3:]))

    out_dir = cr_dir("inversion")
    if grid_tensors:
        save_grid(grid_tensors, grid_titles,
                  os.path.join(out_dir, f"inversion_grid_{args.tag}"))

    write_csv(rows, os.path.join(out_dir, f"inversion_{args.tag}.csv"))
    write_json({"args": vars(args), "rows": rows},
               os.path.join(out_dir, f"inversion_{args.tag}.json"))

    # Aggregate per method (mean over samples) for the LaTeX table.
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for metric in ("mse", "psnr", "ssim", "lpips"):
            if r.get(metric) is not None:
                agg[r["method"]][metric].append(r[metric])
    label = {"splitfed_smashed": "SplitFed (smashed)",
             "hsfp_mu": "H-SFP ($\\mu$)",
             "hsfp_mu_sigma": "H-SFP ($\\mu,\\sigma$)"}
    latex_rows = []
    for method, mdict in agg.items():
        def mean_or_na(key):
            return f"{np.mean(mdict[key]):.3f}" if mdict.get(key) else "--"
        latex_rows.append({
            "method": label.get(method, method),
            "mse": mean_or_na("mse"), "psnr": mean_or_na("psnr"),
            "ssim": mean_or_na("ssim"), "lpips": mean_or_na("lpips"),
        })
    write_table(
        latex_rows,
        columns=[("method", "Transmitted"), ("mse", "MSE $\\downarrow$"),
                 ("psnr", "PSNR $\\uparrow$"), ("ssim", "SSIM $\\uparrow$"),
                 ("lpips", "LPIPS $\\downarrow$")],
        path=os.path.join(out_dir, f"inversion_{args.tag}.tex"),
        caption="Feature-inversion exposure (reconstruction of a specific sample). "
                "H-SFP transmits aggregated class statistics, reducing per-sample "
                "exposure relative to SplitFed smashed activations. Not formal privacy.",
        label="tab:inversion", escape=False,
    )
    print(f"[inversion] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
