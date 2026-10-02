#!/usr/bin/env python3
"""
Experiment 4: Empirical Lipschitz-sensitivity (Lstat) of prototype statistics.

For selected clients/classes/batches we compute clean (mu, sigma) from the client
feature extractor, perturb the *inputs* with Gaussian noise of scale rho, recompute
(mu', sigma'), and report

    Lstat = (||Delta mu||_2 + ||Delta sigma||_2) / ||Delta x||_2,   rho in {0.01,0.10,0.20}.

IMPORTANT: this supports a claim of *bounded empirical sensitivity* only. It does
NOT prove representation preservation or any formal privacy/robustness guarantee.

Smoke:
  python scripts/camera_ready/run_lstat.py \
      --cfg configs/camera_ready/smoke/hsfp_smoke.yaml \
      --num-clients 2 --batch 64 --tag smoke
Full:
  python scripts/camera_ready/run_lstat.py \
      --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
      --ckpt classification/H-SFP/checkpoint_hfl.pt --num-clients 10 --tag cifar100
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hsfp_common as H

ROOT = H.ROOT
sys.path.insert(0, ROOT)
from camera_ready.lstat import estimate_lstat, aggregate_lstat
from camera_ready.io_utils import cr_dir, write_csv, write_json
from camera_ready.latex import write_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/camera_ready/smoke/hsfp_smoke.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--rhos", nargs="+", type=float, default=[0.01, 0.10, 0.20])
    ap.add_argument("--num-clients", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
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
        print("[lstat] checkpoint loaded:", H.load_checkpoint_into(client, edge, cloud, args.ckpt))
    extractor = H.ClientExtractor(client).to(device).eval()

    print("[lstat] loading dataset & partition ...")
    train_ds, valid_ds, test_ds, user_groups = H.build_dataset(cfg)

    client_ids = list(user_groups.keys())[:args.num_clients]
    all_records = []
    for cid in client_ids:
        idx = list(user_groups[cid])
        rng.shuffle(idx)
        idx = idx[:args.batch]
        loader = DataLoader(Subset(train_ds, idx), batch_size=args.batch, shuffle=False)
        images, labels = next(iter(loader))
        recs = estimate_lstat(extractor, images, labels, rhos=args.rhos,
                              device=device, seed=args.seed)
        for r in recs:
            r["client"] = int(cid)
        all_records.extend(recs)
        print(f"  client {cid}: {len(recs)} (rho,class) records")

    agg = aggregate_lstat(all_records)

    out_dir = cr_dir("lstat")
    write_json({"args": vars(args), "raw": all_records, "aggregated": agg},
               os.path.join(out_dir, f"lstat_{args.tag}.json"))
    write_csv(all_records, os.path.join(out_dir, f"lstat_{args.tag}_raw.csv"))
    write_csv(agg, os.path.join(out_dir, f"lstat_{args.tag}_agg.csv"))

    latex_rows = [{
        "rho": f"{a['rho']:.2f}",
        "dmu": f"{a['delta_mu_mean']:.4f} $\\pm$ {a['delta_mu_std']:.4f}",
        "dsigma": f"{a['delta_sigma_mean']:.4f} $\\pm$ {a['delta_sigma_std']:.4f}",
        "lstat": f"{a['lstat_mean']:.4f} $\\pm$ {a['lstat_std']:.4f}",
    } for a in agg]
    write_table(
        latex_rows,
        columns=[("rho", "Perturbation $\\rho$"), ("dmu", "$\\|\\Delta\\mu\\|_2$"),
                 ("dsigma", "$\\|\\Delta\\sigma\\|_2$"), ("lstat", "$L_{\\mathrm{stat}}$")],
        path=os.path.join(out_dir, f"lstat_{args.tag}.tex"),
        caption="Empirical Lipschitz sensitivity of prototype statistics to input "
                "perturbations (CIFAR-100). Bounded empirical sensitivity only.",
        label="tab:lstat", escape=False,
    )
    print(f"[lstat] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
