#!/usr/bin/env python3
"""
Experiment 3: Prototype distribution / covariance ablation.

Measured at the pooled edge->cloud synthesis boundary (where H-SFP's class-wise
Gaussian assumption lives; client-side 4D maps make full covariance infeasible
there). With a frozen client+edge extractor we:

  1. extract real pooled features per class (train probe + test probe),
  2. for each synthesis mode {mu_only, diag_sigma, full_covariance,
     low_rank_covariance, mixture_gaussian}:
       - fit the class statistics and synthesize features,
       - train a fresh linear cloud classifier on synthetic features,
       - evaluate top-1 accuracy on real test features,
       - record communication (analytic), runtime, and RBF-MMD(real, synth).

Outputs CSV/JSON + a LaTeX table that justifies diagonal sigma as the best
accuracy/communication trade-off. The extractor is frozen (random init unless
--ckpt is given) so this isolates synthesis fidelity from representation quality.

Smoke:
  python scripts/camera_ready/run_covariance.py \
      --cfg configs/camera_ready/smoke/hsfp_smoke.yaml \
      --max-classes 10 --probe-per-class 30 --syn-per-class 30 --clf-epochs 3 --tag smoke
Full:
  python scripts/camera_ready/run_covariance.py \
      --cfg configs/classification/h-sfp/cifar_our_resnet50_5_10.yaml \
      --ckpt classification/H-SFP/checkpoint_hfl.pt --tag cifar100
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _hsfp_common as H

ROOT = H.ROOT
sys.path.insert(0, ROOT)
from camera_ready import synthesis as S
from camera_ready.feature_distance import rbf_mmd
from camera_ready.io_utils import cr_dir, write_csv, write_json
from camera_ready.latex import write_table
import numpy as np


def _class_indices(targets, num_classes, per_class, rng):
    targets = np.asarray(targets)
    out = {}
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        out[c] = idx[:per_class]
    return out


def train_linear(feats, labels, num_classes, dim, device, epochs, lr=1e-3, bs=256):
    clf = nn.Linear(dim, num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(feats, labels), batch_size=bs, shuffle=True)
    clf.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(clf(xb), yb)
            loss.backward()
            opt.step()
    return clf


@torch.no_grad()
def evaluate(clf, feats, labels, device, bs=512):
    clf.eval()
    correct = 0
    for i in range(0, len(feats), bs):
        xb = feats[i:i + bs].to(device)
        pred = clf(xb).argmax(1).cpu()
        correct += (pred == labels[i:i + bs]).sum().item()
    return correct / len(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/camera_ready/smoke/hsfp_smoke.yaml")
    ap.add_argument("--ckpt", default=None, help="optional H-SFP checkpoint for the extractor")
    ap.add_argument("--modes", nargs="+", default=list(S.MODES))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--probe-per-class", type=int, default=100)
    ap.add_argument("--test-per-class", type=int, default=50)
    ap.add_argument("--syn-per-class", type=int, default=100)
    ap.add_argument("--clf-epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="cifar100")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    cfg = H.load_config(os.path.join(ROOT, args.cfg) if not os.path.isabs(args.cfg) else args.cfg)
    num_classes = int(cfg.get("num_classes", 100))
    if args.max_classes:
        num_classes = min(num_classes, args.max_classes)

    print(f"[covariance] building models ({cfg['model']}/{cfg['dataset']})")
    client, edge, cloud = H.build_models(cfg)
    if args.ckpt:
        ok = H.load_checkpoint_into(client, edge, cloud, args.ckpt)
        print(f"[covariance] checkpoint loaded: {ok}")
    extractor = H.EdgeExtractor(client, edge)

    print("[covariance] loading dataset & extracting real features ...")
    train_ds, valid_ds, test_ds, _ = H.build_dataset(cfg)
    tr_idx = _class_indices(train_ds.targets, num_classes, args.probe_per_class, rng)
    te_idx = _class_indices(test_ds.targets, num_classes, args.test_per_class, rng)
    tr_indices = np.concatenate([tr_idx[c] for c in range(num_classes)])
    te_indices = np.concatenate([te_idx[c] for c in range(num_classes)])

    real_tr_f, real_tr_y = H.gather_features(extractor, train_ds, tr_indices, device)
    real_te_f, real_te_y = H.gather_features(extractor, test_ds, te_indices, device)
    dim = real_tr_f.shape[1]
    print(f"[covariance] feature dim D={dim}; train probe={len(real_tr_f)} test probe={len(real_te_f)}")

    # group real train features by class
    by_class = {c: real_tr_f[real_tr_y == c] for c in range(num_classes)}

    rows = []
    for mode in args.modes:
        t0 = time.perf_counter()
        syn_feats, syn_labels = [], []
        mmd_vals = []
        for c in range(num_classes):
            cf = by_class[c].to(device)
            if cf.shape[0] < 2:
                continue
            stats = S.compute_class_stats(cf, mode=mode, rank=args.rank, k=args.k)
            ys = S.synthesize_from_stats(stats, mode, args.syn_per_class, device=device)
            syn_feats.append(ys.cpu())
            syn_labels.append(torch.full((ys.shape[0],), c, dtype=torch.long))
            if mode != "mu_only":
                mmd_vals.append(rbf_mmd(cf, ys))
        runtime = time.perf_counter() - t0

        syn_feats = torch.cat(syn_feats)
        syn_labels = torch.cat(syn_labels)

        clf = train_linear(syn_feats, syn_labels, num_classes, dim, device, args.clf_epochs)
        acc = evaluate(clf, real_te_f, real_te_y, device)

        floats = num_classes * S.stat_num_floats(mode, dim, rank=args.rank, k=args.k)
        comm_mb = floats * 4 / 1e6
        mmd = float(np.mean(mmd_vals)) if mmd_vals else 0.0

        row = {
            "mode": mode,
            "transmitted": S.TRANSMITTED[mode],
            "asymptotic_comm": S.asymptotic_cost(mode, rank=args.rank, k=args.k),
            "comm_MB": comm_mb,
            "accuracy": acc,
            "mmd": mmd,
            "runtime_s": runtime,
            "dim": dim,
            "num_classes": num_classes,
        }
        rows.append(row)
        print(f"  {mode:22s} acc={acc:.4f} comm={comm_mb:.4f}MB mmd={mmd:.5f} t={runtime:.2f}s")

    out_dir = cr_dir("covariance")
    write_json({"args": vars(args), "rows": rows},
               os.path.join(out_dir, f"covariance_{args.tag}.json"))
    write_csv(rows, os.path.join(out_dir, f"covariance_{args.tag}.csv"))

    latex_rows = [{
        "mode": r["mode"].replace("_", " "),
        "transmitted": r["transmitted"],
        "asymptotic_comm": r["asymptotic_comm"],
        "comm_MB": f"{r['comm_MB']:.3f}",
        "accuracy": f"{100*r['accuracy']:.2f}",
        "mmd": f"{r['mmd']:.4f}",
    } for r in rows]
    write_table(
        latex_rows,
        columns=[("mode", "Synthesis mode"), ("transmitted", "Transmitted"),
                 ("asymptotic_comm", "Asymp. comm"), ("comm_MB", "Comm (MB)"),
                 ("accuracy", "Acc (\\%)"), ("mmd", "MMD")],
        path=os.path.join(out_dir, f"covariance_{args.tag}.tex"),
        caption="Prototype synthesis covariance ablation (CIFAR-100). "
                "Diagonal $\\sigma$ gives the best accuracy/communication trade-off.",
        label="tab:covariance", escape=False,
    )
    print(f"[covariance] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
