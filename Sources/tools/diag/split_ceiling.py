#!/usr/bin/env python3
"""Split-depth ceiling study for H-SFP on CIFAR-100 (diagnostic only).

For each client split depth d in {layer2, layer3, layer4} of the CIFAR-adapted
ResNet-18 used by classification/H-SFP/models/cnn_cifar.py:
  1. train backbone[:d] + linear head centrally for E epochs on 45k TRAIN images
     (an optimistic stand-in for a well-trained IID federated client encoder),
  2. extract pooled features for those 45k and for a 5k HELD-OUT TRAIN slice,
  3. report on the held-out slice:
     - linear probe (real features)                 -> encoder ceiling
     - gaussian head: trained ONLY on samples from per-class diagonal Gaussians
       (what the H-SFP cloud sees)                   -> synthetic-path ceiling
     - gaussian head after ZCA whitening fit on class means' pooled covariance
       of the TRAIN features (transmittable statistics)
The official CIFAR-100 test split is never touched.
"""
import argparse, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, models

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--root", default="data/cifar")
p.add_argument("--out", default="outputs/diag/split_ceiling.json")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
tr_aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
plain = transforms.Compose([transforms.ToTensor(), norm])
full_aug = datasets.CIFAR100(a.root, train=True, download=False, transform=tr_aug)
full_plain = datasets.CIFAR100(a.root, train=True, download=False, transform=plain)
perm = np.random.RandomState(a.seed).permutation(50000)
tr_idx, va_idx = perm[:45000], perm[45000:]
Sub = torch.utils.data.Subset
DL = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=256, shuffle=sh, num_workers=4, pin_memory=True)


def backbone(depth):
    r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    r.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); r.maxpool = nn.Identity()
    layers = [r.conv1, r.bn1, r.relu, r.layer1, r.layer2]
    dim = 128
    if depth >= 3: layers.append(r.layer3); dim = 256
    if depth >= 4: layers.append(r.layer4); dim = 512
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten()), dim


@torch.no_grad()
def feats(net, ds):
    net.eval(); X, Y = [], []
    for x, y in DL(ds, False):
        X.append(net(x.to(dev, non_blocking=True)).float().cpu()); Y.append(y)
    return torch.cat(X), torch.cat(Y)


def train_head(Xtr, Ytr, Xva, Yva, dim, synth=None, epochs=30):
    head = nn.Linear(dim, 100).to(dev)
    opt = torch.optim.Adam(head.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        if synth is not None:
            Xtr, Ytr = synth()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            loss = F.cross_entropy(head(Xtr[b].to(dev)), Ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (head(Xva.to(dev)).argmax(1).cpu() == Yva).float().mean().item()


def gauss_sampler(mu, sd, n=200):
    def f():
        eps = torch.randn(100, n, mu.shape[1])
        X = (mu[:, None] + sd[:, None] * eps).reshape(-1, mu.shape[1])
        Y = torch.arange(100).repeat_interleave(n)
        return X, Y
    return f


res = {}
for depth in (2, 3, 4):
    t0 = time.time()
    net, dim = backbone(depth); net = net.to(dev)
    clf = nn.Linear(dim, 100).to(dev)
    opt = torch.optim.SGD(list(net.parameters()) + list(clf.parameters()), 0.05, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 0.05, total_steps=a.epochs * (45000 // 256 + 1))
    for ep in range(a.epochs):
        net.train()
        for x, y in DL(Sub(full_aug, tr_idx), True):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(clf(net(x)), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    Xtr, Ytr = feats(net, Sub(full_plain, tr_idx)); Xva, Yva = feats(net, Sub(full_plain, va_idx))
    mu = torch.stack([Xtr[Ytr == c].mean(0) for c in range(100)])
    sd = torch.stack([Xtr[Ytr == c].std(0) for c in range(100)])
    r = {"dim": dim, "train_min": round((time.time() - t0) / 60, 1)}
    r["linear_probe"] = train_head(Xtr, Ytr, Xva, Yva, dim)
    r["gauss_head_raw"] = train_head(None, None, Xva, Yva, dim, synth=gauss_sampler(mu, sd))
    # Whitening from pooled within-class covariance (a d x d statistic the server could hold)
    Xc = Xtr - mu[Ytr]
    W = torch.cov(Xc.T) + 1e-3 * torch.eye(dim)
    ev, U = torch.linalg.eigh(W)
    Z = U @ torch.diag(ev.clamp(min=1e-6).rsqrt()) @ U.T
    mu_w, sd_w = mu @ Z, torch.stack([(Xtr[Ytr == c] @ Z).std(0) for c in range(100)])
    r["gauss_head_whitened"] = train_head(None, None, Xva @ Z, Yva, dim, synth=gauss_sampler(mu_w, sd_w))
    ncm = (torch.cdist(Xva @ Z, mu_w).argmin(1) == Yva).float().mean().item()
    r["nearest_mean_whitened"] = ncm
    res[f"layer{depth}"] = r
    print(f"layer{depth}: {r}", flush=True)
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
