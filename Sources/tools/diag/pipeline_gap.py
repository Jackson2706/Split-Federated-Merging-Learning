#!/usr/bin/env python3
"""Locate where accuracy is lost in an H-SFP checkpoint (diagnostic only).

Uses the run's own val_from_train split (held-out TRAIN images; test untouched):
  encoder_probe    linear head on REAL client features (train minus val) -> val
  gauss_head       linear head on per-class diagonal-Gaussian samples of those features
  pipeline         the checkpoint's own client->edge->cloud on val
  cloud_on_real_edge  a linear head on REAL edge outputs (is the edge destroying info?)
"""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "classification/H-SFP")
from torchvision import datasets, transforms
from models.cnn_cifar import ClientModel, EdgeModel, CloudModel

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ck["config"]; sd = ck["model_state_dict"]; dev = torch.device("cuda")
d = int(cfg.get("client_split_depth", 2))
res = bool(cfg.get("edge_residual", False)); D = ClientModel.SPLIT_DIMS[d]
cl = ClientModel(split_depth=d); ed = EdgeModel(in_dim=D, residual=res); cd = CloudModel({**cfg, "cloud_in_dim": D if res else 256})
for m, p in ((cl, "client."), (ed, "edge."), (cd, "cloud.")):
    m.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)})
    m.to(dev).eval()
norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
ds = datasets.CIFAR100(cfg["dataset_root"], train=True, transform=transforms.Compose([transforms.ToTensor(), norm]))
rng = np.random.RandomState(int(cfg["seed"]) + 12345)
val = np.sort(rng.choice(len(ds), int(round(float(cfg["val_from_train"]) * len(ds))), replace=False))
tr = np.setdiff1d(np.arange(len(ds)), val)


@torch.no_grad()
def run(idx):
    L = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=64, num_workers=2)
    Fc, Fe, P, Y = [], [], [], []
    for x, y in L:
        x = x.to(dev); fc = cl(x); fe = ed(fc)
        Fc.append(fc.flatten(1).float().cpu()); Fe.append(fe.flatten(1).float().cpu())
        P.append(cd(fe).argmax(1).cpu()); Y.append(y)
    return torch.cat(Fc), torch.cat(Fe), torch.cat(P), torch.cat(Y)


def head(X, Y, Xv, Yv, synth=None):
    h = nn.Linear(X.shape[1] if X is not None else Xv.shape[1], 100).to(dev)
    o = torch.optim.Adam(h.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(30):
        if synth: X, Y = synth()
        idx = torch.randperm(len(X))
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            l = F.cross_entropy(h(X[b].to(dev)), Y[b].to(dev)); o.zero_grad(); l.backward(); o.step()
    with torch.no_grad():
        return (h(Xv.to(dev)).argmax(1).cpu() == Yv).float().mean().item()


Fc, Fe, _, Y = run(tr); Fcv, Fev, Pv, Yv = run(val)
mu = torch.stack([Fc[Y == c].mean(0) for c in range(100)]); sdv = torch.stack([Fc[Y == c].std(0) for c in range(100)])
def g():
    e = torch.randn(100, 200, mu.shape[1]); return (mu[:, None] + sdv[:, None] * e).reshape(-1, mu.shape[1]), torch.arange(100).repeat_interleave(200)
r = {"round": ck["epoch"], "pipeline": (Pv == Yv).float().mean().item(),
     "encoder_probe": head(Fc, Y, Fcv, Yv), "gauss_head": head(None, None, Fcv, Yv, synth=g),
     "probe_on_edge_output": head(Fe, Y, Fev, Yv)}
print(json.dumps(r))


# ---- synthetic-path variants (all statistics from TRAIN features only)
def white(M):
    ev, U = torch.linalg.eigh(M + 1e-3 * torch.eye(M.shape[0]))
    return U @ torch.diag(ev.clamp(min=1e-6).rsqrt()) @ U.T
def gs(Z):
    m = mu @ Z; s = torch.stack([(Fc[Y == c] @ Z).std(0) for c in range(100)])
    def f():
        e = torch.randn(100, 200, m.shape[1]); return (m[:, None] + s[:, None] * e).reshape(-1, m.shape[1]), torch.arange(100).repeat_interleave(200)
    return f
Zw = white(torch.cov((Fc - mu[Y]).T))            # pooled within-class covariance (needs d x d upload)
Zb = white(torch.cov(mu.T))                       # between-class covariance of prototypes (free at server)
mean_c = mu.mean(0)
r2 = {"gauss_within_white": head(None, None, Fcv @ Zw, Yv, synth=gs(Zw)),
      "gauss_between_white": head(None, None, (Fcv - mean_c) @ Zb, Yv, synth=gs(Zb) if False else (lambda: (lambda X, Y_: ((X - mean_c) @ Zb, Y_))(*g()))),
      "ncm_raw": (torch.cdist(Fcv, mu).argmin(1) == Yv).float().mean().item(),
      "ncm_cosine": (F.normalize(Fcv, dim=1) @ F.normalize(mu - 0, dim=1).T).argmax(1).eq(Yv).float().mean().item()}
print(json.dumps(r2))


# ---- estimation-noise study: statistics from one round's worth of data
def sub_stats(n_samples, seed):
    g_ = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(Fc), generator=g_)[:n_samples]
    X, Yi = Fc[idx], Y[idx]
    m = torch.stack([X[Yi == c].mean(0) if (Yi == c).any() else mu[c] for c in range(100)])
    C = torch.cov((X - m[Yi]).T)
    return m, C
def cov_head(m, C, shrink=0.0):
    if shrink > 0:
        C = (1 - shrink) * C + shrink * C.diagonal().mean() * torch.eye(C.shape[0])
    L = torch.linalg.cholesky(C + 1e-4 * C.diagonal().mean() * torch.eye(C.shape[0]))
    def f():
        e = torch.randn(100 * 200, m.shape[1]); return m.repeat_interleave(200, 0) + e @ L.T, torch.arange(100).repeat_interleave(200)
    return head(None, None, Fcv, Yv, synth=f)
out = {}
for n, tag in ((900, "edge_round_900"), (4500, "round_4500"), (45000 - 0, "all")):
    m, C = sub_stats(min(n, len(Fc)), 0)
    out[tag] = {s: cov_head(m, C, s) for s in (0.0, 0.3, 0.7)}
print(json.dumps(out))
