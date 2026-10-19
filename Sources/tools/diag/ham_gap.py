#!/usr/bin/env python3
"""HAM10000 synthetic-statistics diagnostic (training images only; test untouched).

Uses a run's best checkpoint encoder, extracts client features for training
images, holds out 1,500 of them, and compares heads trained on synthetic
Gaussians built from per-client statistics computed the way the pipeline does
(a) vs with between-client spread (b) vs (b)+class-frequency prior (c), against
a real-feature linear probe (d). Reports accuracy and macro-F1 on the held-out
training images.
"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "classification/H-SFP")
from sklearn.metrics import f1_score
from models.resnet50_ham10000 import HAM10000ClientModelResNet50
from data.get_data import get_dataset

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cfg = ck["config"]; dev = torch.device("cuda")
cl = HAM10000ClientModelResNet50(split_depth=4)
cl.load_state_dict({k[7:]: v for k, v in ck["model_state_dict"].items() if k.startswith("client.")}); cl.to(dev).eval()
tr, _, _, ug = get_dataset(cfg)
owner = np.full(len(tr), -1)
for c, idx in ug.items():
    owner[np.array(list(idx), dtype=int)] = c
rng = np.random.RandomState(0); allidx = np.where(owner >= 0)[0]; rng.shuffle(allidx)
hold, fit = allidx[:1500], allidx[1500:]

@torch.no_grad()
def feats(idx):
    X, Y = [], []
    for i in range(0, len(idx), 64):
        b = [tr[j] for j in idx[i:i+64]]
        x = torch.stack([t[0] for t in b]).to(dev); y = torch.tensor([int(t[1]) for t in b])
        X.append(cl(x).flatten(1).float().cpu()); Y.append(y)
    return torch.cat(X), torch.cat(Y)
Xf, Yf = feats(fit); Xh, Yh = feats(hold); of = owner[fit]
C = 7
def head(gen=None, X=None, Y=None):
    h = nn.Linear(Xf.shape[1], C).to(dev); o = torch.optim.Adam(h.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(30):
        if gen: X, Y = gen()
        p = torch.randperm(len(X))
        for i in range(0, len(p), 256):
            b = p[i:i+256]; l = F.cross_entropy(h(X[b].to(dev)), Y[b].to(dev)); o.zero_grad(); l.backward(); o.step()
    with torch.no_grad():
        pr = h(Xh.to(dev)).argmax(1).cpu()
    return round((pr == Yh).float().mean().item(), 4), round(f1_score(Yh, pr, average="macro"), 4), np.bincount(pr.numpy(), minlength=C).tolist()
# per-client per-class stats (as clients transmit them)
stats = {}
for c in np.unique(of):
    m = of == c
    for k in range(C):
        mk = m & (Yf.numpy() == k)
        if mk.sum():
            x = Xf[torch.from_numpy(mk)]
            stats[(c, k)] = (x.mean(0), x.std(0, unbiased=False), int(mk.sum()))
def agg(total_var):
    mu, sd, n = [], [], []
    for k in range(C):
        L = [v for (c, kk), v in stats.items() if kk == k]
        M = torch.stack([v[0] for v in L]); S = torch.stack([v[1] for v in L]); N = torch.tensor([v[2] for v in L], dtype=torch.float)
        if total_var:
            w = (N / N.sum())[:, None]; m = (w * M).sum(0); var = (w * (S ** 2 + (M - m) ** 2)).sum(0)
        else:
            m = M.mean(0); var = (S ** 2).mean(0)
        mu.append(m); sd.append(var.sqrt()); n.append(N.sum())
    return torch.stack(mu), torch.stack(sd), torch.stack(n)
def gen(mu, sd, counts):
    def f():
        X = torch.cat([mu[k] + sd[k] * torch.randn(int(c), mu.shape[1]) for k, c in enumerate(counts)])
        Y = torch.cat([torch.full((int(c),), k) for k, c in enumerate(counts)]); return X, Y
    return f
mu_a, sd_a, n = agg(False); mu_b, sd_b, _ = agg(True)
bal = [400] * C; prior = (n / n.sum() * 400 * C).round().clamp(min=5).tolist()
print("holdout true dist", np.bincount(Yh.numpy(), minlength=C).tolist())
print("(a) pipeline stats, balanced     ", head(gen(mu_a, sd_a, bal)))
print("(b) total-variance, balanced     ", head(gen(mu_b, sd_b, bal)))
print("(c) total-variance, class prior  ", head(gen(mu_b, sd_b, prior)))
print("(d) real-feature linear probe    ", head(X=Xf, Y=Yf))

# ---- where does the real pipeline lose it? (same held-out training images)
from models.cnn_cifar import EdgeModel
from models.resnet50_ham10000 import HAM10000CloudModelResNet50
sd_ = ck["model_state_dict"]
ed = EdgeModel(in_dim=2048, residual=True); ed.load_state_dict({k[5:]: v for k, v in sd_.items() if k.startswith("edge.")}); ed.to(dev).eval()
cd = HAM10000CloudModelResNet50(cfg); cd.load_state_dict({k[6:]: v for k, v in sd_.items() if k.startswith("cloud.")}); cd.to(dev).eval()
with torch.no_grad():
    E = ed(Xh.to(dev)[:, :, None, None]).flatten(1); P = cd(E).argmax(1).cpu()
    Ef = ed(Xf.to(dev)[:, :, None, None]).flatten(1).cpu()
print("(e) checkpoint pipeline           ", round((P == Yh).float().mean().item(), 4), np.bincount(P.numpy(), minlength=C).tolist())
print("    edge residual norm ratio      ", round(float((E.cpu() - Xh).norm() / Xh.norm()), 3))
Xh_backup = Xh
Xh = E.cpu()
print("(f) probe on edge outputs         ", head(X=Ef, Y=Yf))

# ---- the actual HAM cloud architecture + pipeline training settings on synthetic stats
Xh = Xh_backup
def cloud_train(bs, epochs, rounds=1, l2norm=False):
    m = HAM10000CloudModelResNet50(cfg).to(dev); o = torch.optim.Adam(m.parameters(), 1e-4, weight_decay=1e-4)
    g = gen(mu_a, sd_a, [200] * C)
    for _ in range(rounds):
        X, Y = g()
        for _ in range(epochs):
            p = torch.randperm(len(X))
            for i in range(0, len(p), bs):
                b = p[i:i+bs]; x = X[b].to(dev)
                if l2norm: x = F.normalize(x, dim=1) * 16
                l = F.cross_entropy(m(x), Y[b].to(dev)); o.zero_grad(); l.backward(); o.step()
    m.eval()
    with torch.no_grad():
        x = Xh.to(dev)
        if l2norm: x = F.normalize(x, dim=1) * 16
        pr = m(x).argmax(1).cpu()
    return round((pr == Yh).float().mean().item(), 4), np.bincount(pr.numpy(), minlength=C).tolist()
print("(g) HAM cloud MLP, bs4, 10ep x 3 rounds ", cloud_train(4, 10, 3))
print("(h) HAM cloud MLP, bs4, +L2-norm input  ", cloud_train(4, 10, 3, l2norm=True))
print("(i) HAM cloud MLP, bs64, 10ep x 3 rounds", cloud_train(64, 10, 3))
print("feature scale: mean |x| =", round(float(Xf.abs().mean()), 4), " mean ||x|| =", round(float(Xf.norm(dim=1).mean()), 2))
