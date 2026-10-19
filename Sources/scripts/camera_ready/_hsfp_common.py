"""
Shared helpers for camera-ready standalone scripts (covariance / lstat / inversion).

Builds the H-SFP split models and dataset from a config file, and exposes a
client+edge feature extractor (client -> edge -> GAP -> flatten) matching the
H-SFP prototype-extraction pipeline (see models/pipeline.py).
"""

import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HSFP_DIR = os.path.join(ROOT, "classification", "H-SFP")


def _ensure_path():
    # Mirror how classification/H-SFP/runner.py resolves its packages: the method
    # dir on sys.path exposes the `config`, `data`, `models`, `hierarchy` packages.
    # ROOT exposes the top-level `camera_ready` package. Do NOT add the inner
    # data/ models/ subdirs (their inner data.py would shadow the `data` package).
    for p in (HSFP_DIR, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_config(cfg_path, overrides=None):
    _ensure_path()
    from config import ConfigLoader  # classification/H-SFP/config
    cfg = ConfigLoader(cfg_path).get_config()
    if overrides:
        cfg.update(overrides)
    return cfg


def build_models(cfg):
    """Return (client, edge, cloud) instantiated nn.Modules."""
    _ensure_path()
    from models import get_model
    Client, Edge, Cloud = get_model(cfg["model"], cfg["dataset"])
    return Client(), Edge(), Cloud(cfg)


def build_dataset(cfg):
    """Return (train_dataset, valid_dataset, test_dataset, user_groups)."""
    _ensure_path()
    from data import get_dataset
    return get_dataset(cfg)


class ClientExtractor(nn.Module):
    """Client feature extractor f(x) -> smashed activation [B, C, H, W]."""
    def __init__(self, client):
        super().__init__()
        self.client = client

    def forward(self, x):
        return self.client(x)


class EdgeExtractor(nn.Module):
    """client -> edge -> GAP -> flatten => pooled feature vector [B, D]."""
    def __init__(self, client, edge):
        super().__init__()
        self.client = client
        self.edge = edge
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.client(x)
        x = self.edge(x)
        if x.dim() == 4:
            x = torch.flatten(self.pool(x), 1)
        return x


def load_checkpoint_into(client, edge, cloud, ckpt_path):
    """Best-effort load of an H-SFP checkpoint (checkpoint_hfl.pt) into modules.

    The checkpoint stores a FullPipelineModel state_dict with keys prefixed
    'client.', 'edge.', 'cloud.'. Missing keys are ignored.
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return False
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    for mod, prefix in ((client, "client."), (edge, "edge."), (cloud, "cloud.")):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            mod.load_state_dict(sub, strict=False)
    return True


def gather_features(extractor, dataset, indices, device, batch_size=128, max_n=None):
    """Run extractor over dataset[indices]; return (features [N,D], labels [N])."""
    from torch.utils.data import DataLoader, Subset
    if max_n is not None:
        indices = list(indices)[:max_n]
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size,
                        shuffle=False)
    extractor = extractor.to(device).eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = extractor(x)
            feats.append(f.reshape(f.shape[0], -1).cpu())
            labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)
