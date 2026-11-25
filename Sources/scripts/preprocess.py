#!/usr/bin/env python3
"""Decode/resize a classification training set once into a tensor cache for --gpu-data.

The cache holds exactly what the dataset's own (deterministic) transform yields
-- same resize, interpolation and normalization as the training pipeline --
so `--gpu-data 1 --set gpu_data_cache=<file>` skips JPEG decoding at run time.

    python scripts/preprocess.py --cfg configs/revision/ham10000_reference.yaml \
        --out data/cache/ham10000_train.pt
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "classification", "H-SFP")]

from config.config_loader import ConfigLoader  # noqa: E402
from data import get_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="only affects the (unused) client split")
    a = ap.parse_args()
    cfg = ConfigLoader(a.cfg).get_config()
    cfg["seed"] = a.seed
    train, _, _, _ = get_dataset(cfg)
    tf = repr(getattr(train, "transform", ""))
    if "Random" in tf:
        sys.exit(f"refusing: training transform is random ({tf})")
    X = torch.empty(0)
    Y = torch.empty(len(train), dtype=torch.long)
    for i in range(len(train)):
        x, y = train[i]
        if i == 0:
            X = torch.empty((len(train), *x.shape), dtype=x.dtype)
        X[i], Y[i] = x, int(y)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"X": X, "Y": Y, "cfg": a.cfg}, a.out)
    print(f"wrote {a.out}: X{tuple(X.shape)} {X.dtype}")


if __name__ == "__main__":
    main()
