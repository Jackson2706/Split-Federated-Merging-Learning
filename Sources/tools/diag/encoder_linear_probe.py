#!/usr/bin/env python3
"""Measure client-encoder linear-probe top-1 on real H-SFP features.

Diagnostic only: real activations are materialized offline and are never part
of H-SFP training or communication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import oracle_probe as oracle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--linear-epochs", type=int, default=10)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    result_path = args.output_dir / "encoder_linear_probe.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic: {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oracle.seed_everything(args.seed)
    device = oracle.resolve_device(args.device)
    config = oracle.ConfigLoader(str(args.config)).get_config()
    config["seed"] = args.seed
    config["is_gpu"] = device.type == "cuda"
    client, _edge, _cloud, checkpoint_meta = oracle.load_models(
        config, args.checkpoint, device
    )
    train_dataset, eval_dataset, _test_dataset, _user_groups = oracle.get_dataset(config)
    identity_edge = torch.nn.Identity()
    train = oracle.extract_features(
        train_dataset, client, identity_edge, device, args.batch_size, args.workers
    )
    evaluate = oracle.extract_features(
        eval_dataset, client, identity_edge, device, args.batch_size, args.workers
    )
    train_x, train_y = train["L1_client_post_gap"], train["labels"]
    eval_x, eval_y = evaluate["L1_client_post_gap"], evaluate["labels"]
    mean = train_x.mean(0)
    std = train_x.std(0, unbiased=False).clamp_min(1e-6)
    probe = oracle.StandardizedLinear(
        train_x.shape[1], int(config["num_classes"]), mean, std
    )
    top1 = oracle.train_and_evaluate(
        probe, train_x, train_y, eval_x, eval_y, device,
        args.linear_epochs, args.batch_size, args.linear_lr, args.seed,
    )
    result = {
        "diagnostic_only": True,
        "violates_no_activation_constraint": True,
        "boundary": "L1_client_post_gap",
        "client_encoder_linear_probe_top1": top1,
        "train_samples": int(train_y.numel()),
        "eval_samples": int(eval_y.numel()),
        "seed": args.seed,
        "device": str(device),
        "config": str(args.config),
        "checkpoint": checkpoint_meta,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"client_encoder_linear_probe_top1={top1:.6f}")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
