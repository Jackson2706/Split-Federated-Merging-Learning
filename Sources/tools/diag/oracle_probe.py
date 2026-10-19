#!/usr/bin/env python3
"""PLAN-3 representation oracle for H-SFP (diagnostic only).

This tool intentionally caches real activations and therefore intentionally
violates H-SFP's no-activation constraint.  It is an offline diagnostic, never
part of the proposed method or its training pipeline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
HSFP_ROOT = ROOT / "classification" / "H-SFP"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HSFP_ROOT))

from config import ConfigLoader  # noqa: E402
from data import get_dataset  # noqa: E402
from models import get_model  # noqa: E402


BOUNDARY_DIMS = {"L1_client_post_gap": 128, "L2_edge_output": 256}
SOURCE_NAMES = ("real", "mean_only", "diagonal_gaussian", "source_prototype_mixture")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models(config: Mapping, checkpoint_path: Path | None, device: torch.device):
    client_cls, edge_cls, cloud_cls = get_model(config["model"], config["dataset"])
    client, edge, cloud = client_cls(), edge_cls(), cloud_cls(config)
    checkpoint_meta = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model_state_dict"]
        for prefix, model in (("client.", client), ("edge.", edge), ("cloud.", cloud)):
            tier_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            model.load_state_dict(tier_state, strict=True)
        checkpoint_meta = {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "best_f1": float(checkpoint["best_f1"]),
            "resolved_config_hash": checkpoint.get("resolved_config_hash"),
        }
    return client.to(device).eval(), edge.to(device).eval(), cloud.to(device).eval(), checkpoint_meta


def source_ids_from_partition(size: int, user_groups: Mapping[int, Iterable[int]]) -> torch.Tensor:
    source_ids = torch.full((size,), -1, dtype=torch.long)
    for source_id, indices in sorted(user_groups.items()):
        idx = torch.as_tensor(np.asarray(list(indices), dtype=np.int64))
        if (source_ids[idx] >= 0).any():
            raise RuntimeError("proxy partition contains overlapping user indices")
        source_ids[idx] = int(source_id)
    if (source_ids < 0).any():
        raise RuntimeError("proxy partition does not cover the complete training set")
    return source_ids


def edge_ids_from_clients(client_source_ids: torch.Tensor, config: Mapping) -> torch.Tensor:
    """Reproduce HierarchicalFL._build_hierarchy's seeded client→edge map."""
    num_clients = int(config["num_users"])
    num_edges = int(config["mid_server"][0])
    provided = config.get("_client_to_edge")
    if provided is not None:
        mapping = torch.as_tensor(provided, dtype=torch.long)
    else:
        mapping = torch.empty(num_clients, dtype=torch.long)
        remaining = list(range(num_clients))
        clients_per_edge = num_clients // num_edges
        for edge_id in range(num_edges):
            assigned = (
                remaining
                if edge_id == num_edges - 1
                else list(np.random.choice(remaining, clients_per_edge, replace=False))
            )
            mapping[torch.as_tensor(assigned, dtype=torch.long)] = edge_id
            remaining = list(set(remaining) - set(assigned))
    return mapping[client_source_ids]


@torch.inference_mode()
def extract_features(dataset, client, edge, device, batch_size, workers, limit=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    l1_parts, l2_parts, label_parts = [], [], []
    seen = 0
    for images, labels in loader:
        if limit is not None and seen >= limit:
            break
        if limit is not None and seen + images.shape[0] > limit:
            keep = limit - seen
            images, labels = images[:keep], labels[:keep]
        l1_4d = client(images.to(device))
        l2_4d = edge(l1_4d)
        l1_parts.append(l1_4d.flatten(1).float().cpu())
        l2_parts.append(l2_4d.flatten(1).float().cpu())
        label_parts.append(labels.long().cpu())
        seen += images.shape[0]
    return {
        "L1_client_post_gap": torch.cat(l1_parts),
        "L2_edge_output": torch.cat(l2_parts),
        "labels": torch.cat(label_parts),
    }


def make_smoke_features(client, edge, device, seed, classes=4):
    generator = torch.Generator().manual_seed(seed)
    train_y = torch.arange(classes).repeat_interleave(8)
    eval_y = torch.arange(classes).repeat_interleave(4)
    train_x = torch.randn(len(train_y), 3, 32, 32, generator=generator)
    eval_x = torch.randn(len(eval_y), 3, 32, 32, generator=generator)
    # Add a deterministic class signal so every metric remains meaningful.
    train_x[:, 0] += train_y[:, None, None].float() / classes
    eval_x[:, 0] += eval_y[:, None, None].float() / classes
    train = extract_features(TensorDataset(train_x, train_y), client, edge, device, 16, 0)
    evaluate = extract_features(TensorDataset(eval_x, eval_y), client, edge, device, 16, 0)
    train["client_source_ids"] = torch.arange(len(train_y)) % 4
    train["edge_source_ids"] = train["client_source_ids"] % 2
    return train, evaluate


def load_or_create_cache(args, config, client, edge, device, checkpoint_meta):
    cache_path = args.output_dir / "real_feature_cache.pt"
    identity = {
        "seed": args.seed,
        "checkpoint_sha256": None if checkpoint_meta is None else checkpoint_meta["sha256"],
        "boundary_dims": BOUNDARY_DIMS,
    }
    if cache_path.exists() and not args.smoke:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload["identity"] != identity:
            raise RuntimeError(f"cache identity mismatch: {cache_path}")
        return payload["train"], payload["eval"], cache_path, True

    if args.smoke:
        train, evaluate = make_smoke_features(client, edge, device, args.seed, args.smoke_classes)
        return train, evaluate, None, False

    train_dataset, valid_dataset, _test_dataset, user_groups = get_dataset(config)
    train = extract_features(train_dataset, client, edge, device, args.batch_size, args.workers)
    evaluate = extract_features(valid_dataset, client, edge, device, args.batch_size, args.workers)
    train["client_source_ids"] = source_ids_from_partition(len(train_dataset), user_groups)
    # get_dataset has consumed NumPy RNG exactly as in the proxy runner; the
    # hierarchy assignment is the next NumPy operation in both paths.
    train["edge_source_ids"] = edge_ids_from_clients(train["client_source_ids"], config)
    payload = {"identity": identity, "train": train, "eval": evaluate}
    torch.save(payload, cache_path)
    return train, evaluate, cache_path, False


def assert_cache(train, evaluate) -> None:
    for split in (train, evaluate):
        count = split["labels"].shape[0]
        for boundary, dim in BOUNDARY_DIMS.items():
            tensor = split[boundary]
            if tensor.shape != (count, dim):
                raise RuntimeError(f"{boundary} shape {tuple(tensor.shape)} != {(count, dim)}")
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite cached feature at {boundary}")
    for key in ("client_source_ids", "edge_source_ids"):
        if train[key].shape != train["labels"].shape:
            raise RuntimeError(f"{key} and training labels have different shapes")


def class_statistics(features: torch.Tensor, labels: torch.Tensor):
    classes = torch.unique(labels, sorted=True)
    centroids = torch.stack([features[labels == label].mean(0) for label in classes])
    normalized = nn.functional.normalize(centroids, dim=1)
    cosine = normalized @ normalized.T
    if len(classes) > 1:
        mean_pairwise_cosine = cosine[~torch.eye(len(classes), dtype=torch.bool)].mean().item()
    else:
        mean_pairwise_cosine = 0.0
    global_mean = features.mean(0)
    within = sum(((features[labels == label] - centroids[i]) ** 2).sum() for i, label in enumerate(classes))
    between = sum((labels == label).sum() * ((centroids[i] - global_mean) ** 2).sum()
                  for i, label in enumerate(classes))
    scatter_ratio = (within / between.clamp_min(torch.finfo(features.dtype).eps)).item()
    return classes, centroids, {
        "mean_centroid_pairwise_cosine": mean_pairwise_cosine,
        "mean_feature_l2_norm": features.norm(dim=1).mean().item(),
        "within_between_class_scatter_ratio": scatter_ratio,
    }


@torch.inference_mode()
def nearest_centroid_accuracy(train_x, train_y, eval_x, eval_y):
    classes, centroids, _ = class_statistics(train_x, train_y)
    # Squared Euclidean nearest centroid, evaluated only on real features.
    predictions = classes[torch.cdist(eval_x, centroids).argmin(1)]
    return (predictions == eval_y).float().mean().item()


class StandardizedLinear(nn.Module):
    def __init__(self, dim, num_classes, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.linear((x - self.mean) / self.std)


class L1DownstreamHead(nn.Module):
    def __init__(self, edge, cloud):
        super().__init__()
        self.edge = edge
        self.cloud = cloud

    def forward(self, x):
        return self.cloud(self.edge(x[:, :, None, None]))


def train_and_evaluate(model, train_x, train_y, eval_x, eval_y, device, epochs, batch_size, lr, seed):
    seed_everything(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=batch_size, shuffle=False)
        for features, labels in eval_loader:
            prediction = model(features.to(device)).argmax(1).cpu()
            correct += int((prediction == labels).sum())
            total += labels.numel()
    return correct / total


def balanced_real(features, labels, samples_per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for label in torch.unique(labels, sorted=True):
        indices = torch.where(labels == label)[0]
        if len(indices) >= samples_per_class:
            indices = indices[torch.randperm(len(indices), generator=generator)[:samples_per_class]]
        else:
            indices = indices[torch.randint(len(indices), (samples_per_class,), generator=generator)]
        xs.append(features[indices])
        ys.append(labels[indices])
    return torch.cat(xs), torch.cat(ys)


def synthetic_sources(features, labels, source_ids, samples_per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    mode_x = {name: [] for name in SOURCE_NAMES[1:]}
    mode_y = {name: [] for name in SOURCE_NAMES[1:]}
    for label in torch.unique(labels, sorted=True):
        class_mask = labels == label
        source_stats = []
        for source in torch.unique(source_ids[class_mask], sorted=True):
            component = features[class_mask & (source_ids == source)]
            mean = component.mean(0)
            std = component.std(0, unbiased=False)
            source_stats.append((mean, std))
        means = torch.stack([item[0] for item in source_stats])
        stds = torch.stack([item[1] for item in source_stats])
        # Match current H-SFP aggregation: unweighted source-prototype mean and
        # sqrt(mean(source variance)), followed by diagonal Gaussian sampling.
        aggregate_mean = means.mean(0)
        aggregate_std = torch.sqrt((stds.square()).mean(0))
        eps = torch.randn(samples_per_class, features.shape[1], generator=generator)
        mode_x["mean_only"].append(aggregate_mean.repeat(samples_per_class, 1))
        mode_x["diagonal_gaussian"].append(aggregate_mean + eps * aggregate_std)
        component_ids = torch.arange(samples_per_class) % len(source_stats)
        component_ids = component_ids[torch.randperm(samples_per_class, generator=generator)]
        mix_eps = torch.randn(samples_per_class, features.shape[1], generator=generator)
        mode_x["source_prototype_mixture"].append(means[component_ids] + mix_eps * stds[component_ids])
        for name in SOURCE_NAMES[1:]:
            mode_y[name].append(label.repeat(samples_per_class))
    for name in SOURCE_NAMES[1:]:
        outputs[name] = torch.cat(mode_x[name]), torch.cat(mode_y[name])
    return outputs


def boundary_diagnostics(name, train_x, train_y, source_ids, eval_x, eval_y, edge, cloud, config, args, device):
    classes, _centroids, stats = class_statistics(train_x, train_y)
    num_classes = int(config["num_classes"])
    mean = train_x.mean(0)
    std = train_x.std(0, unbiased=False).clamp_min(1e-6)
    linear = StandardizedLinear(train_x.shape[1], num_classes, mean, std)
    linear_top1 = train_and_evaluate(linear, train_x, train_y, eval_x, eval_y, device,
                                     args.linear_epochs, args.batch_size, args.linear_lr, args.seed)
    real_x, real_y = balanced_real(train_x, train_y, args.samples_per_class, args.seed)
    sources = {"real": (real_x, real_y)}
    sources.update(synthetic_sources(train_x, train_y, source_ids, args.samples_per_class, args.seed))
    if name == "L1_client_post_gap":
        template = L1DownstreamHead(copy.deepcopy(edge), copy.deepcopy(cloud))
    else:
        template = copy.deepcopy(cloud)
    downstream = {}
    for source_name in SOURCE_NAMES:
        source_x, source_y = sources[source_name]
        downstream[source_name] = train_and_evaluate(
            copy.deepcopy(template), source_x, source_y, eval_x, eval_y, device,
            args.epochs, args.batch_size, args.lr, args.seed,
        )
    return {
        "shape": {"train": list(train_x.shape), "eval": list(eval_x.shape)},
        "classes_observed": int(len(classes)),
        "linear_probe_real_top1": linear_top1,
        "nearest_centroid_real_top1": nearest_centroid_accuracy(train_x, train_y, eval_x, eval_y),
        **stats,
        "downstream_head_top1_on_real_eval": downstream,
        "downstream_train_samples_per_class": args.samples_per_class,
    }


def assert_finite(value, path="results"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite metric at {path}: {value}")


def write_summary(path: Path, results: Mapping) -> None:
    lines = [
        "# PLAN-3 H-SFP representation oracle",
        "",
        "> Diagnostic only: this intentionally caches real activations and violates the no-activation constraint; it is not the proposed method.",
        "",
        f"Device: `{results['device']}`. Evaluation caveat: official CIFAR-100 test is reused as validation by the existing loader.",
        "",
        "| Boundary | Linear real | Nearest centroid | Real→real | Mean-only | Diag Gaussian | Source mixture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for boundary, item in results["boundaries"].items():
        head = item["downstream_head_top1_on_real_eval"]
        lines.append(
            f"| {boundary} | {item['linear_probe_real_top1']:.4f} | "
            f"{item['nearest_centroid_real_top1']:.4f} | {head['real']:.4f} | "
            f"{head['mean_only']:.4f} | {head['diagonal_gaussian']:.4f} | "
            f"{head['source_prototype_mixture']:.4f} |"
        )
    lines += ["", "Unrounded values and embedding statistics are in `diag_results.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/proxy/plan2_cifar100_hsfp_baseline.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--linear-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--linear-lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic-input CPU smoke; skips CIFAR/cache/checkpoint requirement")
    parser.add_argument("--smoke-classes", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if not args.smoke and args.checkpoint is None:
        raise SystemExit("--checkpoint is required outside --smoke mode")
    result_path = args.output_dir / "diag_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic: {result_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = resolve_device("cpu" if args.smoke else args.device)
    config = ConfigLoader(str(args.config)).get_config()
    config["seed"] = args.seed
    config["is_gpu"] = device.type == "cuda"
    client, edge, cloud, checkpoint_meta = load_models(config, args.checkpoint, device)
    train, evaluate, cache_path, cache_reused = load_or_create_cache(
        args, config, client, edge, device, checkpoint_meta,
    )
    assert_cache(train, evaluate)
    results = {
        "diagnostic_only": True,
        "violates_no_activation_constraint": True,
        "test_as_validation": True,
        "seed": args.seed,
        "device": str(device),
        "config": str(args.config),
        "checkpoint": checkpoint_meta,
        "cache": None if cache_path is None else {"path": str(cache_path), "reused": cache_reused},
        "boundary_contract": BOUNDARY_DIMS,
        "feature_source_contract": {
            "same_head_initialization_optimizer_epochs_eval": True,
            "balanced_training_samples_per_class": args.samples_per_class,
            "all_heads_evaluated_on_real_features": True,
        },
        "boundaries": {},
    }
    source_key = {
        "L1_client_post_gap": "client_source_ids",
        "L2_edge_output": "edge_source_ids",
    }
    for boundary in BOUNDARY_DIMS:
        results["boundaries"][boundary] = boundary_diagnostics(
            boundary, train[boundary], train["labels"], train[source_key[boundary]],
            evaluate[boundary], evaluate["labels"], edge, cloud, config, args, device,
        )
    assert_finite(results)
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(args.output_dir / "SUMMARY.md", results)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
