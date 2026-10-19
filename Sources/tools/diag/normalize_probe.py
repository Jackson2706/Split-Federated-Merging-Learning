#!/usr/bin/env python3
"""PLAN-4 normalization factorial on PLAN-3's cached real features.

This is an offline, CPU-only diagnostic.  It never loads an encoder, dataset,
checkpoint, or federated-learning runner.  Transform and classifier statistics
are fit exclusively on the cached training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "outputs/diag/plan3_hsfp_baseline_oracle_seed20260714/real_feature_cache.pt"
BOUNDARIES = {"L1_client_post_gap": 128, "L2_edge_output": 256}
TRANSFORMS = (
    "raw",
    "global_mean_centered",
    "l2_normalized",
    "centered_l2",
    "centered_pca_whitened",
    "centered_whitened_l2",
)


@dataclass(frozen=True)
class HeadConfig:
    epochs: int
    samples_per_class: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    cosine_scale: float
    angular_margin: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_cache(payload: Mapping) -> None:
    if set(payload) != {"identity", "train", "eval"}:
        raise RuntimeError(f"unexpected cache keys: {sorted(payload)}")
    train, evaluate = payload["train"], payload["eval"]
    for split_name, split in (("train", train), ("eval", evaluate)):
        labels = split["labels"]
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise RuntimeError(f"invalid {split_name} labels")
        for boundary, dim in BOUNDARIES.items():
            features = split[boundary]
            if features.shape != (len(labels), dim):
                raise RuntimeError(f"invalid {split_name}/{boundary} shape: {tuple(features.shape)}")
            if not torch.isfinite(features).all():
                raise RuntimeError(f"non-finite cache values in {split_name}/{boundary}")
    if train["labels"].data_ptr() == evaluate["labels"].data_ptr():
        raise RuntimeError("train and eval labels alias")
    for key in ("client_source_ids", "edge_source_ids"):
        if train[key].shape != train["labels"].shape:
            raise RuntimeError(f"invalid training {key}")


class FittedTransform:
    """A transform whose state records and enforces its training-only fit."""

    def __init__(self, name: str, pca_components: int, eps: float = 1e-8):
        if name not in TRANSFORMS:
            raise ValueError(name)
        self.name = name
        self.pca_components = pca_components
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.projection: torch.Tensor | None = None
        self.fit_split: str | None = None
        self.fit_count: int | None = None
        self.fit_fingerprint: str | None = None

    @staticmethod
    def _fingerprint(x: torch.Tensor) -> str:
        summary = torch.stack((x.double().sum(), x.double().square().sum()))
        return hashlib.sha256(summary.numpy().tobytes() + str(tuple(x.shape)).encode()).hexdigest()

    def fit(self, x: torch.Tensor, *, split: str) -> "FittedTransform":
        if split != "train":
            raise AssertionError("normalization transforms may only be fit on TRAIN")
        if self.fit_split is not None:
            raise AssertionError("transform fit called more than once")
        self.fit_split = split
        self.fit_count = len(x)
        self.fit_fingerprint = self._fingerprint(x)
        if "centered" in self.name:
            self.mean = x.mean(0)
        if "whitened" in self.name:
            centered = x - self.mean
            covariance = centered.T.double().matmul(centered.double()) / max(len(x) - 1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            keep = min(self.pca_components, x.shape[1])
            order = torch.argsort(eigenvalues, descending=True)[:keep]
            values = eigenvalues[order].clamp_min(self.eps)
            vectors = eigenvectors[:, order]
            self.projection = (vectors / torch.sqrt(values)[None, :]).float()
        return self

    def apply(self, x: torch.Tensor, *, split: str) -> torch.Tensor:
        if self.fit_split != "train" or self.fit_count is None:
            raise AssertionError("unfitted or non-training-fitted transform")
        if split not in ("train", "eval"):
            raise ValueError(split)
        out = x
        if "centered" in self.name:
            out = out - self.mean
        if "whitened" in self.name:
            out = out @ self.projection
        if self.name in ("l2_normalized", "centered_l2", "centered_whitened_l2"):
            out = nn.functional.normalize(out, dim=1, eps=self.eps)
        if not torch.isfinite(out).all():
            raise RuntimeError(f"non-finite output from {self.name} on {split}")
        return out

    def audit(self) -> dict:
        return {
            "fit_split": self.fit_split,
            "fit_sample_count": self.fit_count,
            "fit_fingerprint": self.fit_fingerprint,
            "pca_components": None if self.projection is None else self.projection.shape[1],
        }


def class_statistics(features: torch.Tensor, labels: torch.Tensor):
    classes = torch.unique(labels, sorted=True)
    class_indices = torch.searchsorted(classes, labels)
    counts = torch.bincount(class_indices, minlength=len(classes)).to(features.dtype)
    sums = torch.zeros(len(classes), features.shape[1], dtype=features.dtype)
    sums.index_add_(0, class_indices, features)
    centroids = sums / counts[:, None]
    cosine = nn.functional.normalize(centroids, dim=1) @ nn.functional.normalize(centroids, dim=1).T
    off_diagonal = ~torch.eye(len(classes), dtype=torch.bool)
    global_mean = features.mean(0)
    within = (features - centroids[class_indices]).square().sum()
    between = (counts[:, None] * (centroids - global_mean).square()).sum()
    return classes, centroids, {
        "mean_centroid_pairwise_cosine": cosine[off_diagonal].mean().item(),
        "within_scatter": within.item(),
        "between_scatter": between.item(),
        "within_between_scatter_ratio": (within / between.clamp_min(torch.finfo(features.dtype).eps)).item(),
    }


@torch.inference_mode()
def nearest_centroid(train_x, train_y, eval_x, eval_y) -> float:
    classes, centroids, _ = class_statistics(train_x, train_y)
    predictions = classes[torch.cdist(eval_x, centroids).argmin(1)]
    return (predictions == eval_y).float().mean().item()


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


def prototype_samples(features, labels, source_ids, samples_per_class, seed):
    """Reproduce PLAN-3/current H-SFP unweighted source-prototype aggregation."""
    generator = torch.Generator().manual_seed(seed)
    classes = torch.unique(labels, sorted=True)
    sources = torch.unique(source_ids, sorted=True)
    class_indices = torch.searchsorted(classes, labels)
    source_indices = torch.searchsorted(sources, source_ids)
    group_indices = class_indices * len(sources) + source_indices
    group_count = len(classes) * len(sources)
    counts = torch.bincount(group_indices, minlength=group_count).to(features.dtype)
    sums = torch.zeros(group_count, features.shape[1], dtype=features.dtype)
    square_sums = torch.zeros_like(sums)
    sums.index_add_(0, group_indices, features)
    square_sums.index_add_(0, group_indices, features.square())
    present = counts > 0
    means_all = torch.zeros_like(sums)
    variances_all = torch.zeros_like(sums)
    means_all[present] = sums[present] / counts[present, None]
    variances_all[present] = (
        square_sums[present] / counts[present, None] - means_all[present].square()
    ).clamp_min(0)
    mean_x, gaussian_x, output_y = [], [], []
    for class_index, label in enumerate(classes):
        start = class_index * len(sources)
        stop = start + len(sources)
        valid = present[start:stop]
        means = means_all[start:stop][valid]
        stds = torch.sqrt(variances_all[start:stop][valid])
        aggregate_mean = means.mean(0)
        aggregate_std = torch.sqrt(stds.square().mean(0))
        noise = torch.randn(samples_per_class, features.shape[1], generator=generator)
        mean_x.append(aggregate_mean.repeat(samples_per_class, 1))
        gaussian_x.append(aggregate_mean + noise * aggregate_std)
        output_y.append(label.repeat(samples_per_class))
    y = torch.cat(output_y)
    return (torch.cat(mean_x), y), (torch.cat(gaussian_x), y)


class CosineMarginHead(nn.Module):
    def __init__(self, dim, classes, scale, margin):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dim))
        nn.init.normal_(self.weight, std=0.01)
        self.scale = scale
        self.margin = margin

    def forward(self, x, labels=None):
        cosine = nn.functional.normalize(x, dim=1) @ nn.functional.normalize(self.weight, dim=1).T
        if labels is not None and self.margin:
            cosine = cosine.clone()
            cosine[torch.arange(len(labels)), labels] -= self.margin
        return cosine * self.scale


class TrainStandardizedLinear(nn.Module):
    def __init__(self, dim, classes, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.linear = nn.Linear(dim, classes)

    def forward(self, x, labels=None):
        del labels
        return self.linear((x - self.mean) / self.std)


def train_head(model, train_x, train_y, eval_x, eval_y, *, epochs, batch_size, lr,
               weight_decay, seed) -> float:
    seed_everything(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, labels) if isinstance(model, CosineMarginHead) else model(features)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    correct = 0
    with torch.inference_mode():
        for start in range(0, len(eval_y), batch_size):
            stop = start + batch_size
            correct += int((model(eval_x[start:stop]).argmax(1) == eval_y[start:stop]).sum())
    return correct / len(eval_y)


def evaluate_transform(train_x, train_y, source_ids, eval_x, eval_y, config, seed):
    classes = int(torch.unique(train_y).numel())
    (mean_x, mean_y), (diag_x, diag_y) = prototype_samples(
        train_x, train_y, source_ids, config.samples_per_class, seed,
    )
    real_x, real_y = balanced_real(train_x, train_y, config.samples_per_class, seed)
    common = dict(epochs=config.epochs, batch_size=config.batch_size,
                  lr=config.learning_rate, weight_decay=config.weight_decay, seed=seed)
    mean_top1 = train_head(nn.Linear(train_x.shape[1], classes), mean_x, mean_y,
                           eval_x, eval_y, **common)
    diag_top1 = train_head(nn.Linear(train_x.shape[1], classes), diag_x, diag_y,
                           eval_x, eval_y, **common)
    cosine_top1 = train_head(
        CosineMarginHead(train_x.shape[1], classes, config.cosine_scale, config.angular_margin),
        real_x, real_y, eval_x, eval_y, **common,
    )
    full_linear = TrainStandardizedLinear(
        train_x.shape[1], classes, train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6),
    )
    linear_top1 = train_head(full_linear, train_x, train_y, eval_x, eval_y, **common)
    _, _, stats = class_statistics(train_x, train_y)
    return {
        "nearest_centroid_top1": nearest_centroid(train_x, train_y, eval_x, eval_y),
        "mean_only_prototype_head_top1": mean_top1,
        "diagonal_gaussian_prototype_head_top1": diag_top1,
        "cosine_angular_margin_head_top1": cosine_top1,
        "full_data_linear_probe_top1": linear_top1,
        **stats,
    }


def assert_finite(value, path="results") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite metric at {path}: {value}")


def write_markdown(path: Path, results: Mapping) -> None:
    lines = [
        "# PLAN-4 normalization factorial",
        "",
        "> Cached-feature, CPU-only diagnostic. Every transform was fit on TRAIN and applied to EVAL.",
        "",
        "| Boundary | Transform | NC | Mean proto | Diag proto | Cosine margin | Linear | Centroid cos | W/B |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for boundary, boundary_results in results["boundaries"].items():
        for name in TRANSFORMS:
            item = boundary_results[name]
            lines.append(
                f"| {boundary} | {name} | {item['nearest_centroid_top1']:.4f} | "
                f"{item['mean_only_prototype_head_top1']:.4f} | "
                f"{item['diagonal_gaussian_prototype_head_top1']:.4f} | "
                f"{item['cosine_angular_margin_head_top1']:.4f} | "
                f"{item['full_data_linear_probe_top1']:.4f} | "
                f"{item['mean_centroid_pairwise_cosine']:.4f} | "
                f"{item['within_between_scatter_ratio']:.4f} |"
            )
    lines += ["", "Unrounded metrics and leakage-audit metadata are in `normalize_results.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-scale", type=float, default=16.0)
    parser.add_argument("--angular-margin", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.cache.resolve()
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "normalize_results.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert_cache(payload)
    train, evaluate = payload["train"], payload["eval"]
    config = HeadConfig(args.epochs, args.samples_per_class, args.batch_size,
                        args.learning_rate, args.weight_decay, args.cosine_scale,
                        args.angular_margin)
    # A single immutable config object is passed to every transform.  This is
    # asserted again in the serialized contract for machine-readable auditing.
    config_snapshot = asdict(config)
    results = {
        "diagnostic_only": True,
        "cached_features_only": True,
        "device": "cpu",
        "seed": args.seed,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path), "identity": payload["identity"]},
        "split_contract": {
            "transform_fit_split": "train",
            "eval_used_for_fit": False,
            "train_samples": len(train["labels"]),
            "eval_samples": len(evaluate["labels"]),
        },
        "head_config": config_snapshot,
        "same_head_config_all_transforms": True,
        "pca_components": {"L1_client_post_gap": 64, "L2_edge_output": 128},
        "boundaries": {},
    }
    source_key = {"L1_client_post_gap": "client_source_ids", "L2_edge_output": "edge_source_ids"}
    for boundary in BOUNDARIES:
        boundary_results = {}
        for name in TRANSFORMS:
            assert asdict(config) == config_snapshot, "head config changed across transforms"
            transform = FittedTransform(name, results["pca_components"][boundary]).fit(
                train[boundary], split="train",
            )
            train_x = transform.apply(train[boundary], split="train")
            eval_x = transform.apply(evaluate[boundary], split="eval")
            if transform.fit_fingerprint != FittedTransform._fingerprint(train[boundary]):
                raise AssertionError("transform fit fingerprint is not the TRAIN feature fingerprint")
            item = evaluate_transform(
                train_x, train["labels"], train[source_key[boundary]],
                eval_x, evaluate["labels"], config, args.seed,
            )
            item["transform_audit"] = transform.audit()
            boundary_results[name] = item
        results["boundaries"][boundary] = boundary_results
    assert_finite(results)
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / "NORMALIZE_SUMMARY.md", results)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
