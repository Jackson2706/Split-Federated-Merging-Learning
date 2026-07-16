"""
Reliability-weighted prototype aggregation for E-HSFP.

Drop-in replacement for the simple mean aggregation in
`_aggregate_prototypes_and_generate_data`. When no reliability network
is provided, falls back to the original simple-mean behavior.
"""

import torch
from typing import Dict, Optional, Tuple, List
from ehsfp.memory import PrototypeRecord, EpisodicPrototypeMemory
from ehsfp.reliability import (
    PrototypeReliabilityNetwork,
    build_reliability_features,
    compute_heuristic_reliability,
)


def aggregate_proto_dicts(
    input_outputs: Dict,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Collapse source outputs into class-keyed mean/std dictionaries."""
    grouped: Dict[int, Dict[str, list]] = {}
    for outputs in input_outputs.values():
        if outputs is None:
            continue
        protos, stds = outputs
        for class_id, proto in protos.items():
            grouped.setdefault(class_id, {"p": [], "d": []})
            grouped[class_id]["p"].append(proto)
            grouped[class_id]["d"].append(stds[class_id])
    merged_p, merged_d = {}, {}
    for class_id, values in grouped.items():
        compute_device = torch.device(device) if device is not None else values["p"][0].device
        protos = [proto.to(compute_device) for proto in values["p"]]
        stds = [std.to(compute_device) for std in values["d"]]
        merged_p[class_id] = torch.stack(protos).mean(0)
        merged_d[class_id] = torch.sqrt(torch.stack([d ** 2 for d in stds]).mean(0))
    return merged_p, merged_d


def reliability_weighted_aggregate(
    input_outputs: Dict,
    num_samples_per_class: int,
    device: torch.device,
    reliability_net: Optional[PrototypeReliabilityNetwork] = None,
    memory: Optional[EpisodicPrototypeMemory] = None,
    generator=None,
    support_map: Optional[Dict] = None,
    runtime_counters=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate prototypes with optional reliability weighting and generate synthetic data.

    When reliability_net is None: identical to the original simple-mean aggregation.
    When provided: uses learned weights for weighted mean/variance.

    Args:
        input_outputs: {source_id: (proto_dict, dist_dict)} or None values.
        num_samples_per_class: synthetic samples per class.
        device: target torch device.
        reliability_net: optional learned reliability network.
        memory: optional episodic memory (for building reliability features).
        generator: optional ResidualPrototypeGenerator for enhanced synthesis.

    Returns:
        (features, labels) tensors.
    """
    if not input_outputs:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    # Group by label
    merged: Dict[int, Dict[str, list]] = {}
    for sid, outputs in input_outputs.items():
        if outputs is None:
            continue
        proto_dict, dist_dict = outputs
        for label, proto in proto_dict.items():
            if label not in merged:
                merged[label] = {"p": [], "d": [], "sids": []}
            merged[label]["p"].append(proto)
            merged[label]["d"].append(dist_dict[label])
            merged[label]["sids"].append(sid)

    if not merged:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    final_labels = sorted(merged.keys())

    if reliability_net is None:
        # Original simple-mean aggregation (baseline behavior)
        final_protos = torch.stack([
            torch.stack([p.to(device) for p in merged[l]["p"]]).mean(0)
            for l in final_labels
        ])
        final_dists = torch.stack([
            torch.sqrt(torch.stack([d.to(device) ** 2 for d in merged[l]["d"]]).mean(0))
            for l in final_labels
        ])
    else:
        # Learnable reliability-weighted aggregation
        if runtime_counters is not None:
            runtime_counters.increment("reliability.aggregation_calls")
        proto_list = []
        dist_list = []
        for l in final_labels:
            mus = torch.stack([mu.to(device) for mu in merged[l]["p"]])
            sigmas = torch.stack([sigma.to(device) for sigma in merged[l]["d"]])

            # Build PrototypeRecord-like objects for feature extraction.
            # Use the real per-source support counts when provided so the
            # reliability features match what the network saw during bootstrap.
            records = []
            for i, sid in enumerate(merged[l]["sids"]):
                sc = 0
                if support_map is not None:
                    sc = support_map.get(sid, {}).get(l, 0)
                records.append(PrototypeRecord(
                    class_id=l,
                    mu=merged[l]["p"][i],
                    sigma=merged[l]["d"][i],
                    support_count=sc,
                    source_id=str(sid),
                    round_idx=0,
                    age=0,
                ))

            # Class center for distance computation
            class_center = mus.mean(0)
            rel_feats = build_reliability_features(records, class_center, device=device)
            weights = reliability_net(rel_feats)  # [N]
            if runtime_counters is not None and weights.numel() > 1:
                normalized_weights = weights / weights.sum().clamp(min=1e-8)
                runtime_counters.increment(
                    "reliability.nonuniform_weight_calls",
                    (~torch.isclose(weights, weights[0])).any().to(torch.int64).detach(),
                )
                runtime_counters.increment(
                    "reliability.weight_variance_sum",
                    normalized_weights.var(unbiased=False).detach(),
                )
                runtime_counters.increment("reliability.weight_variance_observations")
            w_sum = weights.sum().clamp(min=1e-8)
            w = (weights / w_sum).view(-1, *([1] * (mus.dim() - 1)))

            # Weighted mean: mu_class = sum(w_i * mu_i)
            mu_agg = (w * mus).sum(0)
            # Weighted variance: sum(w_i * (sigma_i^2 + (mu_i - mu_agg)^2))
            var_agg = (w * (sigmas ** 2 + (mus - mu_agg.unsqueeze(0)) ** 2)).sum(0)
            sigma_agg = torch.sqrt(var_agg + 1e-8)

            proto_list.append(mu_agg)
            dist_list.append(sigma_agg)

        final_protos = torch.stack(proto_list)
        final_dists = torch.stack(dist_list)

    del merged

    # Generate synthetic data
    if generator is not None:
        features = generator.generate(final_protos, final_dists, num_samples_per_class)
    else:
        features = _generate_synthetic_data(final_protos, final_dists, num_samples_per_class)

    labels = torch.tensor(final_labels, device=device).repeat_interleave(num_samples_per_class)
    return features, labels


@torch.no_grad()
def _generate_synthetic_data(prototypes, distributions_std, num_samples_per_class):
    """Standard synthetic data generation (same as original)."""
    num_classes = prototypes.shape[0]
    shape = prototypes.shape[1:]
    device = prototypes.device
    means = prototypes.unsqueeze(1)
    stds = distributions_std.unsqueeze(1)
    epsilon = torch.randn(
        num_classes, num_samples_per_class, *shape,
        device=device, dtype=prototypes.dtype,
    )
    epsilon.mul_(stds).add_(means)
    return epsilon.flatten(0, 1)


def train_reliability_bootstrap(
    reliability_net: PrototypeReliabilityNetwork,
    optimizer: torch.optim.Optimizer,
    memory: EpisodicPrototypeMemory,
    device: torch.device,
) -> float:
    """One step of bootstrap training using heuristic reliability targets.

    Uses MSE loss between predicted and heuristic reliability scores.
    Returns the training loss value.
    """
    classes = set(r.class_id for r in memory.get_recent())
    if not classes:
        return 0.0

    total_loss = 0.0
    count = 0
    reliability_net.train()

    for cls_id in classes:
        records = memory.get_by_class(cls_id)
        if len(records) < 2:
            continue

        class_center = torch.stack([r.mu.to(device) for r in records]).mean(0)
        feats = build_reliability_features(records, class_center, device=device)
        targets = compute_heuristic_reliability(records, device=device)

        preds = reliability_net(feats)
        loss = torch.nn.functional.mse_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)
