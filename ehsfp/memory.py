"""
Episodic Prototype Memory for E-HSFP.

Stores historical class-wise prototype records and supports replay-based
mixing of current and past prototypes for training stability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class PrototypeRecord:
    """A single prototype entry in the episodic memory."""
    class_id: int
    mu: torch.Tensor            # class-wise mean [feature_dim] or [C, H, W]
    sigma: torch.Tensor         # class-wise std  [feature_dim] or [C, H, W]
    support_count: int          # number of samples used to compute this proto
    source_id: str              # e.g. "client_3", "edge_1"
    round_idx: int              # training round when this was created
    age: int = 0                # incremented each round
    reliability: float = 1.0    # reliability score in [0, 1]
    episode_id: Optional[str] = None  # serverless episode identifier


class EpisodicPrototypeMemory:
    """Fixed-size memory bank for historical prototype records.

    Supports add/update, retrieval by class/age/reliability, and
    age-based eviction.
    """

    def __init__(self, max_size: int = 500, max_age: int = 20):
        self.max_size = max_size
        self.max_age = max_age
        self._records: List[PrototypeRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: PrototypeRecord) -> None:
        """Add a prototype record. Evicts oldest if at capacity."""
        self._records.append(record)
        if len(self._records) > self.max_size:
            # Evict oldest (highest age) records first
            self._records.sort(key=lambda r: r.age)
            self._records = self._records[:self.max_size]

    def add_from_proto_dicts(
        self,
        proto_dict: Dict[int, torch.Tensor],
        dist_dict: Dict[int, torch.Tensor],
        source_id: str,
        round_idx: int,
        support_counts: Optional[Dict[int, int]] = None,
        episode_id: Optional[str] = None,
    ) -> None:
        """Convenience: add all entries from proto/dist dicts at once."""
        for cls_id, mu in proto_dict.items():
            sigma = dist_dict.get(cls_id)
            if sigma is None:
                continue
            n = support_counts.get(cls_id, 0) if support_counts else 0
            self.add(PrototypeRecord(
                class_id=cls_id,
                mu=mu.detach().cpu(),
                sigma=sigma.detach().cpu(),
                support_count=n,
                source_id=source_id,
                round_idx=round_idx,
                age=0,
                episode_id=episode_id,
            ))

    def get_by_class(self, class_id: int) -> List[PrototypeRecord]:
        return [r for r in self._records if r.class_id == class_id]

    def get_recent(self, max_age: Optional[int] = None) -> List[PrototypeRecord]:
        cutoff = max_age if max_age is not None else self.max_age
        return [r for r in self._records if r.age <= cutoff]

    def get_top_k_reliable(self, class_id: int, k: int) -> List[PrototypeRecord]:
        cls_records = self.get_by_class(class_id)
        cls_records.sort(key=lambda r: r.reliability, reverse=True)
        return cls_records[:k]

    def get_balanced(self, k_per_class: int) -> Dict[int, List[PrototypeRecord]]:
        """Retrieve up to k_per_class records per class, preferring reliable ones."""
        result: Dict[int, List[PrototypeRecord]] = {}
        classes = set(r.class_id for r in self._records)
        for cls_id in classes:
            result[cls_id] = self.get_top_k_reliable(cls_id, k_per_class)
        return result

    def age_all(self) -> None:
        """Increment age of all records and evict expired ones."""
        for r in self._records:
            r.age += 1
        self._records = [r for r in self._records if r.age <= self.max_age]

    def to_proto_dist_dicts(
        self, records: Optional[List[PrototypeRecord]] = None
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Convert a list of records to the (proto_dict, dist_dict) format
        used by the existing aggregation pipeline.

        When multiple records exist for the same class, averages them
        (weighted by reliability).
        """
        if records is None:
            records = self._records
        if not records:
            return {}, {}

        # Group by class
        by_class: Dict[int, List[PrototypeRecord]] = {}
        for r in records:
            by_class.setdefault(r.class_id, []).append(r)

        proto_dict: Dict[int, torch.Tensor] = {}
        dist_dict: Dict[int, torch.Tensor] = {}

        for cls_id, recs in by_class.items():
            weights = torch.tensor([r.reliability for r in recs])
            w_sum = weights.sum()
            if w_sum < 1e-8:
                weights = torch.ones(len(recs))
                w_sum = weights.sum()
            weights = weights / w_sum

            mus = torch.stack([r.mu for r in recs])
            sigmas = torch.stack([r.sigma for r in recs])

            # Weighted mean
            w = weights.view(-1, *([1] * (mus.dim() - 1)))
            proto_dict[cls_id] = (w * mus).sum(0)
            # Weighted variance: sum(w_i * (sigma_i^2 + (mu_i - mu_agg)^2))
            mu_agg = proto_dict[cls_id]
            var_agg = (w * (sigmas ** 2 + (mus - mu_agg.unsqueeze(0)) ** 2)).sum(0)
            dist_dict[cls_id] = torch.sqrt(var_agg + 1e-8)

        return proto_dict, dist_dict

    def clear(self) -> None:
        self._records.clear()


def mix_current_and_memory(
    current_proto_dict: Dict[int, torch.Tensor],
    current_dist_dict: Dict[int, torch.Tensor],
    memory: EpisodicPrototypeMemory,
    alpha: float = 0.7,
    top_k: int = 5,
    runtime_counters=None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Mix current-round prototypes with memory prototypes.

    Q_mix = alpha * Q_current + (1 - alpha) * Q_memory

    Returns merged (proto_dict, dist_dict) in the same format as the input.
    """
    if len(memory) == 0 or alpha >= 1.0:
        return current_proto_dict, current_dist_dict

    if runtime_counters is not None:
        runtime_counters.increment("memory.replay_calls")

    all_classes = set(current_proto_dict.keys())
    # Also include memory-only classes
    for r in memory.get_recent():
        all_classes.add(r.class_id)

    mixed_proto: Dict[int, torch.Tensor] = {}
    mixed_dist: Dict[int, torch.Tensor] = {}

    for cls_id in all_classes:
        has_current = cls_id in current_proto_dict
        mem_records = memory.get_top_k_reliable(cls_id, top_k)
        if runtime_counters is not None:
            runtime_counters.increment("memory.reads", len(mem_records))

        if has_current and mem_records:
            compute_device = torch.device(device) if device is not None else current_proto_dict[cls_id].device
            cur_mu = current_proto_dict[cls_id].to(compute_device)
            cur_sigma = current_dist_dict[cls_id].to(compute_device)
            mem_p, mem_d = memory.to_proto_dist_dicts(mem_records)
            mem_mu = mem_p[cls_id].to(compute_device)
            mem_sigma = mem_d[cls_id].to(compute_device)

            mixed_proto[cls_id] = alpha * cur_mu + (1 - alpha) * mem_mu
            # Mix variance: alpha * var_cur + (1-alpha) * var_mem + alpha*(1-alpha)*(mu_cur - mu_mem)^2
            var_mixed = (
                alpha * cur_sigma ** 2
                + (1 - alpha) * mem_sigma ** 2
                + alpha * (1 - alpha) * (cur_mu - mem_mu) ** 2
            )
            mixed_dist[cls_id] = torch.sqrt(var_mixed + 1e-8)
            if runtime_counters is not None:
                runtime_counters.increment("memory.changed_classes", int(not torch.equal(cur_mu, mixed_proto[cls_id])))
                runtime_counters.increment("memory.replays", len(mem_records))
        elif has_current:
            compute_device = torch.device(device) if device is not None else current_proto_dict[cls_id].device
            mixed_proto[cls_id] = current_proto_dict[cls_id].to(compute_device)
            mixed_dist[cls_id] = current_dist_dict[cls_id].to(compute_device)
        elif mem_records:
            mem_p, mem_d = memory.to_proto_dist_dicts(mem_records)
            compute_device = torch.device(device) if device is not None else mem_p[cls_id].device
            mixed_proto[cls_id] = mem_p[cls_id].to(compute_device)
            mixed_dist[cls_id] = mem_d[cls_id].to(compute_device)
            if runtime_counters is not None:
                runtime_counters.increment("memory.replays", len(mem_records))

    return mixed_proto, mixed_dist
