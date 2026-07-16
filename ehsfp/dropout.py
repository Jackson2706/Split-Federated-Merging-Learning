"""
Serverless Prototype Dropout for E-HSFP.

Simulates missing/delayed prototype events in serverless environments
by randomly dropping prototype entries during aggregation.
"""

import torch
from typing import Dict, Tuple


class PrototypeDropout:
    """Randomly drops prototype entries to simulate serverless failure modes.

    Deterministic when seed is fixed (uses a separate RNG).
    """

    def __init__(
        self,
        rate: float = 0.2,
        mode: str = "client_prototype",
        seed: int = 42,
    ):
        """
        Args:
            rate: probability of dropping each prototype entry.
            mode: "client_prototype", "edge_prototype", or "both".
            seed: random seed for reproducibility.
        """
        self.rate = rate
        self.mode = mode
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self._drop_count = 0
        self._total_count = 0
        self._apply_calls = 0
        self._changed_calls = 0
        self._active_set_total = 0
        self._active_set_observations = 0
        self._active_set_last = 0

    def apply(
        self,
        proto_dict: Dict[int, torch.Tensor],
        dist_dict: Dict[int, torch.Tensor],
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Apply dropout to prototype/distribution dicts.

        Drops entire class entries (simulating a lost prototype packet).
        Always keeps at least one entry to avoid empty dicts.

        Returns:
            (filtered_proto_dict, filtered_dist_dict)
        """
        if self.rate <= 0 or not proto_dict:
            return proto_dict, dist_dict

        self._apply_calls += 1

        keys = list(proto_dict.keys())
        self._total_count += len(keys)

        if len(keys) <= 1:
            return proto_dict, dist_dict

        # Determine which keys to keep
        keep_mask = torch.rand(len(keys), generator=self.rng) > self.rate
        # Ensure at least one key survives
        if not keep_mask.any():
            keep_mask[torch.randint(len(keys), (1,), generator=self.rng).item()] = True

        kept_keys = [k for k, keep in zip(keys, keep_mask) if keep]
        self._active_set_last = len(kept_keys)
        self._active_set_total += len(kept_keys)
        self._active_set_observations += 1
        dropped = len(keys) - len(kept_keys)
        self._drop_count += dropped
        self._changed_calls += int(dropped > 0)

        filtered_proto = {k: proto_dict[k] for k in kept_keys}
        filtered_dist = {k: dist_dict[k] for k in kept_keys}

        return filtered_proto, filtered_dist

    def apply_to_source_outputs(
        self,
        input_outputs: Dict,
    ) -> Dict:
        """Apply dropout at the source level (drop entire client/edge outputs).

        Simulates a client/edge function timeout or cold start failure.
        Always keeps at least one source.
        """
        if self.rate <= 0 or not input_outputs:
            return input_outputs

        self._apply_calls += 1

        keys = list(input_outputs.keys())
        self._total_count += len(keys)

        if len(keys) <= 1:
            return input_outputs

        keep_mask = torch.rand(len(keys), generator=self.rng) > self.rate
        if not keep_mask.any():
            keep_mask[torch.randint(len(keys), (1,), generator=self.rng).item()] = True

        kept = {k: input_outputs[k] for k, keep in zip(keys, keep_mask) if keep}
        self._active_set_last = len(kept)
        self._active_set_total += len(kept)
        self._active_set_observations += 1
        self._drop_count += len(keys) - len(kept)
        self._changed_calls += int(set(kept) != set(input_outputs))

        return kept

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_prototypes_seen": self._total_count,
            "dropped_prototypes": self._drop_count,
            "apply_calls": self._apply_calls,
            "changed_calls": self._changed_calls,
            "active_set_total": self._active_set_total,
            "active_set_observations": self._active_set_observations,
            "active_set_last": self._active_set_last,
        }

    def reset_stats(self) -> None:
        self._drop_count = 0
        self._total_count = 0
        self._apply_calls = 0
        self._changed_calls = 0
        self._active_set_total = 0
        self._active_set_observations = 0
        self._active_set_last = 0
