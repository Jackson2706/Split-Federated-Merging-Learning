"""Phase-0 reproducibility, identity, and overwrite guards."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def canonical_config(config: Mapping[str, Any]) -> str:
    """Return stable, unrounded JSON for a fully resolved configuration."""
    return json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"), allow_nan=False)


def resolved_config_hash(config: Mapping[str, Any], effective_ehsfp: Mapping[str, Any]) -> str:
    payload = {"config": dict(config), "effective_ehsfp": dict(effective_ehsfp)}
    return hashlib.sha256(canonical_config(payload).encode()).hexdigest()


def partition_hash(user_groups: Mapping[Any, Iterable[Any]]) -> str:
    normalized = {str(k): sorted(int(i) for i in v) for k, v in user_groups.items()}
    return hashlib.sha256(canonical_config(normalized).encode()).hexdigest()


def git_commit(root: str) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def architecture_manifest(models: Sequence[Any]) -> Dict[str, Any]:
    tiers = []
    for model in models:
        cls = type(model)
        try:
            source = inspect.getsourcefile(cls) or "unknown"
            source_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        except (OSError, TypeError):
            source, source_digest = "unknown", "unknown"
        tiers.append({
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "source": os.path.abspath(source) if source != "unknown" else source,
            "source_sha256": source_digest,
            "state_shapes": {k: list(v.shape) for k, v in model.state_dict().items()},
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })
    digest = hashlib.sha256(canonical_config(tiers).encode()).hexdigest()
    return {"architecture_hash": digest, "architecture_id": digest[:12], "tiers": tiers}


def prepare_run_dir(base_dir: str, architecture_id: str, config_hash: str) -> str:
    """Reserve an architecture/config-specific directory without overwriting."""
    run_dir = os.path.join(base_dir, architecture_id, config_hash)
    if os.path.exists(run_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing run directory {run_dir}; "
            "use a distinct resolved config or archive the existing run."
        )
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def validate_checkpoint_config_hash(checkpoint: Mapping[str, Any], expected_hash: str) -> None:
    found = checkpoint.get("resolved_config_hash")
    if found != expected_hash:
        raise RuntimeError(
            f"Checkpoint config hash mismatch: expected {expected_hash}, found {found!r}"
        )


class RuntimeCounters:
    """Small serializable counter set shared by all E-HSFP components."""

    _DEFAULTS = (
        "memory.reads", "memory.writes", "memory.replays",
        "memory.replay_calls", "memory.changed_classes",
        "reliability.aggregation_calls", "reliability.nonuniform_weight_calls",
        "reliability.weight_variance_sum", "reliability.weight_variance_observations",
        "prc.calls", "prc.shared_classes", "prc.nonzero_loss_calls",
        "prc.loss_sum", "prc.loss_observations",
        "dropout.apply_calls", "dropout.changed_calls", "dropout.total_seen",
        "dropout.active_set_total", "dropout.active_set_observations",
        "dropout.active_set_last",
    )

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter({key: 0 for key in self._DEFAULTS})

    def increment(self, key: str, amount=1) -> None:
        self._counts[key] += amount

    def set(self, key: str, value) -> None:
        self._counts[key] = value

    def snapshot(self) -> Dict[str, Any]:
        # Device counters are deliberately accumulated without .item() in hot
        # CUDA loops.  Materialize them together at this explicit reporting
        # boundary instead of forcing one host/device synchronization per loss
        # or per class.
        result = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in self._counts.items()
        }
        rel_n = result["reliability.weight_variance_observations"]
        prc_n = result["prc.loss_observations"]
        active_n = result["dropout.active_set_observations"]
        result["reliability.weight_variance"] = (
            result["reliability.weight_variance_sum"] / rel_n if rel_n else 0.0
        )
        result["prc.loss_mean"] = result["prc.loss_sum"] / prc_n if prc_n else 0.0
        result["dropout.active_set_mean"] = (
            result["dropout.active_set_total"] / active_n if active_n else 0.0
        )
        return dict(sorted(result.items()))
