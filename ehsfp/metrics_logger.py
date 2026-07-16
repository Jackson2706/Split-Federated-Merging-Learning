"""
E-HSFP Metrics Logger.

Collects all E-HSFP-specific metrics into a structured dict,
compatible with wandb.log() and JSON output.
"""

import json
import os
from typing import Dict, Any, Optional

try:
    import wandb
except ImportError:
    wandb = None


class EHSFPMetricsLogger:
    """Collects E-HSFP metrics per epoch and supports wandb + JSON output."""

    def __init__(self):
        self.history: list = []
        self._current: Dict[str, Any] = {}

    def reset_epoch(self) -> None:
        """Start collecting metrics for a new epoch."""
        if self._current:
            self.history.append(self._current)
        self._current = {}

    def log(self, key: str, value: Any) -> None:
        self._current[key] = value

    def log_dict(self, d: Dict[str, Any]) -> None:
        self._current.update(d)

    def log_memory_stats(
        self,
        num_current: int,
        num_memory: int,
        replay_ratio: float,
        avg_age: float,
        avg_reliability: float,
    ) -> None:
        self.log("ehsfp/num_current_prototypes", num_current)
        self.log("ehsfp/num_memory_prototypes", num_memory)
        self.log("ehsfp/memory_replay_ratio", replay_ratio)
        self.log("ehsfp/avg_prototype_age", avg_age)
        self.log("ehsfp/avg_reliability_weight", avg_reliability)

    def log_losses(
        self,
        task_loss: Optional[float] = None,
        prc_loss: Optional[float] = None,
        dropout_loss: Optional[float] = None,
        contrastive_loss: Optional[float] = None,
    ) -> None:
        if task_loss is not None:
            self.log("ehsfp/task_loss", task_loss)
        if prc_loss is not None:
            self.log("ehsfp/prc_loss", prc_loss)
        if dropout_loss is not None:
            self.log("ehsfp/dropout_consistency_loss", dropout_loss)
        if contrastive_loss is not None:
            self.log("ehsfp/contrastive_loss", contrastive_loss)

    def log_dropout_stats(
        self,
        dropout_rate: float,
        dropped_count: int,
    ) -> None:
        self.log("ehsfp/prototype_dropout_rate", dropout_rate)
        self.log("ehsfp/dropped_prototype_count", dropped_count)

    def log_communication(
        self,
        prototype_payload_MB: float,
        total_comm_MB: float,
    ) -> None:
        self.log("ehsfp/prototype_payload_MB", prototype_payload_MB)
        self.log("ehsfp/total_comm_MB", total_comm_MB)

    def flush_to_wandb(self, extra: Optional[Dict] = None) -> None:
        """Send current epoch metrics to wandb."""
        if wandb is not None and wandb.run is not None:
            data = dict(self._current)
            if extra:
                data.update(extra)
            wandb.log(data)

    def get_current(self) -> Dict[str, Any]:
        return dict(self._current)

    def append_jsonl(self, path: str, extra: Optional[Dict] = None) -> None:
        """Persist an epoch record without display rounding.

        Diagnostic metrics can occasionally be NaN/Inf (e.g. a degenerate
        per-class reliability variance); sanitize to null so a single non-finite
        diagnostic never aborts the whole run's logging (and final test/metrics
        write). This keeps runs robust without hiding real training divergence,
        which is already tracked via the finite loss/accuracy fields.
        """
        import math

        def _sanitize(o):
            if isinstance(o, float):
                return o if math.isfinite(o) else None
            if isinstance(o, dict):
                return {k: _sanitize(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_sanitize(v) for v in o]
            return o

        data = _sanitize(dict(self._current))
        if extra:
            data.update(_sanitize(dict(extra)))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")

    def finalize(self) -> list:
        """Flush remaining metrics and return full history."""
        if self._current:
            self.history.append(self._current)
            self._current = {}
        return self.history
