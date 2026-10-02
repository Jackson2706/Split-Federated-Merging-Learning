"""Pure research metrics used by E-HSFP experiment reporting.

The operational definitions are frozen in ``docs/JOURNAL_SIMULATION_RESULTS.md``
(section 3.9).  This module deliberately contains no training-loop state or
checkpoint-selection logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

import torch

from camera_ready.feature_distance import rbf_mmd


ClassId = TypeVar("ClassId")


def trailing_window_stability(
    metric_history: Sequence[float], window: int = 5
) -> list[float | None]:
    """Return trailing population variance, with ``None`` until W observations."""
    if window <= 0:
        raise ValueError("window must be positive")

    stability: list[float | None] = []
    for end in range(1, len(metric_history) + 1):
        if end < window:
            stability.append(None)
            continue
        values = [float(value) for value in metric_history[end - window : end]]
        mean = sum(values) / window
        stability.append(sum((value - mean) ** 2 for value in values) / window)
    return stability


def rounds_to_convergence(
    metric_history: Sequence[float],
    smoothing_window: int = 3,
    threshold_fraction: float = 0.95,
    patience: int = 3,
) -> int | str:
    """Return the first 1-based round sustaining the frozen convergence rule.

    A trailing moving average is defined only once its full smoothing window is
    available.  The threshold is ``threshold_fraction`` times the best raw
    validation metric observed in the run.  A candidate must remain at or above
    that threshold for ``patience`` consecutive smoothed observations.
    """
    if smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be between 0 and 1")
    if not metric_history:
        return "not_converged"

    values = [float(value) for value in metric_history]
    threshold = threshold_fraction * max(values)
    smoothed: list[tuple[int, float]] = []
    for end in range(smoothing_window, len(values) + 1):
        average = sum(values[end - smoothing_window : end]) / smoothing_window
        smoothed.append((end, average))  # end is also the original 1-based round

    for candidate in range(len(smoothed)):
        patience_window = smoothed[candidate : candidate + patience]
        if len(patience_window) < patience:
            break
        if all(average >= threshold for _, average in patience_window):
            return smoothed[candidate][0]
    return "not_converged"


def prototype_drift(
    prev_prototypes: Mapping[ClassId, torch.Tensor] | None,
    curr_prototypes: Mapping[ClassId, torch.Tensor],
) -> dict[ClassId | str, float] | None:
    """Return per-class L2 drift plus global mean and maximum summaries.

    Only classes present in both rounds can be compared.  ``None`` is the
    explicit no-previous-round sentinel; it prevents round 1 being mislabeled
    as zero drift.  If rounds share no classes, the result is also undefined.
    """
    if not prev_prototypes:
        return None

    shared_classes = prev_prototypes.keys() & curr_prototypes.keys()
    if not shared_classes:
        return None

    per_class = {
        class_id: float(
            torch.linalg.vector_norm(
                curr_prototypes[class_id].detach() - prev_prototypes[class_id].detach()
            ).item()
        )
        for class_id in shared_classes
    }
    distances = list(per_class.values())
    return {
        **per_class,
        "global": sum(distances) / len(distances),
        "max": max(distances),
    }


def recovery_gap(clean_final_metric: float, stressed_final_metric: float) -> float:
    """Return clean minus stressed best-validation-checkpoint metric."""
    return float(clean_final_metric) - float(stressed_final_metric)


def recovery_rounds(
    post_stress_metric_history: Sequence[float],
    pre_stress_metric: float,
    threshold_fraction: float = 0.95,
) -> int | None:
    """Return the first 1-based post-lift round reaching 95% of pre-stress."""
    if not 0.0 <= threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be between 0 and 1")
    threshold = threshold_fraction * float(pre_stress_metric)
    return next(
        (round_index for round_index, value in enumerate(post_stress_metric_history, 1)
         if float(value) >= threshold),
        None,
    )


def prototype_fidelity_mmd(
    real_class_features: torch.Tensor,
    synthetic_class_features: torch.Tensor,
    *,
    sigmas=None,
    max_samples: int = 2000,
) -> float:
    """Compute fidelity for matched class/round feature tensors using RBF MMD².

    Callers are responsible for supplying real client-encoder features and the
    corresponding synthetic prototype-space samples for the same class/round.
    """
    if real_class_features.shape[0] == 0 or synthetic_class_features.shape[0] == 0:
        raise ValueError("real and synthetic feature tensors must be non-empty")
    return rbf_mmd(
        real_class_features,
        synthetic_class_features,
        sigmas=sigmas,
        max_samples=max_samples,
    )
