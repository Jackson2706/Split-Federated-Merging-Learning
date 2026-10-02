"""Shared, auditable classification metric artifacts."""

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def save_prediction_artifact(path, predictions, labels):
    """Persist raw one-dimensional predictions and labels in a stable NPZ schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"predictions and labels must have the same shape: "
            f"{predictions.shape} != {labels.shape}"
        )
    np.savez(path, predictions=predictions, labels=labels)


def recompute_classification_metrics(predictions, labels):
    """Independently compute journal metrics solely from raw arrays."""
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"predictions and labels must have the same shape: "
            f"{predictions.shape} != {labels.shape}"
        )
    classes = np.unique(np.concatenate((labels, predictions)))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "classes": classes.tolist(),
        "per_class_recall": recall_score(
            labels, predictions, labels=classes, average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=classes
        ).tolist(),
    }


def validate_prediction_artifact(path):
    """Load the stable NPZ schema and recompute all metrics from scratch."""
    with np.load(path, allow_pickle=False) as artifact:
        if set(artifact.files) != {"predictions", "labels"}:
            raise ValueError(
                "prediction artifact must contain exactly 'predictions' and 'labels'"
            )
        return recompute_classification_metrics(
            artifact["predictions"], artifact["labels"]
        )
