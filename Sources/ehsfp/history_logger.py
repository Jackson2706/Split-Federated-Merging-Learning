"""Durable, shared per-round experiment-history logging.

The frozen ``history.csv`` schema is, in order:

``round``, ``elapsed_time_s``, ``train_loss``,
``validation_metric_name``, ``validation_metric_value``, one round-delta
column for each route in :data:`ehsfp.communication.COMMUNICATION_ROUTES`,
``comm_total_delta_MB``, phase resource columns (CPU percent, RSS bytes, CUDA
peak allocated bytes, and CUDA peak reserved bytes for client, edge, and
cloud), ``client_memory_records``, ``edge_memory_records``, and ``event``.

Resource cells are empty when a method does not already measure that phase;
this module deliberately performs no resource measurement.  Memory cells are
empty when episodic memory is inapplicable.  ``event`` is an empty string when
nothing notable occurred.  Communication values are binary MiB, retaining the
repository's historical ``*_MB`` spelling.

Each row is flushed and fsynced immediately.  Thus a stopped process leaves a
valid CSV containing every completed round written before the interruption.
"""

from __future__ import annotations

import csv
import math
import os
from collections.abc import Mapping
from pathlib import Path

from ehsfp.communication import COMMUNICATION_ROUTES


COMMUNICATION_DELTA_COLUMNS = tuple(
    f"comm_{route.removesuffix('_MB')}_delta_MB" for route in COMMUNICATION_ROUTES
)
PHASES = ("client", "edge", "cloud")
RESOURCE_COLUMNS = tuple(
    f"{phase}_{measurement}"
    for phase in PHASES
    for measurement in (
        "cpu_percent",
        "rss_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
    )
)
HISTORY_COLUMNS = (
    "round",
    "elapsed_time_s",
    "train_loss",
    "validation_metric_name",
    "validation_metric_value",
    *COMMUNICATION_DELTA_COLUMNS,
    "comm_total_delta_MB",
    *RESOURCE_COLUMNS,
    "client_memory_records",
    "edge_memory_records",
    "event",
)


def communication_snapshot(tracker: Mapping[str, float]) -> dict[str, float]:
    """Copy cumulative public communication counters at a round boundary."""
    return {
        **{route: float(tracker[route]) for route in COMMUNICATION_ROUTES},
        "total_comm_MB": float(tracker["total_comm_MB"]),
    }


def communication_deltas(
    start: Mapping[str, float], current: Mapping[str, float]
) -> dict[str, float]:
    """Return current-minus-start communication for every public route."""
    deltas = {
        column: float(current[route]) - float(start[route])
        for route, column in zip(COMMUNICATION_ROUTES, COMMUNICATION_DELTA_COLUMNS)
    }
    deltas["comm_total_delta_MB"] = (
        float(current["total_comm_MB"]) - float(start["total_comm_MB"])
    )
    return deltas


class RoundHistoryLogger:
    """Append one durable, schema-stable CSV row for each completed round."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), None)
            if header != list(HISTORY_COLUMNS):
                raise ValueError(f"incompatible history schema in {self.path}")
        else:
            self._write_header()

    def _write_header(self) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(HISTORY_COLUMNS)
            handle.flush()
            os.fsync(handle.fileno())

    def append_round(
        self,
        *,
        round_number: int,
        elapsed_time_s: float,
        train_loss: float,
        validation_metric_name: str,
        validation_metric_value: float | None,
        communication_start: Mapping[str, float],
        communication_current: Mapping[str, float],
        resources: Mapping[str, float | int | None] | None = None,
        client_memory_records: int | None = None,
        edge_memory_records: int | None = None,
        event: str = "",
    ) -> None:
        """Append, flush, and fsync one completed-round record."""
        row = {
            column: "" for column in HISTORY_COLUMNS
        }
        row.update(
            {
                "round": int(round_number),
                "elapsed_time_s": self._finite(elapsed_time_s, "elapsed_time_s"),
                "train_loss": self._finite(train_loss, "train_loss"),
                "validation_metric_name": str(validation_metric_name),
                "validation_metric_value": (
                    ""
                    if validation_metric_value is None
                    else self._finite(validation_metric_value, "validation_metric_value")
                ),
                "client_memory_records": (
                    "" if client_memory_records is None else int(client_memory_records)
                ),
                "edge_memory_records": (
                    "" if edge_memory_records is None else int(edge_memory_records)
                ),
                "event": str(event),
            }
        )
        row.update(communication_deltas(communication_start, communication_current))
        for key, value in (resources or {}).items():
            if key not in RESOURCE_COLUMNS:
                raise KeyError(f"unknown history resource column: {key}")
            row[key] = "" if value is None else self._finite(value, key)

        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS).writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _finite(value: float | int, field: str) -> float | int:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field} must be finite")
        return value
