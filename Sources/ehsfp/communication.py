"""Shared, route-based communication accounting.

All values are binary megabytes (MiB, retained as ``*_MB`` for compatibility
with the existing result schema).  Only tensors that cross a simulated node
boundary belong here; local copies and validation forwards do not.
"""

from collections.abc import Mapping

import torch


COMMUNICATION_ROUTES = (
    "client_to_server_MB",
    "server_to_client_MB",
    "client_to_edge_MB",
    "edge_to_client_MB",
    "edge_to_cloud_MB",
    "cloud_to_edge_MB",
)


def new_communication_tracker():
    """Return a tracker with the identical schema used by every method."""
    return {route: 0.0 for route in COMMUNICATION_ROUTES} | {"total_comm_MB": 0.0}


def bytes_of(payload):
    """Exact tensor-storage bytes in a tensor or nested tensor container."""
    if payload is None:
        return 0
    if isinstance(payload, torch.Tensor):
        return payload.numel() * payload.element_size()
    if isinstance(payload, Mapping):
        return sum(bytes_of(value) for value in payload.values())
    if isinstance(payload, (tuple, list)):
        return sum(bytes_of(value) for value in payload)
    return 0


def mb_of(payload):
    return bytes_of(payload) / (1024**2)


def add_communication(tracker, route, *, payload=None, mb=None, copies=1):
    """Add one transmitted payload and keep the compatibility total current."""
    if route not in COMMUNICATION_ROUTES:
        raise KeyError(f"unknown communication route: {route}")
    if (payload is None) == (mb is None):
        raise ValueError("provide exactly one of payload or mb")
    amount = (mb_of(payload) if mb is None else float(mb)) * int(copies)
    tracker[route] += amount
    tracker["total_comm_MB"] += amount
    return amount


def assert_communication_total(tracker):
    """Raise if a tracker total is inconsistent; useful at report boundaries."""
    expected = sum(tracker[route] for route in COMMUNICATION_ROUTES)
    if abs(tracker["total_comm_MB"] - expected) > 1e-9:
        raise AssertionError(
            f"communication total {tracker['total_comm_MB']} != route sum {expected}"
        )
    return expected
