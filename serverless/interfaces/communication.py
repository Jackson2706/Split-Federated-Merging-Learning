"""
Abstract communication interface for H-SFP.

Current implementation: all tiers run in-process (no real network).
Serverless target: each tier runs as an independent function (Lambda / Cloud Run).

To port H-SFP to serverless:
  1. Implement CommunicationBackend for your cloud provider.
  2. Swap the LocalCommunicationBackend in adapters/local.py with the new one.
  3. Serialise Payload via the provided to_bytes() / from_bytes() helpers.

Data exchanged between tiers (already compact thanks to prototypes):
  - Client → Edge : prototypes + distribution stds  (not full model weights)
  - Edge  → Cloud : aggregated prototypes + distribution stds
  - Cloud → Edge  : cloud model weights (one-way, broadcast)
  - Edge  → Client: edge model weights  (one-way, broadcast)
"""

from __future__ import annotations

import io
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Payload:
    """Unit of data exchanged between tiers in one FL round."""

    sender_id: str                          # e.g. "client_3", "edge_1", "cloud"
    receiver_id: str                        # e.g. "edge_1", "cloud", "broadcast"
    round_id: int
    payload_type: str                       # "prototypes" | "weights" | "gradients"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        pickle.dump(self, buf)
        return buf.getvalue()

    @staticmethod
    def from_bytes(raw: bytes) -> "Payload":
        return pickle.loads(raw)  # noqa: S301  (internal trusted data only)

    def size_mb(self) -> float:
        return len(self.to_bytes()) / (1024 ** 2)


class CommunicationBackend(ABC):
    """
    Abstract transport layer between FL tiers.

    In local mode:  send() queues to an in-memory dict, receive() pops from it.
    In serverless:  send() publishes to SQS/Pub-Sub, receive() is triggered by the
                    cloud event (Lambda handler / Cloud Run endpoint).
    """

    @abstractmethod
    def send(self, payload: Payload) -> None:
        """Push a payload toward its receiver."""

    @abstractmethod
    def receive(self, receiver_id: str, round_id: int) -> Optional[Payload]:
        """
        Pull the next payload addressed to `receiver_id` for `round_id`.
        Returns None when no message is available yet.
        """

    @abstractmethod
    def broadcast(self, payload: Payload, receiver_ids: list[str]) -> None:
        """Send the same payload to multiple receivers (e.g. cloud→all edges)."""
