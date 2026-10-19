"""
Local (in-process) implementations of the serverless interfaces.

This is what the current H-SFP codebase effectively does today —
all tiers communicate through shared Python objects.  The classes
here make that pattern explicit and replace it with the abstract
interface so the codebase is ready for a real network backend.

To switch to a cloud backend:
  - Replace LocalCommunicationBackend with (e.g.) SQSCommunicationBackend
  - Keep everything else — hierarchy.py, prototype.py, ssl.py — unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from serverless.interfaces.aggregator import AggregatorBackend
from serverless.interfaces.communication import CommunicationBackend, Payload


class LocalCommunicationBackend(CommunicationBackend):
    """
    In-memory message queue — no serialisation, no network.
    Suitable for single-machine experiments (current setup).
    """

    def __init__(self):
        # {(receiver_id, round_id): [Payload, ...]}
        self._queue: Dict[tuple, List[Payload]] = defaultdict(list)

    def send(self, payload: Payload) -> None:
        self._queue[(payload.receiver_id, payload.round_id)].append(payload)

    def receive(self, receiver_id: str, round_id: int) -> Optional[Payload]:
        queue = self._queue[(receiver_id, round_id)]
        return queue.pop(0) if queue else None

    def receive_all(self, receiver_id: str, round_id: int) -> List[Payload]:
        key = (receiver_id, round_id)
        payloads = list(self._queue[key])
        self._queue[key].clear()
        return payloads

    def broadcast(self, payload: Payload, receiver_ids: List[str]) -> None:
        for rid in receiver_ids:
            p = Payload(
                sender_id=payload.sender_id,
                receiver_id=rid,
                round_id=payload.round_id,
                payload_type=payload.payload_type,
                data=payload.data,
                metadata=payload.metadata,
            )
            self.send(p)


class SyncAggregator(AggregatorBackend):
    """
    Synchronous aggregator: waits for all expected payloads before aggregating.
    Mirrors the current H-SFP behaviour in hierarchy.py.

    Override aggregate() to implement custom merging logic (weighted avg, etc.).
    """

    def aggregate(self, payloads: List[Payload]) -> Dict[str, Any]:
        if not payloads:
            return {}
        # Default: return list of all data dicts — caller decides how to merge.
        return {"payloads": [p.data for p in payloads]}

    def is_ready(self, received: int, expected: int) -> bool:
        return received >= expected
