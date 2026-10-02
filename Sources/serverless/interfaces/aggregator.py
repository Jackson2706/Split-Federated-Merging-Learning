"""
Abstract aggregator interface for H-SFP.

Each tier (Edge, Cloud) implements this to define how it merges
prototypes / weights received from the tier below.

Serverless note:
  In a deployed setting the aggregator runs as a stateless function that:
    1. Reads all incoming Payloads from the message queue / object store.
    2. Calls aggregate().
    3. Writes the result back so the tier above can consume it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from serverless.interfaces.communication import Payload


class AggregatorBackend(ABC):
    """
    Stateless aggregation contract.

    Implement once per tier-type (EdgeAggregator, CloudAggregator).
    The same implementation works locally and serverlessly — only
    the CommunicationBackend changes.
    """

    @abstractmethod
    def aggregate(self, payloads: List[Payload]) -> Dict[str, Any]:
        """
        Merge a list of incoming payloads into a single aggregated result.

        Args:
            payloads: All payloads received for this round from lower-tier nodes.

        Returns:
            A dict with the aggregated data (prototypes, weights, …).
            This dict becomes the `data` field of the outgoing Payload.
        """

    @abstractmethod
    def is_ready(self, received: int, expected: int) -> bool:
        """
        Decide whether aggregation should proceed.

        Synchronous FL: wait for all expected payloads (received == expected).
        Asynchronous FL: proceed after a threshold (received >= min_fraction * expected).
        """
