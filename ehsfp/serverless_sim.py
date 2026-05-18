"""
Lightweight Serverless Episode Simulator for E-HSFP.

Tracks simulated serverless metrics (latency, cold starts, timeouts)
without modifying the actual learning algorithm. For secondary
experimental metrics only.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class ServerlessInvocationRecord:
    """Record of a single simulated serverless function invocation."""
    tier: str                          # "client", "edge", "cloud"
    node_id: int
    round_idx: int
    is_cold_start: bool = False
    simulated_latency: float = 0.0     # seconds
    timed_out: bool = False
    actual_wall_time: float = 0.0      # real elapsed time
    prototypes_processed: int = 0
    memory_replays_used: int = 0
    episode_id: Optional[str] = None


@dataclass
class ServerlessEpisode:
    """A single serverless training episode (one round)."""
    round_idx: int
    episode_id: str
    invocations: List[ServerlessInvocationRecord] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def cold_starts(self) -> int:
        return sum(1 for inv in self.invocations if inv.is_cold_start)

    @property
    def timeouts(self) -> int:
        return sum(1 for inv in self.invocations if inv.timed_out)


class ServerlessMetricsTracker:
    """Tracks serverless simulation metrics across the training run.

    Usage:
        tracker = ServerlessMetricsTracker(config)
        with tracker.episode(round_idx) as ep:
            record = tracker.simulate_invocation("client", cid, round_idx)
            # ... do actual processing ...
            record.actual_wall_time = elapsed
    """

    def __init__(self, config: dict):
        self.cold_start_prob = config.get("cold_start_probability", 0.15)
        self.timeout_prob = config.get("function_timeout_probability", 0.05)
        self.max_duration = config.get("max_episode_duration", 30.0)
        self.latency_mean = config.get("latency_mean", 0.5)
        self.latency_std = config.get("latency_std", 0.2)

        self.rng = torch.Generator()
        self.rng.manual_seed(42)

        self.episodes: List[ServerlessEpisode] = []
        self._current_episode: Optional[ServerlessEpisode] = None

        # Cumulative counters
        self.invocation_count = 0
        self.cold_start_count = 0
        self.warm_start_count = 0
        self.dropped_event_count = 0
        self.stale_event_count = 0
        self.processed_prototype_count = 0
        self.memory_replay_count = 0
        self.total_simulated_latency = 0.0
        self._cost_units = 0.0

    def begin_episode(self, round_idx: int) -> ServerlessEpisode:
        ep = ServerlessEpisode(
            round_idx=round_idx,
            episode_id=f"ep_{round_idx}",
            start_time=time.time(),
        )
        self._current_episode = ep
        return ep

    def end_episode(self) -> None:
        if self._current_episode is not None:
            self._current_episode.end_time = time.time()
            self.episodes.append(self._current_episode)
            self._current_episode = None

    def simulate_invocation(
        self, tier: str, node_id: int, round_idx: int
    ) -> ServerlessInvocationRecord:
        """Simulate a serverless invocation and return the record."""
        is_cold = torch.rand(1, generator=self.rng).item() < self.cold_start_prob
        timed_out = torch.rand(1, generator=self.rng).item() < self.timeout_prob
        latency = max(
            0.01,
            self.latency_mean + self.latency_std * torch.randn(1, generator=self.rng).item()
        )
        if is_cold:
            latency *= 2.5  # cold start penalty

        record = ServerlessInvocationRecord(
            tier=tier,
            node_id=node_id,
            round_idx=round_idx,
            is_cold_start=is_cold,
            simulated_latency=latency,
            timed_out=timed_out,
            episode_id=self._current_episode.episode_id if self._current_episode else None,
        )

        # Update counters
        self.invocation_count += 1
        if is_cold:
            self.cold_start_count += 1
        else:
            self.warm_start_count += 1
        if timed_out:
            self.dropped_event_count += 1
        self.total_simulated_latency += latency
        self._cost_units += latency * (2.0 if tier == "cloud" else 1.0)

        if self._current_episode is not None:
            self._current_episode.invocations.append(record)

        return record

    def record_prototype_processing(self, count: int = 1) -> None:
        self.processed_prototype_count += count

    def record_memory_replay(self, count: int = 1) -> None:
        self.memory_replay_count += count

    def record_stale_event(self, count: int = 1) -> None:
        self.stale_event_count += count

    @property
    def estimated_cost_proxy(self) -> float:
        """Proxy cost in arbitrary units (latency * tier_weight)."""
        return self._cost_units

    def get_summary(self) -> Dict:
        """Summary dict for logging."""
        return {
            "serverless/invocation_count": self.invocation_count,
            "serverless/cold_start_count": self.cold_start_count,
            "serverless/warm_start_count": self.warm_start_count,
            "serverless/dropped_event_count": self.dropped_event_count,
            "serverless/stale_event_count": self.stale_event_count,
            "serverless/processed_prototype_count": self.processed_prototype_count,
            "serverless/memory_replay_count": self.memory_replay_count,
            "serverless/total_simulated_latency": self.total_simulated_latency,
            "serverless/estimated_cost_proxy": self.estimated_cost_proxy,
            "serverless/num_episodes": len(self.episodes),
        }
