"""Episodic stress protocol: neutral settings are exact no-ops; active ones behave."""

import torch

from ehsfp.aggregation import reliability_weighted_aggregate
from ehsfp.episodic_stress import (
    PartialEdgeExecution,
    StalePacketQueue,
    gaussian_w2_drift,
    total_variance_aggregate,
)
from ehsfp.memory import EpisodicPrototypeMemory
from ehsfp.reliability import PrototypeReliabilityNetwork
from ehsfp.serverless_sim import ServerlessMetricsTracker


def _packet(v):
    # Multi-element tensors: record equality must never be evaluated on them.
    return ({0: torch.full((4,), v), 1: torch.full((4,), -v)},
            {0: torch.ones(4), 1: torch.ones(4)})


def test_stale_queue_tau0_is_immediate_and_draws_nothing():
    q = StalePacketQueue(0, seed=3)
    state = q.rng.get_state().clone()
    q.enqueue(5, 7, _packet(1.0), {0: 10})
    got = q.deliver(7)
    assert [p.key for p in got] == ["5@7"] and got[0].age(7) == 0
    assert torch.equal(state, q.rng.get_state())
    assert not q.active


def test_stale_queue_bounded_delays_and_true_ages():
    q = StalePacketQueue(3, seed=0)
    for r in range(1, 21):
        q.enqueue(r, r, _packet(float(r)), {})
    seen = []
    for r in range(1, 30):
        for p in q.deliver(r):
            assert 0 <= p.age(r) <= 3
            assert p.deliver_round == r
            seen.append(p.age(r))
    assert len(seen) == 20 and len(q) == 0
    assert max(seen) > 0  # some packets really were delayed


def test_cold_start_deferral_adds_one_round_even_at_tau0():
    q = StalePacketQueue(0, seed=0)
    q.enqueue(1, 4, _packet(1.0), {}, extra_delay=1)
    assert q.deliver(4) == []
    assert q.deliver(5)[0].age(5) == 1


def test_partial_edge_neutral_and_active():
    neutral = PartialEdgeExecution(0.0, 0.5, seed=1)
    s = neutral.rng.get_state().clone()
    assert neutral.batch_budget(100) == 100
    assert torch.equal(s, neutral.rng.get_state())
    always = PartialEdgeExecution(1.0, 0.25, seed=1)
    assert always.batch_budget(100) == 25 and always.events == 1


def test_memory_per_class_capacity_fifo_and_zero():
    m = EpisodicPrototypeMemory(max_size=1000, per_class_capacity=2)
    for r in range(4):
        m.add_from_proto_dicts(*_packet(float(r)), source_id=f"c{r}", round_idx=r)
        m.age_all()
    by0 = m.get_by_class(0)
    assert len(by0) == 2 and sorted(x.round_idx for x in by0) == [2, 3]
    off = EpisodicPrototypeMemory(per_class_capacity=0)
    off.add_from_proto_dicts(*_packet(1.0), source_id="c", round_idx=0)
    assert len(off) == 0
    legacy = EpisodicPrototypeMemory(max_size=3)
    for r in range(4):
        legacy.add_from_proto_dicts(*_packet(float(r)), source_id="c", round_idx=r, age=r)
    assert len(legacy) == 3 and {x.age for x in legacy._records} <= {0, 1, 2, 3}


def test_w2_drift_definition():
    a = ({0: torch.zeros(3)}, {0: torch.ones(3)})
    b = ({0: torch.ones(3)}, {0: 2 * torch.ones(3)})
    assert gaussian_w2_drift(a, a)["drift_w2"] == 0.0
    d = gaussian_w2_drift(a, b)
    assert abs(d["drift_w2"] - 6.0) < 1e-6  # ||0-1||^2*3 + ||1-2||^2*3


def test_total_variance_aggregate_matches_law_of_total_variance():
    outs = {"a": ({0: torch.tensor([0.0])}, {0: torch.tensor([1.0])}),
            "b": ({0: torch.tensor([2.0])}, {0: torch.tensor([1.0])})}
    p, d = total_variance_aggregate(outs, {"a": {0: 1}, "b": {0: 1}})
    assert torch.allclose(p[0], torch.tensor([1.0]))
    assert torch.allclose(d[0] ** 2, torch.tensor([2.0]), atol=1e-6)  # 1 + 1


def test_aggregate_defaults_unchanged_and_stats_consistent():
    outs = {0: _packet(1.0), 1: _packet(3.0)}
    net = PrototypeReliabilityNetwork(hidden_dim=8)
    for mode in ("average", "sample_count_weighted", "learnable_reliability"):
        torch.manual_seed(0)
        f1, l1 = reliability_weighted_aggregate(outs, 5, torch.device("cpu"), reliability_net=net,
                                                support_map={0: {0: 3, 1: 3}, 1: {0: 1, 1: 1}},
                                                aggregation_mode=mode)
        torch.manual_seed(0)
        f2, l2, (labels, mus, sds) = reliability_weighted_aggregate(
            outs, 5, torch.device("cpu"), reliability_net=net,
            support_map={0: {0: 3, 1: 3}, 1: {0: 1, 1: 1}}, aggregation_mode=mode,
            return_stats=True)
        assert torch.equal(f1, f2) and torch.equal(l1, l2)
        assert labels == [0, 1] and mus.shape == (2, 4)
    torch.manual_seed(0)
    f0, _ = reliability_weighted_aggregate(outs, 5, torch.device("cpu"), reliability_net=net,
                                           aggregation_mode="learnable_reliability")
    torch.manual_seed(0)
    fz, _ = reliability_weighted_aggregate(outs, 5, torch.device("cpu"), reliability_net=net,
                                           aggregation_mode="learnable_reliability",
                                           age_map={0: 0, 1: 0})
    assert torch.equal(f0, fz)


def test_tier_timeouts_default_to_shared_probability():
    base = ServerlessMetricsTracker({"function_timeout_probability": 0.3})
    split = ServerlessMetricsTracker({"function_timeout_probability": 0.3,
                                      "client_timeout_probability": 0.3,
                                      "edge_timeout_probability": 0.3})
    for i in range(200):
        tier = "client" if i % 3 else "edge"
        a = base.simulate_invocation(tier, i, i)
        b = split.simulate_invocation(tier, i, i)
        assert a.timed_out == b.timed_out and a.is_cold_start == b.is_cold_start
    only_edge = ServerlessMetricsTracker({"function_timeout_probability": 0.3,
                                          "client_timeout_probability": 0.0})
    assert not any(only_edge.simulate_invocation("client", i, i).timed_out for i in range(100))


def test_per_class_eviction_when_victim_is_not_first_record():
    # Regression: eviction used list.remove(), whose == on the dataclass compared
    # multi-element tensors and raised once the victim was not the first record.
    m = EpisodicPrototypeMemory(max_size=1000, per_class_capacity=1)
    m.add_from_proto_dicts({5: torch.zeros(4)}, {5: torch.ones(4)}, source_id="a", round_idx=0)
    for r in range(3):
        m.add_from_proto_dicts({7: torch.full((4,), float(r))}, {7: torch.ones(4)},
                               source_id=f"b{r}", round_idx=r)
        m.age_all()
    assert len(m.get_by_class(7)) == 1 and m.get_by_class(7)[0].round_idx == 2
    assert len(m.get_by_class(5)) == 1
