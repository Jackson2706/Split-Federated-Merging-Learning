"""Unit checks for the revision mechanisms (CPU, fast)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ehsfp.aggregation import reliability_weighted_aggregate  # noqa: E402
from ehsfp.dropout import PrototypeDropout  # noqa: E402
from ehsfp.losses import prototype_replay_consistency_loss  # noqa: E402
from ehsfp.revision import bures_sq, ece_nll, prop2_class_loss, w2_report  # noqa: E402
from ehsfp.synthesis import between_source_scatter, within_class_scatter  # noqa: E402


def _src(mu, sd):
    return ({0: torch.tensor(mu)}, {0: torch.tensor(sd)})


def test_mixture_one_source_matches_diagonal():
    torch.manual_seed(0)
    out = {"a": _src([1.0, -2.0, 0.5], [0.5, 2.0, 1.0])}
    n = 100_000
    xm, _ = reliability_weighted_aggregate(out, n, torch.device("cpu"), aggregation_mode="average",
                                           sampler="mixture")
    xg, _ = reliability_weighted_aggregate(out, n, torch.device("cpu"), aggregation_mode="average")
    assert torch.allclose(xm.mean(0), xg.mean(0), atol=0.03)
    assert torch.allclose(xm.var(0), xg.var(0), rtol=0.03)
    assert torch.allclose(xm.mean(0), torch.tensor([1.0, -2.0, 0.5]), atol=0.03)


def test_mixture_matches_component_mixture():
    torch.manual_seed(1)
    out = {"a": _src([0.0], [1.0]), "b": _src([10.0], [1.0])}
    sup = {"a": {0: 3}, "b": {0: 1}}
    x, _ = reliability_weighted_aggregate(out, 100_000, torch.device("cpu"), support_map=sup,
                                          aggregation_mode="sample_count_weighted", sampler="mixture")
    assert abs(float((x > 5).float().mean()) - 0.25) < 0.01  # picks b with weight 1/4


def test_precision_equals_count_when_sigmas_equal():
    out = {"a": ({0: torch.tensor([0.0, 1.0])}, {0: torch.tensor([1.0, 1.0])}),
           "b": ({0: torch.tensor([2.0, 3.0])}, {0: torch.tensor([1.0, 1.0])})}
    sup = {"a": {0: 5}, "b": {0: 15}}
    stats = {}
    for mode in ("sample_count_weighted", "precision_weighted"):
        _, _, stats[mode] = reliability_weighted_aggregate(out, 2, torch.device("cpu"), support_map=sup,
                                                           aggregation_mode=mode, return_stats=True)
    assert torch.allclose(stats["sample_count_weighted"][1], stats["precision_weighted"][1])
    assert torch.allclose(stats["sample_count_weighted"][2], stats["precision_weighted"][2])


def test_scatter_fix_recovers_union_within_class_scatter():
    g = torch.Generator().manual_seed(0)
    sources = []
    for shift in (0.0, 3.0, -2.0):
        sources.append({c: [torch.randn(4, generator=g) + shift + c for _ in range(5 + c)] for c in range(3)})
    within = sum(within_class_scatter(s)[0] for s in sources)
    outs = {k: ({c: torch.stack(v).mean(0) for c, v in s.items()}, {}) for k, s in enumerate(sources)}
    sup = {k: {c: len(v) for c, v in s.items()} for k, s in enumerate(sources)}
    union = within_class_scatter({c: [x for s in sources for x in s[c]] for c in range(3)})[0]
    assert torch.allclose(within + between_source_scatter(outs, sup), union, atol=1e-4)


def _prc_mc(mode, reps=400):
    torch.manual_seed(0)
    A = torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, -1.0]])
    h = lambda z: z @ A.T  # noqa: E731
    mu = {0: torch.zeros(3), 1: torch.ones(3)}
    sd = {0: torch.tensor([1.0, 0.5, 2.0]), 1: torch.tensor([0.3, 0.3, 0.3])}
    vals = [float(prototype_replay_consistency_loss(mu, sd, mu, sd, h, num_samples=64, mode=mode,
                                                    max_replay_batch=10_000)) for _ in range(reps)]
    trV = sum(float(torch.trace(A @ torch.diag(sd[c] ** 2) @ A.T)) for c in mu)
    return sum(vals) / reps, trV


def test_corrected_prc_has_zero_expectation_for_identical_laws():
    m, _ = _prc_mc("corrected")
    assert abs(m) < 0.03


def test_standard_prc_equals_twice_output_variance():
    m, trV = _prc_mc("standard")
    expected = 2 * trV / (2 * 2)  # 2 tr(V) / (C d_h), C = 2 classes, d_h = 2
    assert abs(m - expected) / expected < 0.03


def test_proposition2_class_loss_rate():
    p, S, S_j, n = 0.3, 4, 2, 100_000
    drop = PrototypeDropout(rate=p, seed=0)
    outs = {k: ({0: None} if k < S_j else {1: None}, None) for k in range(S)}
    lost = 0
    for _ in range(n):
        kept = drop.apply_to_source_outputs(outs)
        lost += not any(0 in v[0] for v in kept.values())
    rate, pred = lost / n, prop2_class_loss(p, S, S_j)
    assert abs(rate - pred) < 4 * (pred * (1 - pred) / n) ** 0.5


def test_bures_and_w2_report():
    A = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    assert bures_sq(A, A) < 1e-9
    a, b = torch.tensor([1.0, 4.0]), torch.tensor([9.0, 1.0])
    assert abs(bures_sq(torch.diag(a), torch.diag(b)) - float(((a.sqrt() - b.sqrt()) ** 2).sum())) < 1e-9
    act = ({0: torch.tensor([1.0, 0.0])}, {0: torch.tensor([1.0, 1.0])})
    ref = ({0: torch.tensor([0.0, 0.0])}, {0: torch.tensor([2.0, 1.0])})
    r = w2_report(act, ref, {0: 1.0})
    assert abs(r["mean_term"] - 1.0) < 1e-9 and abs(r["cov_term"] - 1.0) < 1e-9


def test_ece_nll_confident_correct():
    logits = torch.tensor([[20.0, 0.0], [0.0, 20.0]])
    ece, nll = ece_nll(logits, torch.tensor([0, 1]))
    assert ece < 1e-6 and nll < 1e-6
