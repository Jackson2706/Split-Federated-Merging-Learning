"""C1 (drift-compensated replay) and C2 (bias-variance-optimal alpha) unit tests; CPU only."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ehsfp.drift_comp import DriftState, ReplayAdjuster, compensated_copy  # noqa: E402
from ehsfp.memory import EpisodicPrototypeMemory, PrototypeRecord, mix_current_and_memory  # noqa: E402

D, N_SRC, N_CLS, NOW = 16, 6, 5, 10


def world(rate_noise, seed=0):
    """means[(l, j, t)] = base + r_{l,j} t with r = r0 + rate_noise * eps."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(N_SRC, N_CLS, D, generator=g) * 3.0
    r0 = torch.randn(D, generator=g)
    r = r0 + rate_noise * torch.randn(N_SRC, N_CLS, D, generator=g)
    return lambda l, j, t: base[l, j] + r[l, j] * t, r


def fill(state, mean_at, stamps):
    for l in range(N_SRC):
        for j in range(N_CLS):
            for t in stamps:
                state.observe(f"s{l}", j, t, mean_at(l, j, t))


def records_at(mean_at, stamp):
    return [PrototypeRecord(class_id=j, mu=mean_at(l, j, stamp), sigma=torch.ones(D), support_count=5,
                            source_id=f"s{l}", round_idx=stamp) for l in range(N_SRC) for j in range(N_CLS)]


def current_means(mean_at):
    return [mean_at(0, j, NOW) for j in range(N_CLS)]


def error_ratio(adj, state, recs, mean_at):
    h = adj.bandwidth(current_means(mean_at))
    disp, _, ok = adj.estimate(state, recs, NOW, h, torch.device("cpu"))
    assert ok.all()
    num = den = 0.0
    for i, r in enumerate(recs):
        l, j = int(r.source_id[1:]), r.class_id
        star = mean_at(l, j, NOW)
        num += float((r.mu + disp[i] - star).pow(2).sum())
        den += float((r.mu - star).pow(2).sum())
    return num / den


@pytest.mark.parametrize("mode", ["rate", "pair"])
def test_linear_drift_compensation_error_ratio(mode):
    mean_at, _ = world(rate_noise=0.1)
    st = DriftState(window=5)
    fill(st, mean_at, range(4, NOW + 1))
    adj = ReplayAdjuster(drift_comp=mode)
    assert error_ratio(adj, st, records_at(mean_at, 6), mean_at) < 0.1


def test_zero_drift_leaves_records_unchanged():
    mean_at, _ = world(rate_noise=0.0)
    still = lambda l, j, t: mean_at(l, j, 0)  # no drift at all
    st = DriftState(window=5)
    fill(st, still, range(4, NOW + 1))
    recs = records_at(still, 6)
    adj = ReplayAdjuster(drift_comp="rate")
    disp, var, ok = adj.estimate(st, recs, NOW, adj.bandwidth(current_means(still)), torch.device("cpu"))
    assert ok.all() and float(disp.abs().max()) < 1e-7 and float(var.max()) < 1e-7
    assert torch.allclose(compensated_copy(recs[0], recs[0].mu + disp[0]).mu, recs[0].mu, atol=1e-7)


def test_pair_equals_rate_when_drift_is_exactly_linear():
    mean_at, _ = world(rate_noise=0.0)  # one common exactly linear drift
    st = DriftState(window=5)
    fill(st, mean_at, range(4, NOW + 1))
    recs = records_at(mean_at, 7)
    h = ReplayAdjuster.bandwidth(current_means(mean_at))
    d_rate, _, ok_r = ReplayAdjuster("rate").estimate(st, recs, NOW, h, torch.device("cpu"))
    d_pair, _, ok_p = ReplayAdjuster("pair").estimate(st, recs, NOW, h, torch.device("cpu"))
    assert ok_r.all() and ok_p.all()
    assert torch.allclose(d_rate, d_pair, atol=1e-4)


def test_not_compensated_without_samples_or_weight():
    st = DriftState(window=5)
    mu = torch.randn(D)
    st.observe("a", 0, 5, mu)  # a single report: no rate sample
    rec = [PrototypeRecord(class_id=0, mu=mu, sigma=torch.ones(D), support_count=1, source_id="a", round_idx=5)]
    adj = ReplayAdjuster(drift_comp="rate")
    _, _, ok = adj.estimate(st, rec, 8, 1.0, torch.device("cpu"))
    assert not ok.any()
    st.observe("a", 0, 6, mu + 1.0)
    far = [PrototypeRecord(class_id=0, mu=mu + 1e3, sigma=torch.ones(D), support_count=1, source_id="b", round_idx=5)]
    _, _, ok = adj.estimate(st, far, 8, 1.0, torch.device("cpu"))
    assert not ok.any()  # kernel weight underflows below dc_min_weight


def test_window_and_memory_size():
    st = DriftState(window=3)
    for t in range(10):
        st.observe("a", 0, t, torch.zeros(D))
    assert [s for s, _ in st.hist[("a", 0)]] == [7, 8, 9]
    assert st.nbytes() == 3 * (D * 4 + 8)


def test_c2_limits():
    adj = ReplayAdjuster(alpha_mode="optimal", alpha_min=0.05)
    a, fb = adj.optimal_alpha(b2=1e12, v=1.0, n=10, fixed=0.3)
    assert not fb and a == pytest.approx(1.0, abs=1e-9)
    a, fb = adj.optimal_alpha(b2=1.0, v=1e12, n=1, fixed=0.3)
    assert not fb and a == 0.05
    assert adj.optimal_alpha(None, 1.0, 5, 0.3) == (0.3, True)
    assert adj.optimal_alpha(1.0, 1.0, 0, 0.3) == (0.3, True)


def test_c2_alpha_star_minimises_mse_within_one_percent_of_best_grid():
    g = torch.Generator().manual_seed(1)
    d, n, sigma2, bias = 8, 4, 2.0, 1.5
    v = d * sigma2
    b2 = bias ** 2
    M = 200000
    mu = torch.zeros(d)
    u = torch.zeros(d)
    u[0] = 1.0
    cur = mu + (sigma2 / n) ** 0.5 * torch.randn(M, d, generator=g)  # noisy current mean
    mem = mu + bias * u  # stale memory mean with a fixed bias
    mse = lambda a: float(((a * cur + (1 - a) * mem - mu).pow(2).sum(1)).mean())
    adj = ReplayAdjuster(alpha_mode="optimal", alpha_min=0.0)
    a_star, _ = adj.optimal_alpha(b2, v, n, 0.3)
    assert a_star == pytest.approx(b2 / (b2 + v / n))
    grid = [i / 20 for i in range(21)]
    assert mse(a_star) <= min(mse(a) for a in grid) * 1.01


def _memory_with_state(mean_at, adj, stamps=range(4, NOW + 1), window=5):
    mem = EpisodicPrototypeMemory(max_size=10000, max_age=50, drift_window=window)
    for t in stamps:
        for l in range(N_SRC):
            for j in range(N_CLS):
                mem.add(PrototypeRecord(class_id=j, mu=mean_at(l, j, t), sigma=torch.ones(D), support_count=5,
                                        source_id=f"s{l}", round_idx=t, age=NOW - t))
    return mem


def test_mix_with_c1_c2_runs_and_logs():
    mean_at, _ = world(rate_noise=0.05)
    adj = ReplayAdjuster(drift_comp="rate", alpha_mode="optimal")
    mem = _memory_with_state(mean_at, adj)
    cur_p = {j: mean_at(0, j, NOW) for j in range(N_CLS)}
    cur_d = {j: torch.ones(D) for j in range(N_CLS)}
    stats, trace = {}, {"classes": [], "comp": []}
    p, dd = mix_current_and_memory(cur_p, cur_d, mem, alpha=0.3, top_k=5, adjuster=adj,
                                   support={j: 4 for j in range(N_CLS)}, now=NOW, stats=stats, trace=trace)
    assert set(p) == set(range(N_CLS))
    assert len(stats["alphas"]) == N_CLS and 0.05 <= min(stats["alphas"]) and max(stats["alphas"]) <= 1.0
    assert stats["fallback"] == 0 and stats["comp"] > 0
    assert len(trace["classes"]) == N_CLS
    # memory itself is never modified by compensation
    assert torch.equal(mem.get_by_class(0)[0].mu, mean_at(0, 0, 4))


def test_mix_off_is_identical_to_original_arithmetic():
    mean_at, _ = world(rate_noise=0.05)
    mem = _memory_with_state(mean_at, None)
    cur_p = {j: mean_at(0, j, NOW) for j in range(N_CLS)}
    cur_d = {j: torch.ones(D) for j in range(N_CLS)}
    a = mix_current_and_memory(cur_p, cur_d, mem, alpha=0.3, top_k=5)
    b = mix_current_and_memory(cur_p, cur_d, mem, alpha=0.3, top_k=5, adjuster=ReplayAdjuster(), now=NOW)
    for j in range(N_CLS):
        assert torch.equal(a[0][j], b[0][j]) and torch.equal(a[1][j], b[1][j])
