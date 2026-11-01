"""Tests for fault processes, retry policies and the client simulator."""
from __future__ import annotations

import random

import pytest

from dal_lab.faults import Bernoulli, Fault, GilbertElliott
from dal_lab.retry import ARMS, RetryBudget, RetryPolicy
from dal_lab.sim import SimConfig, simulate, summarize


def test_matched_marginal_rate_equals_target():
    # Arrange / Act
    ge = GilbertElliott.matched(rate=0.3, mean_bad=10.0)
    # Assert
    assert ge.marginal_rate == pytest.approx(0.3)
    assert ge.stationary_bad == pytest.approx(0.3)


def test_matched_rejects_rate_above_e_bad():
    with pytest.raises(ValueError):
        GilbertElliott.matched(rate=0.9, mean_bad=10.0, e_bad=0.8)


def test_ge_draw_empirical_rate_matches_marginal():
    # Arrange: calls at random times over a long horizon
    ge = GilbertElliott.matched(rate=0.3, mean_bad=5.0, seed=1)
    rng = random.Random(0)
    times = sorted(rng.uniform(0, 200_000) for _ in range(40_000))
    # Act
    faults = sum(ge.draw("b", t) is not None for t in times)
    # Assert
    assert faults / len(times) == pytest.approx(0.3, abs=0.02)


def test_ge_faults_cluster_in_time():
    # Arrange: two calls 0.1 s apart are far more likely to share an outcome than
    # under independent faults (lag-1 correlation of outcomes).
    ge = GilbertElliott.matched(rate=0.3, mean_bad=10.0, seed=2)
    pairs, both = 0, 0
    for k in range(20_000):
        t = k * 50.0
        a = ge.draw("b", t) is not None
        b = ge.draw("b", t + 0.1) is not None
        pairs += a
        both += a and b
    # Assert: P(fail | previous failed) ≈ 1, far above the marginal 0.3
    assert both / pairs > 0.9


def test_ge_burst_shares_one_fault_type():
    ge = GilbertElliott.matched(rate=0.5, mean_bad=50.0, seed=3)
    kinds = {f.kind for t in range(0, 40) if (f := ge.draw("b", 1000 + t * 0.5))}
    assert len(kinds) <= 2  # at most one burst boundary in a 20 s window


def test_ge_same_seed_same_schedule():
    a = GilbertElliott.matched(0.3, 10.0, seed=7)
    b = GilbertElliott.matched(0.3, 10.0, seed=7)
    ts = [i * 0.7 for i in range(2000)]
    assert [a.in_bad_state("x", t) for t in ts] == [b.in_bad_state("x", t) for t in ts]


def test_ge_backend_schedule_independent_of_other_backends():
    # Calls to another backend must not shift this backend's bursts.
    a = GilbertElliott.matched(0.3, 10.0, seed=7)
    b = GilbertElliott.matched(0.3, 10.0, seed=7)
    for t in range(500):
        b.in_bad_state("other", t * 1.3)
    ts = [i * 0.7 for i in range(2000)]
    assert [a.in_bad_state("x", t) for t in ts] == [b.in_bad_state("x", t) for t in ts]


def test_bernoulli_rate():
    proc = Bernoulli(0.25, seed=1)
    hits = sum(proc.draw("b", 0.0) is not None for _ in range(20_000))
    assert hits / 20_000 == pytest.approx(0.25, abs=0.015)


def test_retry_policy_wait_exponential_capped():
    pol = RetryPolicy("x", backoff="exponential", base_s=1.0, cap_s=5.0)
    f = Fault("server_5xx")
    waits = [pol.wait(f, a, random.Random(0)) for a in range(1, 6)]
    assert waits == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_retry_policy_full_jitter_within_bounds():
    pol = RetryPolicy("x", backoff="exponential", base_s=1.0, jitter="full")
    rng = random.Random(0)
    waits = [pol.wait(Fault("timeout"), 3, rng) for _ in range(500)]
    assert all(0.0 <= w <= 4.0 for w in waits)
    assert max(waits) - min(waits) > 3.0


def test_retry_policy_honors_retry_after():
    pol = ARMS["exp_jitter_retry_after"]
    w = pol.wait(Fault("rate_limit_429", retry_after=10.0), 1, random.Random(0))
    assert w >= 10.0


def test_retry_policy_never_retries_auth():
    assert not ARMS["immediate"].should_retry(Fault("auth_expiry"), 1)


def test_retry_budget_caps_retries():
    budget = RetryBudget(ratio=0.1, floor=2, first_attempts=10)
    allowed = 0
    while budget.allow():
        budget.retries += 1
        allowed += 1
    assert allowed == 3


def test_simulate_no_faults_all_succeed():
    cfg = SimConfig(clients=2, horizon_s=300)
    res = simulate(cfg, Bernoulli(0.0), ARMS["immediate"])
    s = summarize(res, cfg, Bernoulli(0.0))
    assert s["success_rate"] == 1.0
    assert s["amplification"] == pytest.approx(1.0)


def test_simulate_fail_fast_success_matches_independent_rate():
    # With independent faults and no retry, a 4-call task succeeds w.p. (1-p)^4.
    cfg = SimConfig(clients=1, horizon_s=20_000)
    proc = Bernoulli(0.2, seed=3)
    s = summarize(simulate(cfg, proc, ARMS["fail_fast"]), cfg, proc)
    assert s["success_rate"] == pytest.approx(0.8 ** 4, abs=0.03)


def test_simulate_backoff_beats_immediate_under_long_bursts():
    cfg = SimConfig(clients=1, horizon_s=20_000)
    imm = GilbertElliott.matched(0.3, mean_bad=10.0, seed=4)
    exp = GilbertElliott.matched(0.3, mean_bad=10.0, seed=4)
    s_imm = summarize(simulate(cfg, imm, ARMS["immediate"]), cfg, imm)
    s_exp = summarize(simulate(cfg, exp, ARMS["exponential"]), cfg, exp)
    assert s_exp["success_rate"] > s_imm["success_rate"]


def test_simulate_is_deterministic():
    cfg = SimConfig(clients=3, horizon_s=2000, seed=5)
    a = summarize(simulate(cfg, GilbertElliott.matched(0.3, 10.0, seed=1), ARMS["exp_jitter"]),
                  cfg, GilbertElliott.matched(0.3, 10.0, seed=1))
    b = summarize(simulate(cfg, GilbertElliott.matched(0.3, 10.0, seed=1), ARMS["exp_jitter"]),
                  cfg, GilbertElliott.matched(0.3, 10.0, seed=1))
    assert a == b


def test_outage_fails_only_inside_window():
    from dal_lab.faults import Outage
    proc = Outage(5.0, 10.0)
    assert proc.draw("b", 4.9) is None
    assert proc.draw("b", 5.0).kind == "server_5xx"
    assert proc.draw("b", 10.0) is None


def test_sync_start_without_jitter_keeps_clients_in_lockstep():
    # Arrange: every client starts into the same outage; plain exponential backoff
    # keeps their retries aligned, full jitter spreads them.
    from dal_lab.faults import Outage
    cfg = SimConfig(clients=50, horizon_s=60.0, sync_start=True)
    peak = {}
    for arm in ("exponential", "exp_jitter_eqwait"):
        proc = Outage(0.0, 10.0)
        peak[arm] = summarize(simulate(cfg, proc, ARMS[arm]), cfg, proc,
                              herd_window_s=30.0)["peak_rps_after_burst"]
    # Assert
    assert peak["exponential"] > peak["exp_jitter_eqwait"]
