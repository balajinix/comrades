"""Tests for the push vs pull sync simulator."""
from __future__ import annotations

import pytest

from dal_lab.events import STRATEGIES, SyncConfig, change_stream, simulate_sync
from dal_lab.faults import Bernoulli, GilbertElliott, Outage

CFG = SyncConfig(n_entities=500, change_rate=0.05, horizon_s=6 * 3600, seed=1)


def test_change_stream_burst_keeps_volume():
    flat = change_stream(SyncConfig(change_rate=0.1, seed=3))
    bursty = change_stream(SyncConfig(change_rate=0.1, burst_factor=20, seed=3))
    assert abs(len(bursty) - len(flat)) / len(flat) < 0.25


def test_change_stream_stops_before_cooldown():
    cs = change_stream(CFG)
    assert max(c.t for c in cs) < CFG.horizon_s - CFG.cooldown_s


def test_no_faults_every_strategy_converges():
    for s in STRATEGIES:
        r = simulate_sync(CFG, s, Bernoulli(0.0)).summary()
        assert r["delay_missed_frac"] == 0.0, s


def test_poll_delay_is_about_half_the_interval():
    r = simulate_sync(CFG, "poll_delta", Bernoulli(0.0), interval_s=900).summary()
    assert 350 < r["delay_mean_s"] < 550


def test_push_is_fresh_without_faults():
    r = simulate_sync(CFG, "push_full", Bernoulli(0.0)).summary()
    assert r["delay_p95_s"] < 1 and r["lost_events"] == 0


def test_outage_loses_push_events_but_resync_repairs_them():
    push = simulate_sync(CFG, "push_full", Outage(3600, 5400)).summary()
    resync = simulate_sync(CFG, "push_resync", Outage(3600, 5400)).summary()
    assert push["lost_events"] > 0 and push["delay_missed_frac"] > 0.01
    assert resync["delay_missed_frac"] == 0.0


def test_polling_is_level_triggered_under_bursts():
    ge = GilbertElliott.matched(0.05, mean_bad=600, seed=2)
    for s in ("poll_full", "poll_delta"):
        assert simulate_sync(CFG, s, ge).summary()["delay_missed_frac"] == 0.0, s


def test_deletes_without_tombstones_need_a_full_snapshot():
    cfg = SyncConfig(n_entities=500, change_rate=0.05, horizon_s=6 * 3600,
                     delete_frac=0.2, tombstones=False, seed=5)
    miss = {s: simulate_sync(cfg, s, Bernoulli(0.0)).summary()["del_missed_frac"]
            for s in STRATEGIES}
    assert miss["poll_delta"] > 0.5 and miss["push_full"] > 0.5
    assert miss["poll_full"] == 0.0 and miss["push_resync"] == 0.0


def test_gated_polling_wakes_only_on_relevant_changes():
    cfg = SyncConfig(n_entities=500, change_rate=0.001, relevant_frac=0.1,
                     horizon_s=24 * 3600, seed=6)
    r = simulate_sync(cfg, "poll_delta", Bernoulli(0.0), interval_s=300).summary()
    assert r["wakeups"] == 288 and r["gated_wakeups"] < 20


def test_retries_reorder_a_naive_consumer():
    cfg = SyncConfig(n_entities=20, change_rate=0.5, horizon_s=3 * 3600, seed=4)
    r = simulate_sync(cfg, "push_full",
                      GilbertElliott.matched(0.2, mean_bad=30, seed=4)).summary()
    assert r["reorders"] > 0


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        simulate_sync(CFG, "carrier_pigeon", Bernoulli(0.0))


def test_delta_poll_keeps_a_needed_change_masked_by_a_later_one():
    cfg = SyncConfig(n_entities=50, change_rate=0.05, relevant_frac=0.5,
                     horizon_s=6 * 3600, seed=7)
    r = simulate_sync(cfg, "poll_delta", Bernoulli(0.0), interval_s=900).summary()
    assert r["delay_missed_frac"] == 0.0


def test_delta_poll_reads_a_change_at_time_zero():
    from dal_lab.events import Change

    cfg = SyncConfig(n_entities=3, horizon_s=3600)
    stream = [Change(0.0, 0, False, True), Change(10.0, 1, False, True)]
    r = simulate_sync(cfg, "poll_delta", Bernoulli(0.0), interval_s=60,
                      changes=stream).summary()
    assert r["delay_missed_frac"] == 0.0
