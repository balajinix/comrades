"""Tests for harness-enforced retry at the tool boundary (faults.inject_faults)."""
from __future__ import annotations

import asyncio
import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.getenv("COMRADES_EVAL_DIR", str(Path(__file__).resolve().parents[1])))

import faults  # noqa: E402
from faults import inject_faults  # noqa: E402
from policy import FAULT_CATALOG, RetryPolicy  # noqa: E402


def _call(n: int = 1) -> list:
    """Make n tool calls through the patched boundary; return results or exceptions."""
    async def go():
        out = []
        for _ in range(n):
            try:
                out.append(await faults._react.call_tool("t", {}, "u"))
            except Exception as exc:  # noqa: BLE001 - faults raise by design
                out.append(exc)
        return out
    return asyncio.run(go())


def test_inject_faults_never_retries_auth_expiry():
    # Arrange: every attempt faults with a 401
    with inject_faults(rate=1.0, types=["auth_expiry"], seed=1,
                       policy=RetryPolicy()) as stats:
        # Act
        (res,) = _call()
    # Assert
    assert "401" in res["error"] and stats["attempted"] == 1 and stats["retried"] == 0


def test_inject_faults_caps_retries_at_three_attempts():
    with inject_faults(rate=1.0, types=["server_5xx"], seed=1,
                       policy=RetryPolicy()) as stats:
        (res,) = _call()
    assert res["status"] == "error" and stats["attempted"] == 3 and stats["retried"] == 2


def test_inject_faults_retries_raised_transient_faults():
    with inject_faults(rate=1.0, types=["timeout"], seed=1, policy=RetryPolicy()) as stats:
        (res,) = _call()
    assert isinstance(res, TimeoutError) and stats["attempted"] == 3


def test_inject_faults_retry_recovers_when_a_later_attempt_succeeds():
    # Find a seed whose first draw faults and second does not, at rate 0.5.
    seed = next(s for s in range(1000)
                if (r := random.Random(s)).random() < 0.5 and (r.choice(["server_5xx"]) or 1)
                and r.random() >= 0.5)
    with inject_faults(rate=0.5, types=["server_5xx"], seed=seed,
                       policy=RetryPolicy()) as stats:
        (res,) = _call()
    assert res["status"] == "ok" and stats["attempted"] == 2 and stats["retried"] == 1


def test_inject_faults_default_is_fail_fast_and_unchanged():
    # Without a policy the draw sequence must match the original wrapper exactly.
    with inject_faults(rate=0.5, seed=7) as stats:
        got = _call(20)
    rng, expected = random.Random(7), []
    kinds = list(faults.FAULT_TYPES)
    for _ in range(20):
        if rng.random() < 0.5:
            expected.append(rng.choice(kinds))
        else:
            expected.append("ok")
    seen = ["ok" if isinstance(r, dict) and r.get("status") == "ok" else "fault" for r in got]
    assert seen == ["ok" if e == "ok" else "fault" for e in expected]
    assert stats["attempted"] == 20 and stats["retried"] == 0


@pytest.mark.parametrize("kind", list(FAULT_CATALOG))
def test_fault_names_match_policy_catalog(kind):
    assert kind in faults.FAULT_TYPES
