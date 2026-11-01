"""Fault injection at the agent's MCP tool-call boundary.

To study failure-driven communication amplification we inject controlled
failures where real backends fail---at ``kg_agent.react.call_tool``, the single
point through which every tool invocation passes. We monkeypatch the name bound
in the ``react`` module (it does ``from .mcp_client import call_tool``, so
patching ``mcp_client.call_tool`` alone would not take effect).

Faults are drawn from a seeded RNG so a (request, rate, seed) triple is
reproducible. Two delivery modes mirror how real failures reach the ReAct loop:
- error-dict: ``{"status": "error", ...}`` becomes an Observation the model
  reads and tries to recover from (rate limits, auth, partial outage);
- raise: a transport exception the loop catches and wraps (timeouts).
"""
from __future__ import annotations

import random
import sys
from contextlib import contextmanager
from pathlib import Path

_DAL_AGENT = Path(__file__).resolve().parents[2] / "agents" / "dal_agent"
if str(_DAL_AGENT) not in sys.path:
    sys.path.insert(0, str(_DAL_AGENT))

import kg_agent.react as _react  # noqa: E402


# Fault catalog: name -> (delivery, payload-builder). "raise" delivers a
# transport-style exception; "error" delivers a tool-level error dict.
FAULT_TYPES = {
    "timeout":        ("raise", lambda n: TimeoutError(f"injected timeout calling {n}")),
    "rate_limit_429": ("error", lambda n: {"status": "error", "error": "429 Too Many Requests (injected)"}),
    "server_5xx":     ("error", lambda n: {"status": "error", "error": "503 Service Unavailable (injected)"}),
    "auth_expiry":    ("error", lambda n: {"status": "error", "error": "401 Unauthorized: token expired (injected)"}),
    "partial_outage": ("raise", lambda n: ConnectionError(f"NameResolutionError: {n} backend unreachable (injected)")),
}


def _canned(name):
    """Canned successful tool response used in replay mode (so the no-fault
    baseline succeeds without live backends)."""
    return {"status": "ok", "results": [{"id": "rec-1", "value": 1},
                                         {"id": "rec-2", "value": 2}],
            "_comrades_canned": True, "tool": name}


@contextmanager
def inject_faults(rate: float, types: list[str] | None = None, seed: int = 0,
                  replay: bool = True, policy=None):
    """Wrap ``react.call_tool`` so each invocation fails with probability ``rate``.

    ``types`` selects which fault kinds may be drawn (default: all). When
    ``replay`` is True (the default for experiments), non-faulted calls return a
    canned successful response instead of hitting the real backend, so the
    rate=0 baseline succeeds and the success-vs-fault-rate curve is reproducible
    without VPN/live backends. Set ``replay=False`` to use live backends. The
    wrapper records attempted/faulted counts on the yielded ``stats`` dict.

    ``policy`` (a ``policy.RetryPolicy``) adds harness-enforced retry at the tool
    boundary: a call that draws a retryable fault is re-attempted, drawing again from
    the same seeded stream, up to ``policy.max_attempts`` attempts in total, so the
    agent sees a fault only when every attempt fails or the fault is not retryable.
    ``attempted`` then counts every attempt. ``None`` (the default) is fail-fast and
    leaves the call path exactly as it was.
    """
    chosen = types or list(FAULT_TYPES)
    rng = random.Random(seed)
    real_call_tool = _react.call_tool
    stats = {"attempted": 0, "faulted": 0, "by_type": {}, "retried": 0}
    last_kind: list[str | None] = [None]  # fault type of the latest attempt

    async def one_attempt(name, arguments, url):
        stats["attempted"] += 1
        last_kind[0] = None
        if rng.random() < rate:
            stats["faulted"] += 1
            kind = rng.choice(chosen)
            last_kind[0] = kind
            stats["by_type"][kind] = stats["by_type"].get(kind, 0) + 1
            delivery, build = FAULT_TYPES[kind]
            if delivery == "raise":
                raise build(name)
            return build(name)
        return _canned(name) if replay else await real_call_tool(name, arguments, url)

    async def retrying_call_tool(name, arguments, url):
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = await one_attempt(name, arguments, url)
            except Exception:
                if attempt < policy.max_attempts and policy.is_retryable(last_kind[0]):
                    stats["retried"] += 1
                    continue
                raise
            if (last_kind[0] is not None and attempt < policy.max_attempts
                    and policy.is_retryable(last_kind[0])):
                stats["retried"] += 1
                continue
            return result
        return result  # unreachable: the last attempt always returns or raises

    faulted_call_tool = one_attempt if policy is None else retrying_call_tool
    _react.call_tool = faulted_call_tool
    try:
        yield stats
    finally:
        _react.call_tool = real_call_tool
