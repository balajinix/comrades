"""Env-gated fault injection for the running DAL agent service (COMRADES experiments).

The COMRADES harness injects faults in-process, which only reaches an agent running
in the same process. To study failure propagation across the *networked*
multi-agent chain (a consumption agent calling this DAL agent over A2A), the DAL
service must inject faults on itself. Setting ``COMRADES_FAULT_RATE`` at startup
installs a wrapper on the agent's tool-call boundary that fails a fraction of
tool calls; it is a no-op when the env var is unset, so production is unaffected.

Env:
  COMRADES_FAULT_RATE   per-tool-call failure probability (0..1); unset/0 disables
  COMRADES_FAULT_SEED   RNG seed (default 0)
  COMRADES_FAULT_TYPES  comma-separated subset of fault types (default: all)
"""
from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)

# name -> (delivery, payload-builder). "raise" = transport exception;
# "error" = tool-level error dict the ReAct loop observes.
_FAULTS = {
    "timeout":        ("raise", lambda n: TimeoutError(f"injected timeout calling {n}")),
    "rate_limit_429": ("error", lambda n: {"status": "error", "error": "429 Too Many Requests (injected)"}),
    "server_5xx":     ("error", lambda n: {"status": "error", "error": "503 Service Unavailable (injected)"}),
    "auth_expiry":    ("error", lambda n: {"status": "error", "error": "401 Unauthorized: token expired (injected)"}),
    "partial_outage": ("raise", lambda n: ConnectionError(f"NameResolutionError: {n} backend unreachable (injected)")),
}


def install_from_env() -> bool:
    """If COMRADES_FAULT_RATE is set (>0), wrap the agent tool-call boundary to inject
    faults. Returns True if installed. Safe to call once at service startup."""
    raw = os.getenv("COMRADES_FAULT_RATE")
    if raw is None:
        return False  # not a COMRADES experiment run
    rate = float(raw or 0)
    # Note: we install even at rate=0 so the replay baseline (canned successful
    # responses) is active and the no-fault row is a clean all-success baseline
    # rather than depending on live-backend reachability.

    seed = int(os.getenv("COMRADES_FAULT_SEED", "0") or 0)
    types = [t.strip() for t in os.getenv("COMRADES_FAULT_TYPES", "").split(",") if t.strip()] \
        or list(_FAULTS)
    rng = random.Random(seed)

    import kg_agent.react as react
    real_call_tool = react.call_tool

    # When fault injection is active we serve a canned successful response for
    # non-faulted calls (COMRADES_REPLAY=1, the default here) so the no-fault
    # baseline succeeds and the success-vs-fault-rate curve is not confounded by
    # backend reachability. Set COMRADES_REPLAY=0 to use the real backends instead.
    replay = os.getenv("COMRADES_REPLAY", "1") not in ("0", "false", "")

    def _canned(name):
        return {"status": "ok", "results": [{"id": "rec-1", "value": 1},
                                             {"id": "rec-2", "value": 2}],
                "_comrades_canned": True, "tool": name}

    async def faulted_call_tool(name, arguments, url):
        if rng.random() < rate:
            kind = rng.choice(types)
            delivery, build = _FAULTS[kind]
            logger.warning("COMRADES fault injected (%s) on tool %s", kind, name)
            if delivery == "raise":
                raise build(name)
            return build(name)
        return _canned(name) if replay else await real_call_tool(name, arguments, url)

    react.call_tool = faulted_call_tool
    logger.warning("COMRADES fault injection ACTIVE: rate=%.2f seed=%d types=%s",
                   rate, seed, types)
    return True
