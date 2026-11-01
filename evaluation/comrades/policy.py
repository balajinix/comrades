"""Shared fault model + retry policy for the control-discipline experiment.

This module is the single source of truth that makes the workflow / skill /
free-model mechanisms *mechanism-equivalent*: they all draw faults from the same
``FAULT_CATALOG`` and all retry under the same ``RetryPolicy``. The ONLY thing
that varies across the retry arms is whether the policy is dispatched by
deterministic code (workflow), followed from a natural-language playbook (skill,
which may deviate), or absent (free-model). Keeping the policy here — not baked
into each runner — is what neutralizes the policy-sophistication confound.

Faults are injected at the fault-proxy (``harness/fault_proxy.py``), the one
boundary every mechanism crosses, so communication is counted identically.
Through a real MCP proxy every fault surfaces as a tool error carrying its type
string; the ``delivery`` field records the original channel (transport-exception
vs tool-error) for documentation, but the proxy homogenizes delivery on the wire.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


# Fault catalog: name -> (delivery, retryable, message template). Mirrors the
# in-process catalog in faults.py so the two injection paths stay aligned.
FAULT_CATALOG: dict[str, dict] = {
    "timeout":        {"delivery": "raise", "retryable": True,
                       "message": "injected timeout calling {tool}"},
    "rate_limit_429": {"delivery": "error", "retryable": True,
                       "message": "429 Too Many Requests (injected)"},
    "server_5xx":     {"delivery": "error", "retryable": True,
                       "message": "503 Service Unavailable (injected)"},
    "auth_expiry":    {"delivery": "error", "retryable": False,
                       "message": "401 Unauthorized: token expired (injected)"},
    "partial_outage": {"delivery": "raise", "retryable": True,
                       "message": "NameResolutionError: {tool} backend unreachable (injected)"},
}

ALL_FAULT_TYPES = list(FAULT_CATALOG)

# Faults a retry policy treats as worth re-attempting. Auth expiry is excluded:
# an immediate blind retry will not clear an expired token, so retrying it is
# thrash by construction. This set is part of the SHARED policy — identical for
# the workflow and skill arms.
DEFAULT_RETRYABLE = frozenset(
    name for name, meta in FAULT_CATALOG.items() if meta["retryable"]
)


def pick_fault(rng: random.Random, allowed: list[str] | None = None) -> tuple[str, str]:
    """Draw a fault type from ``rng`` and render its message. Returns (type, message)."""
    kinds = allowed or ALL_FAULT_TYPES
    kind = rng.choice(kinds)
    return kind, FAULT_CATALOG[kind]["message"]


def fault_type_of(message: str) -> str | None:
    """Recover the fault type from an injected error message (for log/agent traces)."""
    if not message:
        return None
    for name, meta in FAULT_CATALOG.items():
        token = meta["message"].split("{")[0].strip() or name
        if token and token in message:
            return name
    return "injected" if "injected" in message else None


@dataclass(frozen=True)
class RetryPolicy:
    """The one shared retry policy. ``max_attempts`` counts the FIRST try plus
    retries (so max_attempts=3 means up to 2 retries). ``retryable`` names the
    fault types worth re-attempting; non-retryable faults stop immediately.
    ``backoff_s`` is recorded and (optionally) slept; default 0 keeps the canned
    runs fast and latency measurement clean."""

    max_attempts: int = 3
    retryable: frozenset[str] = field(default_factory=lambda: DEFAULT_RETRYABLE)
    backoff_s: float = 0.0

    def is_retryable(self, fault_type: str | None) -> bool:
        return bool(fault_type) and fault_type in self.retryable

    def as_dict(self) -> dict:
        return {"max_attempts": self.max_attempts,
                "retryable": sorted(self.retryable),
                "backoff_s": self.backoff_s}

    def to_prose(self) -> str:
        """Render the policy as the natural-language playbook the skill follows.
        The skill arm consumes THIS string, so its authored policy is provably the
        same one the workflow executes in code."""
        retry_list = ", ".join(sorted(self.retryable))
        return (
            f"Failure-handling policy: when a tool call returns an injected error, "
            f"retry the SAME call up to {self.max_attempts - 1} more time(s) "
            f"(at most {self.max_attempts} attempts total) ONLY if the error is one of: "
            f"{retry_list}. If the error is an auth/401 expiry, do NOT retry — stop and "
            f"report failure. Never re-issue a call that already succeeded."
        )


# The single policy instance the retry arms share.
SHARED_RETRY = RetryPolicy()
# A fail-fast policy: one attempt, no retries (the under-communication end).
FAIL_FAST = RetryPolicy(max_attempts=1, retryable=frozenset())
