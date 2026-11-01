"""Classify a run's tool calls into necessary recovery vs. redundant thrash.

We reuse the agent's OWN notion of "same intent": ``kg_agent.react._fingerprint_args``
maps paraphrased/equivalent arguments to the same fingerprint (it is what the
loop-detector uses). A tool call is counted as:

- \emph{necessary}  : the first time a given (action, fingerprint) is issued, OR
                      a call that returned usable data (no error);
- \emph{redundant}  : a repeat of an (action, fingerprint) already seen, OR a
                      call that errored and was re-issued with the same intent
                      --- the recovery thrash a failure induces.

This is a post-hoc, deterministic pass over the persisted trace, so it does not
perturb timing or token measurements.
"""
from __future__ import annotations

import sys
from pathlib import Path

_DAL_AGENT = Path(__file__).resolve().parents[2] / "agents" / "dal_agent"
if str(_DAL_AGENT) not in sys.path:
    sys.path.insert(0, str(_DAL_AGENT))

from kg_agent.react import _fingerprint, _fingerprint_args  # noqa: E402

_ERROR_MARKERS = ("status=error", "ERROR:", "Failed to", "NameResolution",
                  "Unauthorized", "429", "503", "401", "injected")


def _step_fingerprint(step: dict) -> str:
    """Fingerprint a trace step by its action + action_input, matching the
    loop-detector's same-intent test as closely as the trace allows."""
    action = step.get("action") or ""
    ai = step.get("action_input")
    # action_input in the trace is the raw string; _fingerprint handles strings.
    return f"{action}::{_fingerprint(ai if isinstance(ai, str) else str(ai))}"


def _errored(step: dict) -> bool:
    tr = step.get("tool_result")
    if isinstance(tr, dict) and str(tr.get("status", "")).lower() == "error":
        return True
    obs = step.get("observation") or ""
    return any(m in obs for m in _ERROR_MARKERS)


def classify_trace(trace: list[dict]) -> dict:
    """Return counts: total tool steps, necessary, redundant, and the redundant
    fraction. Steps with no action (pure reasoning / final answer) are ignored.
    """
    seen: set[str] = set()
    necessary = redundant = total = errored = 0
    for step in trace or []:
        if not step.get("action"):
            continue
        total += 1
        if _errored(step):
            errored += 1
        fp = _step_fingerprint(step)
        if fp in seen:
            # Re-issuing an already-tried intent (the classic failure-recovery
            # thrash: same query, fails again) is redundant communication.
            redundant += 1
        else:
            necessary += 1
            seen.add(fp)
    return {
        "tool_steps": total,
        "errored_steps": errored,
        "necessary": necessary,
        "redundant": redundant,
        "redundant_frac": round(redundant / total, 3) if total else None,
    }
