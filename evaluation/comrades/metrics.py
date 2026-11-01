"""Metric capture + aggregation for the COMRADES harness.

Two concerns:
1. Token capture. kg_agent drives the LLM via litellm but does not persist token
   usage. We register a litellm success callback that sums usage within a
   context manager, so any mechanism can wrap a run and read tokens back. This is
   best-effort: if the provider/callback yields nothing, tokens stay None and the
   scorer treats them as missing rather than zero.
2. Aggregation. Roll RunResults up per (mechanism[, model]) into the success /
   latency / tokens / messages / recovery summary the RQ1 table reports.
"""
from __future__ import annotations

import statistics
from contextlib import contextmanager
from typing import Any, Iterable

from runner import RunResult


# --- 1. token capture -------------------------------------------------------

# Module-global accumulator (NOT thread-local): litellm runs async success
# callbacks on a separate thread, so a thread-local "active" gate set on the
# caller's thread would never be seen. The harness runs one request at a time,
# so a single global accumulator is correct here.
_TOK = {"active": False, "prompt": 0, "completion": 0, "total": 0}


def _accumulate(response_obj) -> None:
    """Add one response's usage to the active accumulator."""
    if not _TOK["active"]:
        return
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        return

    def _g(name):
        if isinstance(usage, dict):
            return usage.get(name) or 0
        return getattr(usage, name, 0) or 0

    _TOK["prompt"] += _g("prompt_tokens")
    _TOK["completion"] += _g("completion_tokens")
    _TOK["total"] += _g("total_tokens")


_callback_registered = False


def _ensure_callback() -> None:
    """Register a litellm CustomLogger. The plain async success_callback does NOT
    fire for acompletion in this litellm build, but a CustomLogger's
    async_log_success_event does (usage is on response_obj either way)."""
    global _callback_registered
    if _callback_registered:
        return
    try:
        import litellm
        from litellm.integrations.custom_logger import CustomLogger

        class _TokenLogger(CustomLogger):
            def log_success_event(self, kwargs, response_obj, start_time, end_time):
                _accumulate(response_obj)

            async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
                _accumulate(response_obj)

        litellm.callbacks = list(litellm.callbacks or []) + [_TokenLogger()]
        _callback_registered = True
    except Exception:
        pass  # litellm not importable; tokens stay None


@contextmanager
def capture_tokens():
    """Context manager yielding a dict filled with prompt/completion/total tokens
    consumed by litellm calls inside the block (sequential use only)."""
    _ensure_callback()
    _TOK.update(active=True, prompt=0, completion=0, total=0)
    box: dict[str, int] = {}
    try:
        yield box
    finally:
        _TOK["active"] = False
        box["prompt"] = _TOK["prompt"]
        box["completion"] = _TOK["completion"]
        box["total"] = _TOK["total"]


# --- 2. aggregation ---------------------------------------------------------

def _mean(xs: list[float]) -> float | None:
    return round(statistics.mean(xs), 3) if xs else None


def aggregate(results: Iterable[RunResult]) -> list[dict[str, Any]]:
    """Group RunResults by (mechanism, model) and summarize for the RQ1 table."""
    groups: dict[tuple[str, str | None], list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.mechanism, r.model), []).append(r)

    summary = []
    for (mech, model), rs in sorted(groups.items(), key=lambda k: str(k[0])):
        n = len(rs)
        ok = [r for r in rs if r.success]
        # recovery rate over runs that hit >=1 transient failure
        recoverable = [r for r in rs if r.recovered or (not r.success and r.error)]
        toks = [r.tokens_total for r in rs if r.tokens_total]
        summary.append({
            "mechanism": mech,
            "model": model,
            "n": n,
            "success_rate": round(len(ok) / n, 3) if n else None,
            "latency_s_mean": _mean([r.latency_s for r in rs]),
            "tokens_mean": _mean([float(t) for t in toks]) if toks else None,
            "messages_mean": _mean([float(r.messages) for r in rs]),
            "recovery_rate": (round(sum(r.recovered for r in recoverable) / len(recoverable), 3)
                              if recoverable else None),
        })
    return summary
