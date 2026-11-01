"""Fault injection and retry placement for LLM agents, on a virtual clock.

Three ways to handle transient data-access faults, compared on the same tools:

- ``native``: faults reach the model as tool errors; no policy, no way to wait
  (what deployed agents do today).
- ``prose``: the model gets a written retry policy (a skill-style playbook) and a
  ``wait`` tool; whether it backs off is up to the model.
- ``coded``: the DAL retries inside the tool with exponential backoff and jitter;
  the model sees only the final outcome.

Time is virtual: each model turn and each call advance a shared clock, and ``wait``
advances it without sleeping, so bursts can be outlasted and runs stay fast.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .faults import Fault, FaultProcess
from .retry import RetryPolicy

PROSE_POLICY = (
    "Data-access failures are usually transient (timeouts, 429 rate limits, 503 outages) "
    "and often come in bursts. When a tool returns an error: (1) never retry an "
    "auth_expiry error; (2) otherwise call the wait tool before retrying the SAME call, "
    "waiting 1 s, then 2 s, then 4 s, then 8 s (honour retry_after if it is larger); "
    "(3) give up after 5 attempts of the same call and say the data is unavailable.")

WAIT_DESCRIPTOR = {
    "name": "wait",
    "description": "Pause for the given number of seconds before your next call.",
    "inputSchema": {"type": "object",
                    "properties": {"seconds": {"type": "number"}},
                    "required": ["seconds"]},
}


@dataclass
class VirtualClock:
    """Shared simulated time for one run."""

    now: float = 0.0

    def advance(self, seconds: float) -> None:
        """Move time forward by ``seconds`` (negative values are ignored)."""
        self.now += max(0.0, float(seconds))


@dataclass
class FaultStats:
    """What the fault layer saw during one run."""

    attempts: int = 0
    faults: int = 0
    faults_in_burst: int = 0
    waits: int = 0
    waited_s: float = 0.0
    dal_retries: int = 0
    log: list[dict] = field(default_factory=list)


def _error_payload(fault: Fault) -> dict:
    payload = {"status": "error", "fault_type": fault.kind,
               "error": f"{fault.kind} (transient data-access failure)"}
    if fault.retry_after is not None:
        payload["retry_after"] = round(fault.retry_after, 1)
    return payload


def inject(tools: dict[str, Callable[..., Any]], process: FaultProcess, clock: VirtualClock,
           stats: FaultStats, backend: str = "store", latency_s: float = 0.2,
           retry: RetryPolicy | None = None, seed: int = 0) -> dict[str, Callable[..., Any]]:
    """Wrap tools so each call may fail according to ``process`` at the current time.

    Args:
        tools: Tool callables (already metered, so every attempt is counted).
        process: Fault process (Bernoulli or Gilbert–Elliott).
        clock: The run's virtual clock.
        stats: Accumulates fault, wait and retry counts.
        backend: Backend name used for the fault chain.
        latency_s: Virtual service time per attempt.
        retry: If given, the DAL retries inside the tool (the ``coded`` arm).
        seed: Seed for jitter.

    Returns:
        Wrapped tools with the same names.
    """
    rng = random.Random(f"dal-retry:{seed}")

    def wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(**kwargs: Any) -> Any:
            attempt = 0
            while True:
                attempt += 1
                stats.attempts += 1
                clock.advance(latency_s)
                fault = process.draw(backend, clock.now)
                if fault is None:
                    return fn(**kwargs)
                stats.faults += 1
                stats.faults_in_burst += int(fault.in_burst)
                stats.log.append({"t": round(clock.now, 2), "tool": name, "fault": fault.kind})
                if retry is None or not retry.should_retry(fault, attempt):
                    return _error_payload(fault)
                stats.dal_retries += 1
                clock.advance(retry.wait(fault, attempt, rng))
        return call

    return {n: wrap(n, f) for n, f in tools.items()}


def wait_tool(clock: VirtualClock, stats: FaultStats, cap_s: float = 120.0) -> Callable[..., dict]:
    """The ``wait`` tool for the prose arm: advances virtual time, capped per call."""
    def wait(seconds: float = 1.0) -> dict:
        s = min(cap_s, max(0.0, float(seconds)))
        clock.advance(s)
        stats.waits += 1
        stats.waited_s += s
        return {"status": "ok", "waited_s": s}
    return wait
