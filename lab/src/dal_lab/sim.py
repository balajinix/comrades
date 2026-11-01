"""Discrete-event simulation of K agent clients calling shared backends.

Each client runs tasks back to back. A task is a fixed plan of data-access calls
(a coded workflow: the cheapest, most faithful arm, with no LLM in the loop).
Between calls the client "thinks" for a fixed LLM-turn latency. On a fault, the
client's :class:`~dal_lab.retry.RetryPolicy` decides whether and when to retry.
A task fails if a call exhausts its attempts, hits a non-retryable fault, the
retry budget runs out, or the task exceeds its deadline.

Everything runs on a virtual clock, so a 2-hour experiment takes seconds and is
reproducible from the seed. Backends optionally have a capacity (requests per
second); requests beyond it fail with a load-induced 429. That is how
synchronized retries after a burst (a thundering herd) turn into extra failures.
"""
from __future__ import annotations

import heapq
import random
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Generator

from .faults import Fault, FaultProcess, GilbertElliott
from .retry import RetryBudget, RetryPolicy


@dataclass(frozen=True)
class SimConfig:
    """Simulation parameters.

    Attributes:
        clients: Number of concurrent clients (K).
        horizon_s: Virtual seconds to simulate.
        plan: Backends a task calls, in order (a multi-hop task).
        call_latency_s: Service time of one call.
        think_s: LLM turn latency between calls.
        idle_mean_s: Mean idle time between tasks (exponential), de-synchronizing clients.
        deadline_s: Task deadline, as agents enforce wall-clock limits.
        capacity_rps: Per-backend capacity in requests/second (None = unlimited).
        sync_start: All clients start their first task at t=0 (synchronized onset).
        seed: RNG seed for client timing and jitter.
    """

    clients: int = 1
    horizon_s: float = 3600.0
    plan: tuple[str, ...] = ("itsm", "observability", "cost", "itsm")
    call_latency_s: float = 0.2
    think_s: float = 1.0
    idle_mean_s: float = 2.0
    deadline_s: float = 60.0
    capacity_rps: float | None = None
    sync_start: bool = False
    seed: int = 0


@dataclass
class TaskRecord:
    """Outcome of one task."""

    client: int
    start: float
    end: float
    success: bool
    attempts: int
    wasted: int
    reason: str = "ok"


@dataclass
class _Request:
    backend: str


@dataclass
class _Sleep:
    seconds: float


@dataclass
class SimResult:
    """Raw outputs of one simulation run."""

    tasks: list[TaskRecord] = field(default_factory=list)
    requests: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    load_faults: int = 0
    first_attempts: int = 0
    first_attempt_faults: int = 0


class _Capacity:
    """Sliding one-second window of accepted requests per backend."""

    def __init__(self, rps: float | None):
        self.rps = rps
        self.windows: dict[str, deque] = defaultdict(deque)

    def admit(self, backend: str, now: float) -> bool:
        if self.rps is None:
            return True
        window = self.windows[backend]
        while window and window[0] <= now - 1.0:
            window.popleft()
        if len(window) >= self.rps:
            return False
        window.append(now)
        return True


def _client(cid: int, cfg: SimConfig, policy: RetryPolicy, rng: random.Random,
            result: SimResult) -> Generator:
    """One client's lifetime, as a generator of requests and sleeps."""
    budget = RetryBudget(policy.budget) if policy.budget is not None else None
    first_wait = 0.0 if cfg.sync_start else rng.expovariate(1.0 / cfg.idle_mean_s)
    now = yield _Sleep(first_wait)
    while True:
        start, attempts, wasted = now, 0, 0
        success, reason = True, "ok"
        for i, backend in enumerate(cfg.plan):
            if i:
                now = yield _Sleep(cfg.think_s)
            attempt = 0
            while True:
                attempt += 1
                attempts += 1
                if attempt == 1 and budget is not None:
                    budget.first_attempts += 1
                fault, now = yield _Request(backend)
                if attempt == 1:
                    result.first_attempts += 1
                    result.first_attempt_faults += int(fault is not None)
                if fault is None:
                    break
                wasted += int(fault.in_burst)
                if now - start > cfg.deadline_s:
                    success, reason = False, "deadline"
                    break
                if not policy.should_retry(fault, attempt):
                    success = False
                    reason = "non_retryable" if not fault.retryable else "attempts"
                    break
                if budget is not None:
                    if not budget.allow():
                        success, reason = False, "budget"
                        break
                    budget.retries += 1
                delay = policy.wait(fault, attempt, rng)
                if now + delay - start > cfg.deadline_s:
                    success, reason = False, "deadline"
                    break
                if delay > 0:
                    now = yield _Sleep(delay)
            if not success:
                break
        result.tasks.append(TaskRecord(cid, start, now, success, attempts, wasted, reason))
        now = yield _Sleep(rng.expovariate(1.0 / cfg.idle_mean_s))


def simulate(cfg: SimConfig, process: FaultProcess, policy: RetryPolicy) -> SimResult:
    """Run one simulation and return the raw records.

    Args:
        cfg: Simulation parameters.
        process: Fault process shared by all clients (per-backend state).
        policy: Retry policy every client uses.

    Returns:
        The :class:`SimResult` with every task and request timestamp.
    """
    result = SimResult()
    capacity = _Capacity(cfg.capacity_rps)
    heap: list[tuple[float, int, int, object]] = []
    seq = 0
    clients = {}
    attempt_index: dict[int, int] = {}
    for cid in range(cfg.clients):
        rng = random.Random(f"client:{cfg.seed}:{cid}")
        gen = _client(cid, cfg, policy, rng, result)
        clients[cid] = gen
        op = next(gen)
        heapq.heappush(heap, (0.0, seq, cid, op))
        seq += 1
    while heap:
        now, _, cid, op = heapq.heappop(heap)
        if now > cfg.horizon_s:
            break
        gen = clients[cid]
        if isinstance(op, _Sleep):
            wake = now + op.seconds
            heapq.heappush(heap, (wake, seq, cid, ("wake",)))
            seq += 1
            continue
        if isinstance(op, tuple) and op and op[0] == "wake":
            nxt = gen.send(now)
        elif isinstance(op, _Request):
            result.requests[op.backend].append(now)
            fault = process.draw(op.backend, now)
            if fault is None and not capacity.admit(op.backend, now):
                fault = Fault("rate_limit_429", retry_after=1.0, in_burst=False)
                result.load_faults += 1
            done = now + cfg.call_latency_s
            heapq.heappush(heap, (done, seq, cid, ("reply", fault)))
            seq += 1
            continue
        elif isinstance(op, tuple) and op and op[0] == "reply":
            nxt = gen.send((op[1], now))
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unknown op {op!r}")
        heapq.heappush(heap, (now, seq, cid, nxt))
        seq += 1
    return result


def summarize(result: SimResult, cfg: SimConfig, process: FaultProcess,
              herd_window_s: float = 15.0) -> dict:
    """Aggregate a run into the experiment's metrics.

    Args:
        result: Output of :func:`simulate`.
        cfg: The configuration that produced it.
        process: The fault process (used for burst-end times).
        herd_window_s: Seconds after each burst end in which to measure peak load.

    Returns:
        A flat dict of metrics.
    """
    tasks = [t for t in result.tasks if t.end <= cfg.horizon_s]
    n = len(tasks)
    ok = [t for t in tasks if t.success]
    attempts = sum(t.attempts for t in tasks)
    min_calls = n * len(cfg.plan)
    reasons: dict[str, int] = defaultdict(int)
    for t in tasks:
        reasons[t.reason] += 1
    durations = sorted(t.end - t.start for t in ok)
    peak_after_burst = 0.0
    if hasattr(process, "burst_ends"):
        peaks = []
        for backend, times in result.requests.items():
            times = sorted(times)
            for end in process.burst_ends(backend, cfg.horizon_s):
                lo, hi = end, end + herd_window_s
                window = [x for x in times if lo <= x < hi]
                if not window:
                    continue
                per_sec = defaultdict(int)
                for x in window:
                    per_sec[int(x - lo)] += 1
                peaks.append(max(per_sec.values()))
        peak_after_burst = statistics.mean(peaks) if peaks else 0.0
    return {
        "tasks": n,
        "success_rate": len(ok) / n if n else 0.0,
        "attempts_per_task": attempts / n if n else 0.0,
        "amplification": attempts / min_calls if min_calls else 0.0,
        "attempts_per_success": attempts / len(ok) if ok else float("inf"),
        "wasted_per_task": sum(t.wasted for t in tasks) / n if n else 0.0,
        "latency_p50_s": durations[len(durations) // 2] if durations else float("nan"),
        "latency_p95_s": durations[int(0.95 * (len(durations) - 1))] if durations else float("nan"),
        "throughput_per_min": len(ok) / (cfg.horizon_s / 60.0),
        "first_attempt_fault_rate": (result.first_attempt_faults / result.first_attempts
                                     if result.first_attempts else 0.0),
        "load_faults": result.load_faults,
        "peak_rps_after_burst": round(peak_after_burst, 3),
        "fail_attempts": reasons.get("attempts", 0),
        "fail_deadline": reasons.get("deadline", 0),
        "fail_budget": reasons.get("budget", 0),
    }
