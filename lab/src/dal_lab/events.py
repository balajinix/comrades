"""Push vs pull: keeping a consumer in sync with a changing source.

Every interface in the main study is *pull*: the consumer asks when it needs data.
This module models the other direction. A source holds ``n_entities`` records that
are updated and deleted over time, and a consumer keeps a copy. Strategies:

- ``poll_full``   every T s, read the whole snapshot. Level-triggered: the snapshot
  is the state, so a failed poll is repaired by the next one and a deleted record is
  detected by its absence. This is what a typical snapshot poller does.
- ``poll_delta``  every T s, read records changed since the last successful cursor.
  Level-triggered for updates, but a delete is seen only if the source reports
  tombstones.
- ``push_full``   each change is pushed as an event carrying the record (MCP Events
  style webhook: at-least-once, retried with backoff, bounded attempts, possibly out
  of order). Edge-triggered: an event that exhausts its retries is lost, and a delete
  needs an explicit delete event.
- ``push_ref``    each event carries only id and version; the consumer reads the
  current record with a read call ("summary plus read tool").
- ``push_resync`` ``push_full`` plus a full snapshot every R s that repairs whatever
  push lost (Kubernetes-style list-then-watch).

Two consumers:

- a **pipeline** (a poller feeding a message queue) needs every change;
- an **agent** watching a standing condition needs only *relevant* changes (a
  fraction ``relevant_frac``). Push filters on the server; a poll filters with its
  query. An agent's cost is its *wakeups* (model turns). A timer-driven agent wakes on
  every poll; a *gated* poll (plain code polls, and wakes the model only if a
  relevant change arrived) wakes only when there is something to see.

Two fault processes act on two backends: ``source`` (polls and reads) and ``webhook``
(event delivery). Everything runs on one virtual clock, so fault draws arrive in time
order. Changes stop ``cooldown_s`` before the horizon, so slow strategies are not
charged for changes they had no chance to see.
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
from dataclasses import dataclass, field

from .faults import Fault, FaultProcess
from .retry import RetryPolicy

STRATEGIES = ("poll_full", "poll_delta", "push_full", "push_ref", "push_resync")

# Webhook delivery as documented for MCP Events in ChatGPT: exponential backoff, a
# bounded number of attempts. The exact numbers are not published; these are stated
# assumptions. Read calls get the same policy, so push_ref is not handicapped.
WEBHOOK_RETRY = RetryPolicy("webhook", max_attempts=5, backoff="exponential",
                            base_s=2.0, cap_s=300.0)
POLL_RETRY = RetryPolicy("poll", max_attempts=3, backoff="exponential", base_s=2.0)


@dataclass(frozen=True)
class SyncConfig:
    """One sync scenario.

    Attributes:
        n_entities: Records at the source at the start.
        change_rate: Mean changes per second across all records.
        delete_frac: Fraction of changes that delete a record.
        relevant_frac: Fraction of changes the consumer needs (1 for a pipeline).
        tombstones: Whether deltas and events report deletes explicitly.
        burst_factor: Change-rate multiplier during incidents (1 = no incidents).
        incident_every_s: Mean time between incidents.
        incident_len_s: Mean incident length.
        horizon_s: Simulated time.
        cooldown_s: No changes in the last ``cooldown_s`` seconds.
        row_bytes: Size of one full record.
        ref_bytes: Size of an id + version event or a page header.
        page_rows: Records per page of a poll response (one call per page).
        latency_s: One-way delivery or call latency.
        timeout_dup_p: For a timed-out delivery, the chance the consumer had in fact
            processed it, so the retry is a duplicate.
        seed: RNG seed for the change stream.
    """

    n_entities: int = 10_000
    change_rate: float = 0.1
    delete_frac: float = 0.0
    relevant_frac: float = 1.0
    tombstones: bool = True
    burst_factor: float = 1.0
    incident_every_s: float = 6 * 3600
    incident_len_s: float = 1800
    horizon_s: float = 24 * 3600
    cooldown_s: float = 3600
    row_bytes: int = 1024
    ref_bytes: int = 128
    page_rows: int = 1000
    latency_s: float = 0.2
    timeout_dup_p: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class Change:
    """One source change."""

    t: float
    entity: int
    delete: bool
    relevant: bool


def change_stream(cfg: SyncConfig) -> list[Change]:
    """Generate time-ordered changes, optionally with incident bursts.

    Changes are Poisson at ``change_rate``. With ``burst_factor > 1``, incidents
    arrive as a Poisson process and multiply the rate while they last; the base rate
    is scaled down so the total stays the same, isolating burstiness from volume (the
    matched-marginal idea used for the fault processes). A deleted record is
    recreated by its next change, as when a host or pod is replaced.
    """
    rng = random.Random(f"changes:{cfg.seed}")
    end_t = cfg.horizon_s - cfg.cooldown_s
    windows: list[tuple[float, float]] = []
    if cfg.burst_factor > 1:
        t = rng.expovariate(1 / cfg.incident_every_s)
        while t < end_t:
            end = t + rng.expovariate(1 / cfg.incident_len_s)
            windows.append((t, min(end, end_t)))
            t = end + rng.expovariate(1 / cfg.incident_every_s)
    in_incident = sum(e - s for s, e in windows) / end_t
    base = cfg.change_rate / (1 + (cfg.burst_factor - 1) * in_incident)
    out: list[Change] = []
    edges = sorted({0.0, end_t, *(x for w in windows for x in w)})
    for a, b in zip(edges, edges[1:]):
        hot = any(s <= a < e for s, e in windows)
        rate = base * (cfg.burst_factor if hot else 1.0)
        t = a + rng.expovariate(rate) if rate > 0 else b
        while t < b:
            out.append(Change(t, rng.randrange(cfg.n_entities),
                              rng.random() < cfg.delete_frac,
                              rng.random() < cfg.relevant_frac))
            t += rng.expovariate(rate)
    return out


@dataclass
class SyncResult:
    """Outcome of one strategy on one scenario."""

    strategy: str
    delays: list[float] = field(default_factory=list)  # per relevant change; inf = never
    delete_delays: list[float] = field(default_factory=list)
    bytes: int = 0
    source_calls: int = 0
    deliveries: int = 0
    wakeups: int = 0          # timer-driven agent: every poll, or every pushed batch
    gated_wakeups: int = 0    # wakes only when a relevant change arrived
    lost_events: int = 0
    duplicates: int = 0
    reorders: int = 0  # times a naive (arrival-order) consumer overwrote newer data

    def summary(self) -> dict:
        """Aggregate metrics for result tables."""
        def stats(xs: list[float], prefix: str) -> dict:
            fin = sorted(d for d in xs if math.isfinite(d))

            def q(p: float) -> float:
                return fin[min(len(fin) - 1, int(p * len(fin)))] if fin else math.nan

            return {f"{prefix}missed_frac": round((len(xs) - len(fin)) / len(xs), 5)
                    if xs else 0.0,
                    f"{prefix}p50_s": round(q(0.5), 1),
                    f"{prefix}p95_s": round(q(0.95), 1),
                    f"{prefix}mean_s": round(sum(fin) / len(fin), 1) if fin else math.nan}

        out = {"strategy": self.strategy, "changes": len(self.delays),
               "deletes": len(self.delete_delays)}
        out.update(stats(self.delays, "delay_"))
        out.update(stats(self.delete_delays, "del_"))
        out.update({"mbytes": round(self.bytes / 1e6, 3),
                    "source_calls": self.source_calls, "deliveries": self.deliveries,
                    "wakeups": self.wakeups, "gated_wakeups": self.gated_wakeups,
                    "lost_events": self.lost_events, "duplicates": self.duplicates,
                    "reorders": self.reorders})
        return out


class _Sim:
    """Discrete-event simulation of one strategy."""

    def __init__(self, cfg: SyncConfig, strategy: str, faults: FaultProcess,
                 interval_s: float, resync_s: float, webhook: RetryPolicy,
                 coalesce_s: float) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        self.cfg, self.strategy, self.faults = cfg, strategy, faults
        self.interval_s, self.resync_s = interval_s, resync_s
        self.webhook, self.coalesce_s = webhook, coalesce_s
        self.rng = random.Random(f"sync:{cfg.seed}:{strategy}")
        self.heap: list = []
        self.seq = itertools.count()
        self.res = SyncResult(strategy)
        n = cfg.n_entities
        self.src_ver = [0] * n         # source version per entity
        self.src_alive = [True] * n
        self.dst_ver = [0] * n         # consumer's highest version held
        self.naive_ver = [0] * n       # arrival-order consumer, for reorders
        # entity -> [(version, time, is_delete)] of relevant changes not yet held
        self.pending: dict[int, list[tuple[int, float, bool]]] = {}
        self.seen: set[tuple[int, int]] = set()  # (entity, version) processed
        self.changes: list[Change] = []
        self.applied = 0  # changes applied to the source so far
        self.cursor = -math.inf  # nothing read yet; a change at t = 0 is still new
        self.last_wakeup = {"timer": -math.inf, "gated": -math.inf}

    # -- scheduling -----------------------------------------------------------------
    def at(self, t: float, kind: str, *args) -> None:
        if t <= self.cfg.horizon_s:
            heapq.heappush(self.heap, (t, next(self.seq), kind, args))

    def run(self, changes: list[Change]) -> SyncResult:
        self.changes = changes
        for i, c in enumerate(changes):
            self.at(c.t, "change", i)
        if self.strategy == "poll_full":
            self.at(self.interval_s, "poll", False, 1)
        elif self.strategy == "poll_delta":
            self.at(self.interval_s, "poll", True, 1)
        elif self.strategy == "push_resync":
            self.at(self.resync_s, "poll", False, 1)
        while self.heap:
            t, _, kind, args = heapq.heappop(self.heap)
            getattr(self, f"_{kind}")(t, *args)
        for items in self.pending.values():
            for _, _, is_del in items:
                (self.res.delete_delays if is_del else self.res.delays).append(math.inf)
        return self.res

    # -- source side ----------------------------------------------------------------
    def _change(self, t: float, i: int) -> None:
        c = self.changes[i]
        e = c.entity
        self.applied = i + 1
        self.src_ver[e] += 1
        v = self.src_ver[e]
        self.src_alive[e] = not c.delete
        if c.relevant:
            self.pending.setdefault(e, []).append((v, t, c.delete))
        if not c.relevant or self.strategy.startswith("poll"):
            return  # push filters on the server; polls see it on their schedule
        if c.delete and not self.cfg.tombstones:
            return  # nothing to push: the source does not report deletes
        ref = self.strategy == "push_ref"
        self.at(t + self.cfg.latency_s, "deliver", e, v, 1, ref)

    # -- consumer side --------------------------------------------------------------
    def _hold(self, t: float, e: int, v: int) -> None:
        """The consumer now holds version ``v`` of ``e`` (keeps the max)."""
        if v <= self.dst_ver[e]:
            return
        self.dst_ver[e] = v
        keep = []
        for pv, pt, is_del in self.pending.get(e, []):
            if pv <= v:
                (self.res.delete_delays if is_del else self.res.delays).append(t - pt)
            else:
                keep.append((pv, pt, is_del))
        if keep:
            self.pending[e] = keep
        else:
            self.pending.pop(e, None)

    def _wake(self, t: float, relevant: bool) -> None:
        # Arrivals within ``coalesce_s`` of the last wakeup share it (a batched run).
        for kind, hit in (("timer", True), ("gated", relevant)):
            if hit and t - self.last_wakeup[kind] >= self.coalesce_s:
                self.last_wakeup[kind] = t
                if kind == "timer":
                    self.res.wakeups += 1
                else:
                    self.res.gated_wakeups += 1

    def _deliver(self, t: float, e: int, v: int, attempt: int, ref: bool) -> None:
        self.res.deliveries += 1
        self.res.bytes += self.cfg.ref_bytes if ref else self.cfg.row_bytes
        fault: Fault | None = self.faults.draw("webhook", t)
        processed = fault is None or (fault.kind == "timeout"
                                      and self.rng.random() < self.cfg.timeout_dup_p)
        if processed:
            if (e, v) in self.seen:
                self.res.duplicates += 1
            self.seen.add((e, v))
            self._wake(t, True)
            if ref:
                self.at(t, "fetch", e, 1)
            else:
                if v < self.naive_ver[e]:
                    self.res.reorders += 1
                self.naive_ver[e] = v
                self._hold(t, e, v)
        if fault is not None:
            if attempt < self.webhook.max_attempts:
                wait = self.webhook.wait(fault, attempt, self.rng)
                self.at(t + wait, "deliver", e, v, attempt + 1, ref)
            elif not processed:
                self.res.lost_events += 1

    def _fetch(self, t: float, e: int, attempt: int) -> None:
        """Read tool: fetch the current record (always the latest version)."""
        self.res.source_calls += 1
        fault = self.faults.draw("source", t)
        if fault is None:
            self.res.bytes += self.cfg.row_bytes
            self._hold(t + self.cfg.latency_s, e, self.src_ver[e])
        elif self.webhook.should_retry(fault, attempt):
            wait = self.webhook.wait(fault, attempt, self.rng)
            self.at(t + wait, "fetch", e, attempt + 1)
        else:
            self.res.lost_events += 1

    def _poll(self, t: float, delta: bool, attempt: int) -> None:
        cfg = self.cfg
        period = self.resync_s if self.strategy == "push_resync" else self.interval_s
        if attempt == 1:
            self.at(t + period, "poll", delta, 1)  # the schedule does not slip
        if delta:
            since = self._first_change_after(self.cursor)
            touched = {self.changes[i].entity for i in range(since, self.applied)}
            # The query filters to rows the consumer needs; without tombstones,
            # deleted rows are simply absent from a delta.
            rows = [e for e in touched if (e in self.pending or cfg.relevant_frac == 1)
                    and (self.src_alive[e] or cfg.tombstones)]
        else:
            rows = None
        n_rows = len(rows) if delta else sum(self.src_alive)
        pages = max(1, math.ceil(n_rows / cfg.page_rows))
        for _ in range(pages):
            self.res.source_calls += 1
            fault = self.faults.draw("source", t)
            if fault is not None:
                if POLL_RETRY.should_retry(fault, attempt):
                    wait = POLL_RETRY.wait(fault, attempt, self.rng)
                    self.at(t + wait, "poll", delta, attempt + 1)
                return  # the next scheduled poll repairs a failed one
        self.res.bytes += n_rows * cfg.row_bytes + pages * cfg.ref_bytes
        held = t + cfg.latency_s
        if delta:
            targets = rows
            self.cursor = t
        else:
            # A full snapshot shows every live record, and every deleted one by its
            # absence, so the consumer can catch up on all of them.
            targets = list(self.pending)
        relevant = False
        for e in targets:
            if self.src_ver[e] > self.dst_ver[e]:
                relevant = relevant or e in self.pending
                self._hold(held, e, self.src_ver[e])
        if self.strategy != "push_resync":
            self._wake(t, relevant)

    def _first_change_after(self, since: float) -> int:
        lo, hi = 0, self.applied
        while lo < hi:
            mid = (lo + hi) // 2
            if self.changes[mid].t <= since:
                lo = mid + 1
            else:
                hi = mid
        return lo


def simulate_sync(cfg: SyncConfig, strategy: str, faults: FaultProcess,
                  interval_s: float = 300.0, resync_s: float = 3600.0,
                  webhook: RetryPolicy = WEBHOOK_RETRY,
                  coalesce_s: float = 0.0,
                  changes: list[Change] | None = None) -> SyncResult:
    """Run one strategy over one change stream.

    Args:
        cfg: Scenario.
        strategy: One of :data:`STRATEGIES`.
        faults: Fault process over backends ``source`` and ``webhook``.
        interval_s: Poll interval for the poll strategies (e.g. 300 s).
        resync_s: Full-snapshot interval for ``push_resync``.
        webhook: Delivery (and read-call) retry policy for the push strategies.
        coalesce_s: Batch window for counting consumer wakeups.
        changes: A recorded change stream (time-ordered, times relative to the
            start, entities in ``range(cfg.n_entities)``) to replay instead of the
            generated one.

    Returns:
        The strategy's :class:`SyncResult`.
    """
    sim = _Sim(cfg, strategy, faults, interval_s, resync_s, webhook, coalesce_s)
    return sim.run(changes if changes is not None else change_stream(cfg))
