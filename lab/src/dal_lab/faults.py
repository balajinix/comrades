"""Fault processes for the data-access boundary: independent and bursty.

The ICSE experiments drew every fault independently (``rng.random() < rate``).
Real outages are correlated: rate-limit storms, deploys and failovers arrive in
bursts. :class:`GilbertElliott` models that with a per-backend two-state Markov
chain whose dwell times are exponential in *time*, not in call count. A chain
indexed by calls would make backoff useless by construction, because waiting
would never end a burst.

Both processes answer one question, ``draw(backend, now)``: does a call that
arrives at virtual time ``now`` fail, and with which fault type? Calls must be
drawn in non-decreasing ``now`` per backend (the simulator guarantees this),
because each chain is advanced lazily.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# Fault catalog, aligned with evaluation/comrades/policy.py. ``retry_after`` marks
# the faults a server would answer with a Retry-After hint.
FAULT_TYPES: dict[str, dict] = {
    "timeout": {"retryable": True, "retry_after": False},
    "rate_limit_429": {"retryable": True, "retry_after": True},
    "server_5xx": {"retryable": True, "retry_after": True},
    "partial_outage": {"retryable": True, "retry_after": False},
    "auth_expiry": {"retryable": False, "retry_after": False},
}
BURST_TYPES = ("timeout", "rate_limit_429", "server_5xx", "partial_outage")


@dataclass(frozen=True)
class Fault:
    """An injected fault as the client sees it.

    Attributes:
        kind: Fault type from :data:`FAULT_TYPES`.
        retry_after: Server hint in seconds, or None when the fault carries none.
        in_burst: Whether the backend was in the bad state (for wasted-call accounting).
    """

    kind: str
    retry_after: float | None = None
    in_burst: bool = False

    @property
    def retryable(self) -> bool:
        """Whether a retry policy may re-attempt this fault."""
        return FAULT_TYPES[self.kind]["retryable"]


class FaultProcess:
    """Interface: decide whether a call to ``backend`` at time ``now`` fails."""

    def draw(self, backend: str, now: float) -> Fault | None:
        """Return the fault for this call, or None if it succeeds."""
        raise NotImplementedError

    def in_bad_state(self, backend: str, now: float) -> bool:
        """Whether ``backend`` is inside a burst at ``now`` (False if unmodelled)."""
        return False

    def describe(self) -> dict:
        """Parameters for result logging."""
        raise NotImplementedError


class Bernoulli(FaultProcess):
    """Independent faults: each call fails with probability ``rate`` (the ICSE model)."""

    def __init__(self, rate: float, seed: int = 0, types: tuple[str, ...] = BURST_TYPES):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        self.rate = rate
        self.types = types
        self._rng = random.Random(f"bernoulli:{seed}")

    def draw(self, backend: str, now: float) -> Fault | None:
        if self._rng.random() >= self.rate:
            return None
        kind = self._rng.choice(self.types)
        return Fault(kind, retry_after=None)

    def describe(self) -> dict:
        return {"process": "bernoulli", "rate": self.rate}


@dataclass
class _Chain:
    """Lazily advanced state of one backend's Gilbert–Elliott chain."""

    bad: bool
    until: float  # virtual time at which the current state ends
    burst_kind: str = "server_5xx"
    burst_id: int = 0
    burst_ends: list[float] = field(default_factory=list)


class GilbertElliott(FaultProcess):
    """Bursty faults: a per-backend good/bad Markov chain in continuous time.

    Dwell times are exponential with means ``mean_good`` and ``mean_bad`` seconds.
    A call fails with probability ``e_good`` in the good state and ``e_bad`` in the
    bad state. All faults inside one burst share a type, drawn when the burst
    starts (a burst is *a* 429 storm or *a* 503 outage). Retry-After hints, when
    the type carries one, report the burst's mean length, as a server would.
    """

    def __init__(self, mean_good: float, mean_bad: float, e_good: float = 0.0,
                 e_bad: float = 1.0, seed: int = 0, types: tuple[str, ...] = BURST_TYPES):
        if mean_good <= 0 or mean_bad <= 0:
            raise ValueError("mean dwell times must be positive")
        self.mean_good, self.mean_bad = mean_good, mean_bad
        self.e_good, self.e_bad = e_good, e_bad
        self.types = types
        self.seed = seed
        self._chains: dict[str, _Chain] = {}
        self._call_rng = random.Random(f"ge-calls:{seed}")

    @classmethod
    def matched(cls, rate: float, mean_bad: float, e_bad: float = 1.0, seed: int = 0,
                types: tuple[str, ...] = BURST_TYPES) -> "GilbertElliott":
        """Build a chain whose long-run fault rate equals ``rate`` (with e_good = 0).

        Holding the marginal rate fixed and sweeping only ``mean_bad`` isolates the
        effect of correlation from the effect of having more faults.

        Args:
            rate: Target marginal fault rate, 0 < rate <= e_bad.
            mean_bad: Mean burst length in seconds.
            e_bad: Error probability inside a burst.
            seed: RNG seed.
            types: Fault types a burst may take.

        Returns:
            A calibrated :class:`GilbertElliott`.
        """
        if not 0.0 < rate <= e_bad:
            raise ValueError(f"need 0 < rate <= e_bad, got rate={rate}, e_bad={e_bad}")
        pi_bad = rate / e_bad
        if pi_bad >= 1.0:
            mean_good = 1e-9  # permanently bad
        else:
            mean_good = mean_bad * (1.0 - pi_bad) / pi_bad
        return cls(mean_good, mean_bad, 0.0, e_bad, seed, types)

    @property
    def stationary_bad(self) -> float:
        """Long-run fraction of time spent in the bad state."""
        return self.mean_bad / (self.mean_good + self.mean_bad)

    @property
    def marginal_rate(self) -> float:
        """Long-run fault probability for a call arriving at a random time."""
        pb = self.stationary_bad
        return (1 - pb) * self.e_good + pb * self.e_bad

    def _chain(self, backend: str) -> _Chain:
        chain = self._chains.get(backend)
        if chain is None:
            rng = self._chain_rng(backend)
            bad = rng.random() < self.stationary_bad  # start in the stationary distribution
            mean = self.mean_bad if bad else self.mean_good
            chain = _Chain(bad=bad, until=rng.expovariate(1.0 / mean),
                           burst_kind=rng.choice(self.types), burst_id=int(bad))
            self._chains[backend] = chain
        return chain

    def _chain_rng(self, backend: str) -> random.Random:
        # One RNG per backend, so a backend's burst schedule does not depend on how
        # many calls other backends received (mechanism-independent bursts).
        key = f"_rng_{backend}"
        rng = getattr(self, key, None)
        if rng is None:
            rng = random.Random(f"ge:{self.seed}:{backend}")
            setattr(self, key, rng)
        return rng

    def _advance(self, backend: str, now: float) -> _Chain:
        chain = self._chain(backend)
        rng = self._chain_rng(backend)
        while chain.until <= now:
            if chain.bad:
                chain.burst_ends.append(chain.until)
            chain.bad = not chain.bad
            mean = self.mean_bad if chain.bad else self.mean_good
            start = chain.until
            chain.until = start + rng.expovariate(1.0 / mean)
            if chain.bad:
                chain.burst_id += 1
                chain.burst_kind = rng.choice(self.types)
        return chain

    def in_bad_state(self, backend: str, now: float) -> bool:
        return self._advance(backend, now).bad

    def burst_ends(self, backend: str, horizon: float) -> list[float]:
        """Times at which bursts on ``backend`` ended, up to ``horizon``."""
        chain = self._advance(backend, horizon)
        return [t for t in chain.burst_ends if t <= horizon]

    def draw(self, backend: str, now: float) -> Fault | None:
        chain = self._advance(backend, now)
        p = self.e_bad if chain.bad else self.e_good
        if self._call_rng.random() >= p:
            return None
        if chain.bad:
            kind = chain.burst_kind
            hint = self.mean_bad if FAULT_TYPES[kind]["retry_after"] else None
            return Fault(kind, retry_after=hint, in_burst=True)
        return Fault(self._call_rng.choice(self.types), retry_after=None)

    def describe(self) -> dict:
        return {"process": "gilbert_elliott", "mean_good": round(self.mean_good, 4),
                "mean_bad": self.mean_bad, "e_good": self.e_good, "e_bad": self.e_bad,
                "marginal_rate": round(self.marginal_rate, 4)}


class Outage(FaultProcess):
    """A deterministic outage window on every backend: [start, end) fails with ``kind``.

    Used for the synchronized-onset (thundering-herd) scenario, where all clients
    are hit by the same failure at the same moment, as when a backend restarts.
    """

    def __init__(self, start: float, end: float, kind: str = "server_5xx"):
        self.start, self.end, self.kind = start, end, kind

    def in_bad_state(self, backend: str, now: float) -> bool:
        return self.start <= now < self.end

    def burst_ends(self, backend: str, horizon: float) -> list[float]:
        """The single outage end, if within ``horizon``."""
        return [self.end] if self.end <= horizon else []

    def draw(self, backend: str, now: float) -> Fault | None:
        if not self.in_bad_state(backend, now):
            return None
        hint = (self.end - self.start) if FAULT_TYPES[self.kind]["retry_after"] else None
        return Fault(self.kind, retry_after=hint, in_burst=True)

    def describe(self) -> dict:
        return {"process": "outage", "start": self.start, "end": self.end, "kind": self.kind}


def expected_burst_count(process: GilbertElliott, horizon: float) -> float:
    """Expected number of bursts per backend in ``horizon`` seconds (for sanity checks)."""
    cycle = process.mean_good + process.mean_bad
    return horizon / cycle if cycle > 0 and math.isfinite(cycle) else 0.0
