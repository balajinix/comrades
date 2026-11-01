"""Retry policies: when to re-attempt a failed data-access call, and how long to wait.

Extends the ICSE ``RetryPolicy`` (fixed attempts, no wait) with the waiting
strategies that matter under bursty faults: constant and exponential backoff,
jitter, Retry-After, and a client-side retry budget. Every arm gets the same
``max_attempts`` so the arms differ only in *timing*, not in persistence.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .faults import Fault


@dataclass(frozen=True)
class RetryPolicy:
    """A retry policy.

    Attributes:
        name: Arm label for result tables.
        max_attempts: First try plus retries.
        backoff: "none", "constant" or "exponential".
        base_s: Constant wait, or the first exponential wait.
        cap_s: Upper bound on one exponential wait.
        jitter: "none" or "full" (wait ~ U(0, computed wait)).
        honor_retry_after: Wait at least the server's Retry-After hint.
        budget: Max retries per client as a fraction of first attempts (None = unlimited).
    """

    name: str
    max_attempts: int = 5
    backoff: str = "none"
    base_s: float = 1.0
    cap_s: float = 30.0
    jitter: str = "none"
    honor_retry_after: bool = False
    budget: float | None = None

    def should_retry(self, fault: Fault, attempt: int) -> bool:
        """Whether to re-attempt after ``attempt`` failed attempts (1-based)."""
        return fault.retryable and attempt < self.max_attempts

    def wait(self, fault: Fault, attempt: int, rng: random.Random) -> float:
        """Seconds to wait before the next attempt, after ``attempt`` failures."""
        if self.backoff == "none":
            delay = 0.0
        elif self.backoff == "constant":
            delay = self.base_s
        elif self.backoff == "exponential":
            delay = min(self.cap_s, self.base_s * 2 ** (attempt - 1))
        else:
            raise ValueError(f"unknown backoff {self.backoff!r}")
        if self.jitter == "full":
            delay = rng.uniform(0.0, delay)
        elif self.jitter != "none":
            raise ValueError(f"unknown jitter {self.jitter!r}")
        if self.honor_retry_after and fault.retry_after is not None:
            delay = max(delay, fault.retry_after)
        return delay


@dataclass
class RetryBudget:
    """Client-side retry budget: retries may not exceed ``ratio`` × first attempts.

    A small floor lets a fresh client retry at all. This caps retry amplification
    during long outages, which is the standard defence against retry storms.
    """

    ratio: float
    floor: int = 3
    first_attempts: int = 0
    retries: int = 0

    def allow(self) -> bool:
        """Whether one more retry fits in the budget."""
        return self.retries < self.floor + self.ratio * self.first_attempts


# The arms of the bursty-fault experiment. Same max_attempts everywhere.
ARMS: dict[str, RetryPolicy] = {
    "fail_fast": RetryPolicy("fail_fast", max_attempts=1),
    "immediate": RetryPolicy("immediate"),
    "constant": RetryPolicy("constant", backoff="constant", base_s=2.0),
    "exponential": RetryPolicy("exponential", backoff="exponential"),
    "exp_jitter": RetryPolicy("exp_jitter", backoff="exponential", jitter="full"),
    # Full jitter halves the expected wait; this arm doubles the base so its expected
    # wait matches "exponential", separating "randomized" from "waits less".
    "exp_jitter_eqwait": RetryPolicy("exp_jitter_eqwait", backoff="exponential",
                                     base_s=2.0, cap_s=60.0, jitter="full"),
    "exp_jitter_retry_after": RetryPolicy("exp_jitter_retry_after", backoff="exponential",
                                          jitter="full", honor_retry_after=True),
    "exp_jitter_budget": RetryPolicy("exp_jitter_budget", backoff="exponential",
                                     jitter="full", budget=0.2),
}
