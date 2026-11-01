"""Analytic communication-cost model for agentic data access.

A ReAct-style agent re-sends its whole context on every turn. With base context
``b`` (system prompt + tool descriptions + question) and per-call additions
``s_1..s_T`` (request + observation tokens), the input tokens billed over a run are

    C_in = sum_{t=0..T} (b + s_1 + ... + s_t)

which for uniform ``s`` is ``(T+1)·b + s·T(T+1)/2``: linear in the base, quadratic
in the number of calls. Prompt caching bills the re-sent prefix at a discount.

Interface effects follow:

- Data shipping (fine-grained tools) puts ``rows_needed(sf)·τ`` tokens in context,
  where τ is tokens per row; with ``rows_needed`` linear in scale factor ``sf`` it
  overflows a budget ``B`` at ``sf* = (B − b) / (ρ·τ·(1 + o/(P·τ)))`` (ρ = rows per
  unit sf, o = per-call overhead, P = page size).
- Query shipping (one generic query) costs ``b + q + out(sf)·τ_out``, which is
  independent of scale when the answer size is (aggregates, top-k, lookups).
- Result handles cost a constant ``h`` (one extra call + preview), so they beat
  in-context results once ``out(sf)·τ_out > h``: a crossover at ``sf_x = h/(ρ_out·τ_out)``.

Models are calibrated at the smallest scale factor and used to *predict* larger
ones; ``exp_scale_crossover`` compares predictions with measurements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# TPC-H tables whose size does not depend on the scale factor.
CONSTANT_TABLES = frozenset({"nation", "region"})


def cumulative_input(base: int, steps: list[int], cache_hit: float = 0.0,
                     cache_discount: float = 0.1) -> float:
    """Input tokens billed over a ReAct run.

    Args:
        base: Turn-0 context (system + tools + question).
        steps: Tokens added by each tool call (request + observation).
        cache_hit: Fraction of the re-sent prefix served from a prompt cache.
        cache_discount: Price of a cached token relative to an uncached one.

    Returns:
        Billed input tokens (equals the plain sum of contexts when ``cache_hit=0``).
    """
    total, ctx = float(base), base
    prefix_price = cache_hit * cache_discount + (1.0 - cache_hit)
    for s in steps:
        total += ctx * prefix_price + s  # prefix re-sent, new tokens fresh
        ctx += s
    return total


def cumulative_uniform(base: int, s: int, calls: int) -> int:
    """Closed form of :func:`cumulative_input` for uniform steps, no caching.

    Args:
        base: Turn-0 context.
        s: Tokens per call.
        calls: Number of calls T.

    Returns:
        ``(T+1)·base + s·T(T+1)/2``.
    """
    return (calls + 1) * base + s * calls * (calls + 1) // 2


@dataclass
class ShippingModel:
    """Data-shipping cost of a fine-grained plan, calibrated at one scale factor.

    Attributes:
        base: Turn-0 context tokens of the fine-grained facade.
        const_tokens: Tokens of rows from scale-independent tables.
        linear_tokens_per_sf: Tokens of scale-dependent rows per unit sf.
        linear_rows_per_sf: Scale-dependent rows per unit sf.
        page: Rows per list call.
        call_overhead: Request + envelope tokens per call.
    """

    base: int
    const_tokens: float
    linear_tokens_per_sf: float
    linear_rows_per_sf: float
    page: int = 1000
    call_overhead: int = 60

    @classmethod
    def calibrate(cls, base: int, step_rows: dict[str, int],
                  tokens_per_row: dict[str, float], sf: float, **kw: int) -> "ShippingModel":
        """Fit the model from row counts measured at scale factor ``sf``.

        Args:
            base: Turn-0 context tokens.
            step_rows: ``{"i:table": rows}`` from :func:`policies.plan_rows`.
            tokens_per_row: Tokens per row per table.
            sf: The calibration scale factor.
            **kw: ``page`` / ``call_overhead`` overrides.

        Returns:
            The calibrated model.
        """
        const_t = lin_t = lin_r = 0.0
        for key, rows in step_rows.items():
            table = key.split(":", 1)[1]
            tok = rows * tokens_per_row[table]
            if table in CONSTANT_TABLES:
                const_t += tok
            else:
                lin_t += tok / sf
                lin_r += rows / sf
        return cls(base, const_t, lin_t, lin_r, **kw)

    def calls(self, sf: float) -> int:
        """Predicted list calls (at least one per plan step is ignored: pages only)."""
        return max(1, math.ceil(self.linear_rows_per_sf * sf / self.page))

    def context_tokens(self, sf: float) -> float:
        """Predicted peak context at ``sf``."""
        return (self.base + self.const_tokens + self.linear_tokens_per_sf * sf
                + self.calls(sf) * self.call_overhead)

    def overflow_sf(self, budget: int) -> float:
        """Smallest scale factor at which the plan overflows ``budget`` tokens."""
        room = budget - self.base - self.const_tokens
        if room <= 0:
            return 0.0
        per_sf = self.linear_tokens_per_sf + self.linear_rows_per_sf / self.page * self.call_overhead
        return math.inf if per_sf <= 0 else room / per_sf


def handle_crossover_sf(handle_overhead: float, out_tokens_per_sf: float) -> float:
    """Scale factor beyond which returning results by handle beats in-context rows.

    Args:
        handle_overhead: Constant extra tokens of the handle path (descriptor delta,
            preview, extra turn).
        out_tokens_per_sf: Answer tokens per unit sf when rows are shipped in context.

    Returns:
        The crossover scale factor (``inf`` if the answer does not grow).
    """
    return math.inf if out_tokens_per_sf <= 0 else handle_overhead / out_tokens_per_sf
