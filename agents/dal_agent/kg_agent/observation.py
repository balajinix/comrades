"""Compact a tool's structured result into a string suitable for LLM context.

Each step of the loop appends the tool's response back into the conversation
so the LLM can decide what to do next. Full mcp_kg responses can be tens of
KB (raw GraphQL row sets); without truncation the context window grows
linearly with steps. Cap each observation at ~2 KB.

The full untruncated dict stays on `TraceStep.tool_result` so the synthesizer
can still extract renderings from it later.
"""
from __future__ import annotations

import json
from typing import Any


_OBS_LIMIT = 2000


def format_observation(raw: dict[str, Any]) -> str:
    """Best-effort summary of a tool response. Falls through to JSON dump for
    unrecognised shapes."""
    if raw.get("status") == "error" or raw.get("error"):
        msg = raw.get("message") or raw.get("error") or "unknown"
        return _truncate(f"ERROR: {msg}")

    # FinOps / APM / ITSM generators carry endpoint + rows.
    if raw.get("endpoint"):
        bits = [f"endpoint={raw.get('endpoint')}"]
        results = raw.get("results")
        if isinstance(results, dict):
            # Surface server-computed totals before the rows. FinOps
            # cost reports return aggregates at `results.meta.aggregates`
            # (NOT at the top level — that path looked obvious and was
            # wrong). The row sum is pre-computed; without surfacing it
            # the LLM adds up sample rows and gives up (observed: "the
            # exact total would need summing").
            meta = results.get("meta")
            if isinstance(meta, dict):
                agg_str = _format_aggregates(meta.get("aggregates"))
                if agg_str:
                    bits.append(agg_str)
            # Surface the first list-bearing key. The candidate list is broad
            # because different products nest their rows under different names
            # — FinOps uses `results`, Resilience uses `applications`,
            # others use `items`/`records`/etc. Without this, dict payloads
            # like Resilience's `{pagination, applications:[...]}` previously
            # rendered as a bare `keys=[...]` and the LLM couldn't see the rows
            # it had actually fetched.
            for key in ("results", "result", "applications", "items", "records",
                        "incidents", "users", "articles", "workflows", "risks",
                        "assessments", "data"):
                rows = results.get(key)
                if isinstance(rows, list):
                    bits.append(f"{key}: {len(rows)} rows; sample: {rows[:3]}")
                    break
            else:
                bits.append(f"keys={list(results)[:5]}")
        elif isinstance(results, list):
            bits.append(f"{len(results)} rows; sample: {results[:3]}")
        return _truncate(" | ".join(str(b) for b in bits))

    # GraphQL generators (CEG, Registry, ITOM).
    if raw.get("graphql_query"):
        report = raw.get("report") or ""
        results = raw.get("results") or {}
        data = results.get("data") if isinstance(results, dict) else None
        sample = json.dumps(data, default=str)[:600] if data else ""
        return _truncate(
            f"report={report}\nquery={str(raw['graphql_query'])[:300]}\ndata={sample}"
        )

    # ITSM query templates.
    if raw.get("table"):
        rows = raw.get("results") or []
        return _truncate(f"table={raw['table']}; {len(rows)} rows; sample: {rows[:3]}")

    # Metric calculators. Keep `per_application` — it's the ranked breakdown
    # the agent needs to answer "which application has the worst X" without
    # going on a hunt across other tools. The few-shot prompt explicitly
    # references this field; stripping it makes the example a lie.
    # Drop only the genuinely bulky / low-signal fields.
    if "value" in raw:
        slim = {k: v for k, v in raw.items() if k not in (
            "query_used", "calculation_logic",
        )}
        return _truncate(json.dumps(slim, default=str))

    return _truncate(json.dumps(raw, default=str))


def _format_aggregates(aggregates: Any) -> str | None:
    """Format FinOps `aggregates[]` as 'totals: name=$X.XX, ...'.
    Returns None if the field is missing or empty."""
    if not isinstance(aggregates, list) or not aggregates:
        return None
    parts: list[str] = []
    for agg in aggregates:
        if not isinstance(agg, dict):
            continue
        name = agg.get("name") or agg.get("label") or "metric"
        val = agg.get("value")
        if val is None:
            continue
        try:
            num = float(val)
            parts.append(f"{name}={num:,.2f}")
        except (TypeError, ValueError):
            parts.append(f"{name}={val}")
    if not parts:
        return None
    return "totals: " + ", ".join(parts)


def _truncate(text: str, limit: int = _OBS_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…(truncated, {len(text) - limit} chars omitted)"
