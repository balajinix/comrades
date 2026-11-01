"""Pydantic models for the agent.

`KgOutput` mirrors the existing `mcp_kg.KgOutput` exactly — it's the wire
contract the UI's `static/js/app.js::renderResult` consumes (see
`ui/UI_INTEGRATION.md` §3 / §4). Don't drift from this shape without
coordinating with the UI team.

`TraceStep` is internal: it captures one ReAct iteration so the synthesizer
can build `renderings` from each leaf-tool response and so the persistence
layer can write a per-step audit row.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class Rendering(BaseModel):
    """One typed visual element. Six concrete types are wired in the UI:
    scalar, timeseries, table, graph, code, markdown."""
    type: Literal["scalar", "timeseries", "table", "graph", "code", "markdown"]
    label: str | None = None
    title: str | None = None

    # scalar
    value: float | int | str | None = None
    unit: str | None = None

    # timeseries: series=[{name, unit?, points: [[t, v], ...]}, ...]
    series: list[dict[str, Any]] | None = None

    # table
    columns: list[dict[str, Any]] | None = None  # [{key, header, type?}]
    rows: list[dict[str, Any]] | None = None

    # graph (neutral schema, frontend translates to vis.js)
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None

    # code / markdown
    language: str | None = None
    body: str | None = None


class Graph(BaseModel):
    """Legacy top-level graph field. Prefer renderings[].type === 'graph'."""
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class KgOutput(BaseModel):
    """The wire shape returned to the UI. See ui/UI_INTEGRATION.md §3."""
    status: Literal["completed", "error"]
    message: str
    tool_invoked: str | None = None
    routing_reason: str | None = None
    answer_markdown: str | None = None
    renderings: list[Rendering] | None = None
    raw_result: dict[str, Any] | None = None
    graph: Graph | None = None
    source: str | None = None
    elapsed_seconds: float = 0.0
    # Resolved LLM id used for this run (e.g. "provider/meta-llama/llama-3-3-70b-instruct").
    # Optional — older clients ignore it; UI surfaces it as a chip.
    model: str | None = None
    # True iff this response was served from the read-through cache rather
    # than from a live ReAct run. Additive to the wire contract — older
    # clients ignore it; UI can show a "(cached)" chip. See CACHE_DESIGN.md.
    cache_hit: bool = False


class TraceStep(BaseModel):
    """One iteration of the ReAct loop.

    `tool_result` keeps the FULL untruncated mcp_kg response so the
    synthesizer can lift its `renderings` straight through to the UI.
    `observation` is the truncated form fed back into the LLM context.
    """
    step: int
    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    tool_result: dict[str, Any] | None = None
    final_answer: str | None = None
    # Wall-clock time the iteration took (LLM + tool + bookkeeping).
    # None when the step pre-dates the timing instrumentation or the
    # caller didn't time it. Surfaced in the dashboard's drill-down.
    elapsed_ms: int | None = None
