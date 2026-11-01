"""COMRADES benchmark harness — core types.

A ``WorkflowRunner`` executes one natural-language data-access request under one
*communication mechanism* (Direct / MCP-only / A2A / ACP) and returns a
``RunResult`` carrying the produced artifact plus the metrics the benchmark
scores: success, latency, token cost, messages-per-task, and failure recovery.

The agents, tools, dataset, and model are held fixed across mechanisms; only the
runner changes. That is what makes the mechanism a controlled variable
(see COMRADES_PLAN.md §3 and the paper's Section "Communication Mechanisms").
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RunResult:
    """Outcome + metrics of running one request under one mechanism."""

    request_id: str
    mechanism: str          # "direct" | "mcp_only" | "a2a" | "acp"
    model: str | None = None

    # --- task outcome ---
    success: bool = False           # produced a usable artifact (not an error)
    artifact: Any = None            # answer text / renderings / query / mapping
    error: str | None = None

    # --- communication metrics ---
    latency_s: float = 0.0          # end-to-end wall clock
    tokens_total: int | None = None  # LLM tokens consumed (None if not captured)
    tokens_prompt: int | None = None
    tokens_completion: int | None = None
    messages: int = 0               # inter-agent / ReAct steps == coordination volume
    recovered: bool = False         # recovered from >=1 transient failure mid-run

    # --- provenance ---
    tool_invoked: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)  # mechanism-specific detail

    def to_row(self) -> dict[str, Any]:
        """Flatten for CSV/DataFrame output."""
        d = asdict(self)
        d.pop("raw", None)
        # artifact can be large/nested; keep a short preview only in the row.
        art = d.get("artifact")
        d["artifact"] = (str(art)[:200] if art is not None else None)
        return d


class WorkflowRunner(abc.ABC):
    """Executes a request under one communication mechanism.

    Async-first: every mechanism (in-process or networked) ultimately drives an
    async ReAct loop, so the canonical entry point is ``arun``. Subclasses set
    ``mechanism`` and implement ``arun``.
    """

    mechanism: str = "base"

    def __init__(self, model: str | None = None):
        self.model = model

    @abc.abstractmethod
    async def arun(self, request_id: str, prompt: str) -> RunResult:
        """Run one request; never raise — wrap failures into RunResult.error."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release resources (HTTP clients, etc.). Override if needed."""
        return None
