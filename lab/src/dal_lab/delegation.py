"""Delegation to another agent (A2A-style) as a data-access modality.

The client agent has one tool, ``ask_data_agent(question)``. Behind it, a separate
data agent answers the question with its own tools (any facade) and returns only the
answer rows. The client's context stays small, but the work moves rather than
disappears: the data agent's tokens are metered separately and reported with the
client's, so the total cost of delegation is visible.

The hop is in-process; ICSE RQ3 showed the transport itself (in-process, MCP, A2A)
is second-order, so only the delegation pattern is modelled here.
"""
from __future__ import annotations

from dataclasses import dataclass

from .agent import run_agent
from .facades import Facade
from .llm import LLMGateway
from .meter import Meter


@dataclass
class DelegateCost:
    """Cumulative cost incurred by the data agent behind the client's calls."""

    calls: int = 0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    data_calls: int = 0

    def reset(self) -> None:
        """Zero the counters before a new client run."""
        self.calls = self.turns = self.prompt_tokens = self.completion_tokens = 0
        self.data_calls = 0


class DelegationFacade(Facade):
    """One ``ask_data_agent`` tool backed by a data agent over ``inner``."""

    def __init__(self, inner: Facade, model: str, allow_handles: bool = False) -> None:
        super().__init__(f"a2a:{inner.name}")
        self.inner, self.allow_handles = inner, allow_handles
        self.llm = LLMGateway(model, max_tokens=4096)
        self.cost = DelegateCost()
        self._add("ask_data_agent", self.ask,
                  "Ask the data agent a self-contained data question in natural language. "
                  "It has direct access to the data and returns the answer rows. State "
                  "the answer columns you need.",
                  {"question": {"type": "string"},
                   "columns": {"type": "array", "items": {"type": "string"}}},
                  ["question"])

    def ask(self, question: str, columns: list[str] | None = None) -> dict:
        """Run the data agent on ``question`` and return its answer.

        Args:
            question: The delegated question.
            columns: Answer columns the client wants, if it said.

        Returns:
            ``{"status", "rows"}`` as an A2A task result would carry them.
        """
        meter = Meter()
        run = run_agent(question, list(columns or []), self.inner.descriptors,
                        self.inner.bind(meter), meter, self.llm,
                        allow_handles=self.allow_handles)
        totals = run.llm
        self.cost.calls += 1
        self.cost.turns += int(totals.get("turns", 0))
        self.cost.prompt_tokens += int(totals.get("prompt_tokens", 0))
        self.cost.completion_tokens += int(totals.get("completion_tokens", 0))
        self.cost.data_calls += int(run.data.get("calls", 0))
        if run.status != "answered":
            return {"status": "failed", "reason": run.status}
        return {"status": "completed", "rows": run.answer}

