"""Tests for the neutral ReAct agent, with a scripted stand-in for the model."""
from __future__ import annotations

from dal_lab.agent import _parse, run_agent
from dal_lab.llm import LLMGateway, TurnRecord
from dal_lab.meter import Meter


class FakeLLM(LLMGateway):
    """Replays fixed replies and records fake usage."""

    def __init__(self, replies: list[str], prompt_tokens: int = 100):
        super().__init__("fake")
        self.replies, self.prompt_tokens = list(replies), prompt_tokens

    def chat(self, messages):  # noqa: D401 - test double
        self.turns.append(TurnRecord(len(self.turns) + 1, self.prompt_tokens, 5, 0.0))
        return self.replies.pop(0)


def _tools(meter: Meter):
    return {"echo": meter.wrap("echo", lambda x: {"rows": [{"x": x}]})}


def test_parse_repairs_missing_braces():
    assert _parse('{"tool": "t", "args": {"a": {"b": 1}') == {"tool": "t", "args": {"a": {"b": 1}}}


def test_run_agent_calls_tool_then_answers():
    # Arrange
    meter = Meter()
    llm = FakeLLM(['{"tool": "echo", "args": {"x": 1}}', '{"answer": [[1]]}'])
    # Act
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm)
    # Assert
    assert run.status == "answered" and run.answer == [[1]]
    assert run.steps == 1 and run.data["calls"] == 1 and run.llm["turns"] == 2


def test_run_agent_unknown_tool_counts_error():
    meter = Meter()
    llm = FakeLLM(['{"tool": "nope", "args": {}}', '{"answer": []}'])
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm)
    assert run.tool_errors == 1 and run.status == "answered"


def test_run_agent_stops_on_context_limit():
    meter = Meter()
    llm = FakeLLM(['{"tool": "echo", "args": {"x": 1}}'], prompt_tokens=200_000)
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm, context_limit=120_000)
    assert run.status == "overflow"


def test_run_agent_rejects_handle_answer_when_not_allowed():
    meter = Meter()
    llm = FakeLLM(['{"answer_handle": "h1"}', '{"answer": [[2]]}'])
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm, allow_handles=False)
    assert run.answer == [[2]] and run.answer_handle is None
