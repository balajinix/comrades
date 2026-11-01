"""Tests for the chain-of-thought and self-reflection agent strategies."""
from __future__ import annotations

import pytest

from dal_lab.agent import run_agent
from dal_lab.meter import Meter
from test_agent import FakeLLM, _tools


class RecordingLLM(FakeLLM):
    """FakeLLM that also keeps the messages of every call."""

    def __init__(self, replies, prompt_tokens=100):
        super().__init__(replies, prompt_tokens)
        self.calls = []

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        return super().chat(messages)


def test_run_agent_cot_plans_before_acting():
    # Arrange
    meter = Meter()
    llm = RecordingLLM(["1. call echo 2. answer", '{"tool": "echo", "args": {"x": 1}}',
                        '{"answer": [[1]]}'])
    # Act
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm, strategy="cot")
    # Assert
    assert run.status == "answered" and run.answer == [[1]]
    assert run.transcript[0] == {"plan": "1. call echo 2. answer"}
    assert run.llm["turns"] == 3 and run.steps == 1
    assert [m["role"] for m in llm.calls[0]] == ["system", "user"]
    roles = [m["role"] for m in llm.calls[1]]
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))  # no repeated roles


def test_run_agent_reflect_keeps_confirmed_answer():
    meter = Meter()
    llm = FakeLLM(['{"answer": [[1]]}', '{"reflection": "looks right", "answer": [[1]]}'])
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm, strategy="reflect")
    assert run.status == "answered" and run.answer == [[1]]
    assert run.reflections == 1 and not run.revised
    assert run.transcript[-1]["reflection"] == "looks right"


def test_run_agent_reflect_revises_answer():
    meter = Meter()
    llm = FakeLLM(['{"answer": [[1]]}', '{"reflection": "wrong", "answer": [[2]]}'])
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm, strategy="reflect")
    assert run.answer == [[2]] and run.revised


def test_run_agent_reflect_can_reopen_search():
    meter = Meter()
    llm = FakeLLM(['{"answer": []}', '{"reflection": "missing", "tool": "echo", "args": {"x": 3}}',
                   '{"answer": [[3]]}'])
    run = run_agent("q", ["x"], [], _tools(meter), meter, llm, strategy="reflect")
    assert run.answer == [[3]] and run.revised and run.steps == 1 and run.reflections == 1


def test_run_agent_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        run_agent("q", ["x"], [], {}, Meter(), FakeLLM([]), strategy="tot")
