"""Tests for the react, cot and reflect strategies of the data-access agent loop.

A scripted model and an in-memory tool stand in for the LLM and the MCP servers, so
the tests exercise only the loop's mechanics.
"""
from __future__ import annotations

import asyncio

import pytest

import kg_agent.react as react

TOOL = {"name": "get_cost", "server": "mcp_dal", "url": "http://x", "description": "cost",
        "compact_schema": "  month: string", "single_string_key": None}


class _Resp:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}


class _LLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return _Resp(self.replies.pop(0))


def _run(monkeypatch, replies, strategy, tool_result=None, max_steps=6):
    llm = _LLM(replies)
    calls = []

    async def list_tools(servers=None):
        return [TOOL]

    async def call_tool(name, args, url):
        calls.append((name, args))
        return tool_result or {"status": "ok", "data": [{"cost": 42}]}

    monkeypatch.setattr(react, "list_tools", list_tools)
    monkeypatch.setattr(react, "call_tool", call_tool)
    monkeypatch.setattr(react, "build_llm", lambda model=None: llm)
    trace, _, tokens = asyncio.run(react.run_react("cost?", max_steps=max_steps,
                                                   strategy=strategy))
    return trace, tokens, llm, calls


ACT = 'Thought: fetch it\nAction: get_cost\nAction Input: {"month": "2026-10"}'
ANSWER = "Thought: done\nFinal Answer: 42"


def test_run_react_default_strategy_unchanged(monkeypatch):
    # Arrange / Act
    trace, tokens, llm, calls = _run(monkeypatch, [ACT, ANSWER], None)
    # Assert
    assert [s.final_answer for s in trace] == [None, "42"]
    assert len(llm.calls) == 2 and tokens["total"] == 24 and len(calls) == 1


def test_run_react_cot_plans_first_and_counts_tokens(monkeypatch):
    trace, tokens, llm, calls = _run(monkeypatch, ["1. call get_cost", ACT, ANSWER], "cot")
    assert trace[0].thought.startswith("Plan: 1. call get_cost") and trace[0].action is None
    assert trace[-1].final_answer == "42" and tokens["total"] == 36 and len(calls) == 1
    roles = [type(m).__name__ for m in llm.calls[1]]
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))  # no repeated roles


def test_run_react_reflect_keeps_confirmed_answer(monkeypatch):
    trace, tokens, llm, _ = _run(monkeypatch,
                                 [ACT, ANSWER, "Thought: checks out\nFinal Answer: 42"],
                                 "reflect")
    assert trace[-1].final_answer == "42" and len(llm.calls) == 3
    assert any((s.observation or "").startswith("REFLECT") for s in trace)


def test_run_react_reflect_revises_answer(monkeypatch):
    trace, *_ = _run(monkeypatch, [ACT, ANSWER, "Thought: unit\nFinal Answer: 42 USD"],
                     "reflect")
    assert trace[-1].final_answer == "42 USD"


def test_run_react_reflect_reopened_repeat_stays_blocked(monkeypatch):
    # The reflection tries the identical call again: the loop guard must block it.
    trace, _, _, calls = _run(monkeypatch, [ACT, ANSWER, ACT, "Thought: ok\nFinal Answer: 42"],
                              "reflect")
    assert len(calls) == 1
    assert any("already called" in (s.observation or "") for s in trace)
    assert trace[-1].final_answer == "42"


def test_run_react_reflect_bare_critique_keeps_held_answer(monkeypatch):
    trace, *_ = _run(monkeypatch, [ACT, ANSWER, "Thought: the answer looks fine"], "reflect")
    assert trace[-1].final_answer == "42"


def test_run_react_rejects_unknown_strategy(monkeypatch):
    with pytest.raises(ValueError):
        _run(monkeypatch, [], "tot")


def test_strategy_reads_environment(monkeypatch):
    monkeypatch.setenv("COMRADES_AGENT_STRATEGY", "reflect")
    monkeypatch.setenv("COMRADES_MAX_REFLECTIONS", "2")
    assert react._strategy() == ("reflect", 2)
    monkeypatch.delenv("COMRADES_AGENT_STRATEGY")
    assert react._strategy()[0] == "react"
