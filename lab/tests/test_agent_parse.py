"""Tests for parsing the model's JSON action."""
from __future__ import annotations

from dal_lab.agent import _parse


def test_parse_takes_the_first_action_when_two_are_sent():
    text = '{"tool": "run_sql", "args": {"query": "x"}}\n{"answer": [["invented"]]}'
    assert _parse(text) == {"tool": "run_sql", "args": {"query": "x"}}


def test_parse_repairs_dropped_closing_braces():
    assert _parse('{"tool": "a", "args": {"b": {"c": 1}}') == {"tool": "a",
                                                                "args": {"b": {"c": 1}}}


def test_parse_reads_fenced_json():
    assert _parse('```json\n{"answer": [[1]]}\n```') == {"answer": [[1]]}


def test_parse_returns_none_without_json():
    assert _parse("I cannot answer that.") is None


def test_normalise_accepts_nested_string_and_namespaced_actions():
    from dal_lab.agent import _normalise

    tools = {"run_sql": None}
    q = {"query": "SELECT 1"}
    assert _normalise({"tool": {"name": "run_sql", "args": q}}, tools) == ("run_sql", q)
    assert _normalise({"tool": "run_sql", "args": '{"query": "SELECT 1"}'}, tools) == (
        "run_sql", q)
    assert _normalise({"tool": "repo_browser.run_sql", "args": q}, tools) == ("run_sql", q)
    assert _normalise({"tool": "repo_browser.run_code", "args": {}}, tools)[0] == (
        "repo_browser.run_code")
