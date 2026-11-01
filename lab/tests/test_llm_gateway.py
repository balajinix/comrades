"""Tests for routing, logging, temperature fallback and the cost cap (no network)."""
from __future__ import annotations

import json
import sys
import types

import pytest

from dal_lab import llm


def _fake_litellm(calls: list, reject_temperature: bool = False):
    def completion(**kw):
        calls.append(kw)
        if reject_temperature and "temperature" in kw:
            raise ValueError("Unsupported value: 'temperature' does not support 0")
        usage = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                                      completion_tokens_details=None)
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content="{}"),
                                       finish_reason="stop")
        return types.SimpleNamespace(usage=usage, choices=[choice])
    return types.SimpleNamespace(completion=completion, suppress_debug_info=False)


@pytest.fixture
def gateway_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gw.example")
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_TOKEN_LOG", str(tmp_path / "tok.jsonl"))
    monkeypatch.setattr(llm, "_PRICES", {"aws/m": (1e-6, 5e-6)})
    return tmp_path / "tok.jsonl"


def test_gateway_routes_logs_and_prices(gateway_env, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm(calls))
    g = llm.LLMGateway("gateway/aws/m", max_tokens=4096)
    g.chat([{"role": "user", "content": "hi"}])
    kw = calls[0]
    assert kw["model"] == "openai/aws/m" and kw["api_base"] == "https://gw.example/v1"
    assert kw["max_completion_tokens"] == 4096 and "max_tokens" not in kw
    rec = json.loads(gateway_env.read_text())
    assert rec["cost_usd"] == pytest.approx(1000 * 1e-6 + 200 * 5e-6)


def test_gateway_drops_rejected_temperature_once(gateway_env, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm(calls, True))
    g = llm.LLMGateway("gateway/aws/m")
    g.chat([{"role": "user", "content": "a"}])
    g.chat([{"role": "user", "content": "b"}])
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    assert "temperature" not in calls[2]


def test_gateway_stops_at_the_cost_cap(gateway_env, monkeypatch):
    gateway_env.write_text(json.dumps({"cost_usd": 30.0}) + "\n")
    monkeypatch.setenv("LITELLM_COST_CAP_USD", "25")
    monkeypatch.setitem(sys.modules, "litellm", _fake_litellm([]))
    with pytest.raises(llm.CostCapReached):
        llm.LLMGateway("gateway/aws/m").chat([{"role": "user", "content": "x"}])


def test_native_tool_call_becomes_json_action():
    fn = types.SimpleNamespace(name="run_sql", arguments='{"query": "SELECT 1"}')
    msg = types.SimpleNamespace(content=None,
                                tool_calls=[types.SimpleNamespace(function=fn)])
    assert json.loads(llm._as_text(msg)) == {"tool": "run_sql",
                                             "args": {"query": "SELECT 1"}}


def test_chat_waits_out_rate_limits(monkeypatch):
    class RateLimitError(Exception):
        pass

    calls, sleeps = [], []
    fake = _fake_litellm(calls)
    ok = fake.completion

    def flaky(**kw):
        if len(calls) < 2:
            calls.append(kw)
            raise RateLimitError("Rate limit of 8 requests per 1s was reached")
        return ok(**kw)

    fake.completion, fake.RateLimitError = flaky, RateLimitError
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    g = llm.LLMGateway("provider/m")
    assert g.chat([{"role": "user", "content": "hi"}]) == "{}"
    assert len(calls) == 3 and len(sleeps) == 2 and len(g.turns) == 1
