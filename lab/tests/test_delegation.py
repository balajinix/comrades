"""Tests for the delegation (A2A-style) facade, with a scripted data agent."""
from __future__ import annotations

from dal_lab.delegation import DelegationFacade
from dal_lab.facades import GenericFacade

from test_agent import FakeLLM


def test_delegation_returns_rows_and_meters_the_data_agent(con, catalog):
    # Arrange
    fac = DelegationFacade(GenericFacade(con, catalog), "fake")
    fac.llm = FakeLLM(['{"tool": "run_sql", "args": {"query": "SELECT count(*) FROM nation"}}',
                       '{"answer": [[25]]}'], prompt_tokens=50)
    # Act
    out = fac.ask("How many nations?", ["n"])
    # Assert
    assert out == {"status": "completed", "rows": [[25]]}
    assert fac.cost.turns == 2 and fac.cost.prompt_tokens == 100 and fac.cost.data_calls == 1
