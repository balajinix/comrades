"""Tests for the skills (progressive disclosure) facade."""
from __future__ import annotations

import pytest

from dal_lab.meter import count_tokens
from dal_lab.skills import SkillsFacade


def test_skills_index_lists_domains_and_body_has_schema(con, catalog):
    fac = SkillsFacade(con, catalog)
    names = [s["name"] for s in fac.list_skills()["skills"]]
    body = fac.load_skill("customers-and-orders")["body"]
    assert "geography" in names
    assert "customer(c_custkey" in body and "o_custkey = customer.c_custkey" in body


def test_skills_sql_tool_carries_no_schema(con, catalog):
    fac = SkillsFacade(con, catalog)
    run_sql = next(d for d in fac.descriptors if d["name"] == "run_sql")
    assert "c_custkey" not in run_sql["description"]
    assert count_tokens(str(fac.descriptors)) < 600


def test_skills_unknown_name_raises(con, catalog):
    with pytest.raises(ValueError):
        SkillsFacade(con, catalog).load_skill("nope")
