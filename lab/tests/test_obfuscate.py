"""Tests for the renamed (memorisation-control) TPC-H schema."""
from __future__ import annotations

from dal_lab.obfuscate import rename_question, rename_sql, rename_workflows
from dal_lab.workload import WORKFLOWS, load_questions


def test_rename_sql_renames_identifiers_not_literals():
    sql = "SELECT c_name FROM customer WHERE c_mktsegment = 'BUILDING' AND c_acctbal < 0"
    assert rename_sql(sql) == ("SELECT client_label FROM client WHERE industry = 'BUILDING' "
                               "AND client_balance < 0")


def test_rename_sql_leaves_no_tpch_identifier():
    import re
    for q in load_questions():
        assert not re.search(r"\b(lineitem|partsupp|[a-z]{1,2}_[a-z]+key)\b",
                             rename_question(q).gold_sql)
    for wf in rename_workflows(WORKFLOWS).values():
        assert "o_orderkey" not in wf["sql"] and "c_custkey" not in wf["sql"]


def test_rename_question_maps_plan_tables_and_filters():
    q = next(q for q in load_questions() if q.id == "G3")
    plan = rename_question(q).plan
    assert plan[0]["table"] == "client" and "industry" in plan[0]["filters"]
