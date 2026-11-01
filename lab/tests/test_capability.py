"""Tests for capability-matched fine-grained tools and provenance-checked code mode."""
from __future__ import annotations

import pytest

from dal_lab.capability import FineGrainedOpsFacade
from dal_lab.code_mode import CodeFacade
from dal_lab.facades import HandleFacade


def test_ops_join_follows_foreign_keys(con, catalog):
    fac = FineGrainedOpsFacade(con, catalog)
    out = fac.join([{"table": "customer"}, {"table": "nation"}],
                   ["customer.c_name", "nation.n_name"],
                   {"customer.c_custkey": 1})
    gold = con.execute("SELECT c_name, n_name FROM customer JOIN nation "
                       "ON c_nationkey = n_nationkey WHERE c_custkey = 1").fetchall()
    assert [tuple(r.values()) for r in out["rows"]] == gold


def test_ops_aggregate_topk_runs_on_server(con, catalog):
    fac = FineGrainedOpsFacade(con, catalog)
    out = fac.aggregate([{"table": "customer"}, {"table": "nation"}],
                        [{"op": "count", "column": "*"}], group_by=["nation.n_name"],
                        order_by="m0", desc=True, limit=3)
    gold = con.execute("SELECT n_name, count(*) FROM customer JOIN nation "
                       "ON c_nationkey = n_nationkey GROUP BY 1 ORDER BY 2 DESC "
                       "LIMIT 3").fetchall()
    assert [r["m0"] for r in out["rows"]] == [g[1] for g in gold]


def test_ops_rejects_unknown_columns_and_unlinked_tables(con, catalog):
    fac = FineGrainedOpsFacade(con, catalog)
    with pytest.raises(ValueError):
        fac.join([{"table": "customer"}], ["customer.nope"])
    with pytest.raises(ValueError):
        fac.join([{"table": "region"}, {"table": "lineitem"}], ["region.r_name"])


def test_ops_value_filters_are_bound_not_spliced(con, catalog):
    fac = FineGrainedOpsFacade(con, catalog)
    out = fac.join([{"table": "nation"}], ["nation.n_name"],
                   {"nation.n_name": "x' OR '1'='1"})
    assert out["rows"] == []


def test_code_prov_rejects_invented_values(con, catalog):
    fac = CodeFacade(con, catalog, HandleFacade(con, catalog), provenance=True)
    out = fac.run_python("sql('SELECT 1')\nsave_answer([['Customer#999999999']])")
    assert "provenance" in out.get("stderr", "") and "handle" not in out


def test_code_prov_rejects_redefined_sql(con, catalog):
    fac = CodeFacade(con, catalog, HandleFacade(con, catalog), provenance=True)
    out = fac.run_python("def sql(q):\n    return [{'a': 'x'}]\n"
                         "save_answer(sql('SELECT 1'))")
    assert "redefined" in out.get("stderr", "")


def test_code_prov_accepts_database_values(con, catalog):
    fac = CodeFacade(con, catalog, HandleFacade(con, catalog), provenance=True)
    out = fac.run_python("rows = sql('SELECT c_name, c_acctbal FROM customer "
                         "WHERE c_custkey < 4')\nsave_answer(rows)")
    assert out.get("saved_rows") == 3
