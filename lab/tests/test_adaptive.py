"""Tests for the adaptive interfaces: auto-handles, on-demand schema, router."""
from __future__ import annotations

from dal_lab.adaptive import AutoHandleFacade, OnDemandSchemaFacade, router_facade
from dal_lab.facades import GenericFacade, HandleFacade, WorkflowFacade
from dal_lab.workload import WORKFLOWS


def test_auto_returns_small_results_inline(con, catalog):
    fac = AutoHandleFacade(con, catalog, HandleFacade(con, catalog))
    out = fac.run_sql("SELECT n_name FROM nation WHERE n_regionkey = 0")
    assert "handle" not in out and out["row_count"] == len(out["rows"]) == 5


def test_auto_returns_large_results_by_handle(con, catalog):
    handles = HandleFacade(con, catalog)
    fac = AutoHandleFacade(con, catalog, handles)
    out = fac.run_sql("SELECT c_custkey FROM customer")
    assert out["row_count"] == 1500 and len(out["preview"]) == 5
    assert handles.table_of(out["handle"])


def test_on_demand_schema_hides_schema_until_asked(con, catalog):
    fac = OnDemandSchemaFacade(con, catalog, GenericFacade(con, catalog))
    run_sql = next(d for d in fac.descriptors if d["name"] == "run_sql")
    assert "lineitem(" not in run_sql["description"]
    desc = fac.tools["describe_table"]("orders")
    assert any(c["column"] == "o_custkey" for c in desc["columns"])
    assert any("o_custkey -> customer.c_custkey" in f for f in desc["foreign_keys"])


def test_on_demand_schema_serves_show_and_describe(con, catalog):
    fac = OnDemandSchemaFacade(con, catalog, GenericFacade(con, catalog))
    tables = fac.tools["run_sql"]("SHOW TABLES;")["tables"]
    assert {"table": "orders", "columns": 9} in tables
    cols = fac.tools["run_sql"]("DESCRIBE Orders")["columns"]
    assert any(c["column"] == "o_custkey" for c in cols)
    assert fac.tools["run_sql"]("SELECT count(*) AS n FROM region")["rows"] == [{"n": 5}]


def test_on_demand_schema_points_other_statements_to_schema_tools(con, catalog):
    fac = AutoHandleFacade(con, catalog, HandleFacade(con, catalog), schema="on_demand")
    assert len(fac.tools["run_sql"]("show tables")["tables"]) == len(catalog.tables)
    try:
        fac.tools["run_sql"]("PRAGMA table_info('orders')")
    except ValueError as exc:
        assert "describe_table" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_inline_schema_keeps_rejecting_show_tables(con, catalog):
    fac = AutoHandleFacade(con, catalog, HandleFacade(con, catalog))
    try:
        fac.run_sql("SHOW TABLES")
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_router_combines_workflows_and_adaptive_sql(con, catalog):
    fac = router_facade(con, catalog, HandleFacade(con, catalog),
                        WorkflowFacade(con, WORKFLOWS))
    names = set(fac.tools)
    assert set(WORKFLOWS) <= names and {"run_sql", "fetch_rows"} <= names
    # TPC-H's schema is small, so the router inlines it.
    assert "lineitem(" in next(d for d in fac.descriptors
                               if d["name"] == "run_sql")["description"]


def test_lazy_handles_materialise_nothing(con, catalog):
    from dal_lab.meter import Meter, duckdb_probe

    lazy = HandleFacade(con, catalog, lazy=True)
    meter = Meter(db_probe=duckdb_probe(con))
    out = lazy.bind(meter)["run_query"](query="SELECT * FROM lineitem")
    assert out["row_count"] > 0
    assert con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = ?",
                       [out["handle"]]).fetchone()[0] == 1
    assert meter.totals()["db_temp_table_peak_bytes"] == 0


def test_materialised_handles_show_up_in_db_memory(con, catalog):
    from dal_lab.meter import Meter, duckdb_probe

    eager = HandleFacade(con, catalog)
    meter = Meter(db_probe=duckdb_probe(con))
    eager.bind(meter)["run_query"](query="SELECT * FROM lineitem")
    assert meter.totals()["db_temp_table_peak_bytes"] > 0
