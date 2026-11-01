"""Tests for the modality facades and observation compaction."""
from __future__ import annotations

import pytest

from dal_lab.facades import (CompactedFacade, FineGrainedFacade, GenericFacade, HandleFacade,
                             QueryAnswerFacade, compact)
from dal_lab.meter import Meter, to_json


def test_catalog_load_has_tpch_tables(catalog):
    # Arrange / Act
    tables = set(catalog.tables)
    # Assert
    assert {"lineitem", "orders", "customer", "nation"} <= tables
    assert catalog.key("customer") == "c_custkey"


def test_run_sql_caps_rows_and_flags_truncation(con, catalog):
    # Arrange
    fac = GenericFacade(con, catalog, max_rows=10)
    # Act
    out = fac.run_sql("SELECT * FROM lineitem")
    # Assert
    assert out["row_count"] == 10 and out["truncated"] is True


def test_run_sql_rejects_non_queries(con, catalog):
    # Arrange
    fac = GenericFacade(con, catalog)
    # Act / Assert
    with pytest.raises(ValueError):
        fac.run_sql("DROP TABLE nation")


def test_fine_grained_generates_two_tools_per_table(con, catalog):
    # Arrange / Act
    fac = FineGrainedFacade(con, catalog)
    # Assert
    assert len(fac.tools) == 2 * len(catalog.tables)
    assert {"get_orders", "list_orders"} <= set(fac.tools)


def test_list_tool_paginates_until_exhausted(con, catalog):
    # Arrange
    fac = FineGrainedFacade(con, catalog)
    lst = fac.tools["list_nation"]
    # Act
    first = lst(limit=10)
    rest = lst(limit=100, offset=first["next_offset"])
    # Assert
    assert len(first["rows"]) == 10 and first["next_offset"] == 10
    assert len(rest["rows"]) == 15 and rest["next_offset"] is None


def test_list_tool_applies_range_and_in_filters(con, catalog):
    # Arrange
    lst = FineGrainedFacade(con, catalog).tools["list_nation"]
    # Act
    out = lst(filters={"n_nationkey": {"ge": 3, "lt": 6}, "n_regionkey": {"in": [0, 1, 2, 3, 4]}})
    # Assert
    assert sorted(r["n_nationkey"] for r in out["rows"]) == [3, 4, 5]


def test_list_tool_rejects_unknown_field(con, catalog):
    # Arrange
    lst = FineGrainedFacade(con, catalog).tools["list_nation"]
    # Act / Assert
    with pytest.raises(ValueError):
        lst(filters={"bogus": 1})


def test_handle_facade_returns_count_and_preview(con, catalog):
    # Arrange
    fac = HandleFacade(con, catalog, preview_rows=5)
    # Act
    out = fac.run_query("SELECT * FROM customer")
    agg = fac.aggregate(out["handle"], "count")
    # Assert
    assert out["row_count"] == 1500 and len(out["preview"]) == 5
    assert "5 of 1500" in out["note"]
    assert agg["rows"][0]["value"] == 1500


def test_query_answer_facade_saves_full_result_without_rows(con, catalog):
    # Arrange
    fac = QueryAnswerFacade(con, catalog, max_rows=10)
    # Act
    page = fac.run_sql("SELECT * FROM customer")
    saved = fac.save_query("SELECT c_custkey FROM customer")
    stored = con.execute(f"SELECT count(*) FROM {fac.handles[saved['handle']]}").fetchone()[0]
    fac.release()
    # Assert
    assert page["truncated"] and page["row_count"] == 10
    assert saved["row_count"] == 1500 and saved["columns"] == ["c_custkey"]
    assert "rows" not in saved and stored == 1500 and not fac.handles


def test_generic_facade_truncation_modes_report_cap(con, catalog):
    # Arrange
    q = "SELECT * FROM customer"
    silent = GenericFacade(con, catalog, max_rows=10, truncation="silent")
    labelled = GenericFacade(con, catalog, max_rows=10, truncation="labelled")
    # Act
    quiet, loud = silent.run_sql(q), labelled.run_sql(q)
    # Assert
    assert silent.name == "sql+silent" and "truncated" not in quiet
    assert "up to" not in silent.descriptors[0]["description"]
    assert loud["truncated"] and loud["total_rows"] == 1500
    assert "INCOMPLETE" in loud["note"] and len(loud["rows"]) == 10


def test_compact_cuts_lists_uniformly_to_fit(con):
    # Arrange
    obj = {"rows": [{"i": i, "pad": "x" * 50} for i in range(500)], "tags": list(range(500))}
    # Act
    out = compact(obj, limit=4000)
    # Assert
    assert len(to_json(out)) <= 4000
    assert len(out["rows"]) == len(out["tags"]) < 500


def test_compact_leaves_small_results_untouched():
    # Arrange
    obj = {"rows": [1, 2, 3]}
    # Act / Assert
    assert compact(obj, limit=4000) is obj


def test_compacted_facade_meters_delivered_payload(con, catalog):
    # Arrange
    fac = CompactedFacade(GenericFacade(con, catalog), limit=1000)
    meter = Meter()
    tools = fac.bind(meter)
    # Act
    out = tools["run_sql"](query="SELECT * FROM customer LIMIT 100")
    # Assert
    assert len(out["rows"]) < 100
    assert meter.records[0].response_bytes <= 1000
