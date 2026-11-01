"""Tests for the ITBench-Lite loader (skipped when the corpus is not downloaded)."""
from __future__ import annotations

import duckdb
import pytest

from dal_lab import itbench

pytestmark = pytest.mark.skipif(1 not in itbench.scenario_dirs(),
                                reason="ITBench-Lite Scenario-1 not downloaded")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    db = itbench.build(tmp_path_factory.mktemp("itb") / "s.duckdb", [1])
    con = duckdb.connect(str(db), read_only=True)
    yield con
    con.close()


def test_itbench_loads_every_source(store):
    for table in ("scenario", "k8s_object", "k8s_event", "span", "alert", "log"):
        assert store.execute(f"SELECT count(*) FROM {table}").fetchone()[0] > 0, table


def test_itbench_ground_truth_parsed(store):
    assert store.execute("SELECT fault_kind, root_cause_namespace FROM scenario"
                         ).fetchone() == ("Pod", "otel-demo")


def test_itbench_uid_joins_resolve(store):
    spans, joined = store.execute(
        "SELECT count(*), count(o.uid) FROM span s LEFT JOIN k8s_object o "
        "ON s.pod_uid = o.uid").fetchone()
    assert joined == spans
    pods, placed = store.execute(
        "SELECT count(*), count(node_uid) FROM k8s_object WHERE kind = 'Pod'").fetchone()
    assert placed == pods


def test_itbench_service_call_graph(store):
    edges = store.execute(
        "SELECT DISTINCT p.service, c.service FROM span c JOIN span p "
        "ON c.parent_span_id = p.span_id AND c.trace_id = p.trace_id "
        "WHERE p.service <> c.service").fetchall()
    assert ("frontend", "cart") in edges


def test_itbench_foreign_keys_name_real_columns(store):
    cols = {(t, c) for t, c in store.execute(
        "SELECT table_name, column_name FROM information_schema.columns").fetchall()}
    for (ct, cc), (pt, pc) in itbench.FOREIGN_KEYS.items():
        assert (ct, cc) in cols and (pt, pc) in cols


def test_itbench_data_dictionary_reaches_the_schema_text(store, tmp_path):
    from dal_lab.facades import Catalog, open_store

    db = itbench.build(tmp_path / "n.duckdb", [1])
    cat = Catalog.load(open_store(db))
    assert "span.status: 'Ok', 'Error'" in cat.describe()


def test_tpch_schema_text_has_no_notes(con, catalog):
    assert "Column notes" not in catalog.describe()
