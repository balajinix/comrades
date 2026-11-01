"""Tests for the TPC-DS question workload and its gold answers."""

from __future__ import annotations

import re

import duckdb
import pytest

from dal_lab.tpcds import (
    FOREIGN_KEYS,
    TABLES,
    db_path,
    ensure_tpcds,
    fk_orphans,
    plan_answer,
    topk_boundary,
)
from dal_lab.workload import WORKLOAD_DIR, load_questions, same_answer

SMALLEST_SF = 0.01
WORKLOAD = WORKLOAD_DIR / "tpcds_questions.jsonl"
QUESTIONS = load_questions(WORKLOAD)
PLANNED = [q for q in QUESTIONS if q.plan]  # scripted (v1); v2 questions are LLM-only
FAMILIES = {
    "L": "lookup",
    "F": "filtered_list",
    "A": "aggregation",
    "T": "topk",
    "J": "join",
    "G": "large_list",
}


@pytest.fixture(scope="module")
def ds_con():
    """Read-only connection to TPC-DS at the smallest scale factor."""
    con = duckdb.connect(str(ensure_tpcds(SMALLEST_SF)), read_only=True)
    yield con
    con.close()


def _tables_in(sql: str) -> set[str]:
    return set(re.findall(r"\b(?:FROM|JOIN)\s+([a-z_]+)", sql)) & set(TABLES)


def test_load_questions_tpcds_has_four_per_family():
    # Arrange / Act
    ids = [q.id for q in QUESTIONS]
    # Assert: the 24 scripted (v1) questions plus 11 LLM-only (v2) ones
    assert len(QUESTIONS) == 35 and len(set(ids)) == 35
    assert len(PLANNED) == 24
    for prefix, family in FAMILIES.items():
        fam = [q for q in PLANNED if q.id.startswith(f"DS-{prefix}")]
        assert len(fam) == 4 and all(q.family == family for q in fam)
    assert all(q.workflow is None and q.workflow_params == {} for q in QUESTIONS)


def test_load_questions_tpcds_spans_broad_schema():
    # Arrange
    used = [_tables_in(q.gold_sql) for q in QUESTIONS]
    # Act
    wide = sum(len(t) >= 3 for t in used)
    covered = set().union(*used)
    # Assert
    assert wide >= 8
    assert {"store_sales", "catalog_sales", "web_sales", "inventory"} <= covered
    assert {"store_returns", "web_returns", "promotion", "date_dim"} <= covered


def test_load_questions_tpcds_plan_covers_every_gold_table():
    # Arrange / Act / Assert
    for q in PLANNED:
        planned = {step["table"] for step in q.plan}
        assert _tables_in(q.gold_sql) <= planned, q.id
        for step in q.plan:
            semis = [
                s
                for s in step["filters"].values()
                if isinstance(s, dict) and "in_from" in s
            ]
            assert len(semis) <= 1, f"{q.id}: one semi-join per step"


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q.id)
def test_gold_sql_runs_at_smallest_sf(ds_con, q):
    # Arrange / Act
    rows = ds_con.execute(q.gold_sql).fetchall()
    # Assert
    assert len(rows) >= 1
    if q.family != "large_list":
        assert len(rows) <= 50


@pytest.mark.parametrize("q", PLANNED, ids=lambda q: q.id)
def test_plan_answer_matches_gold_at_smallest_sf(ds_con, q):
    # Arrange
    gold = ds_con.execute(q.gold_sql).fetchall()
    # Act
    got = plan_answer(db_path(SMALLEST_SF), q.gold_sql, q.plan)
    # Assert
    assert same_answer(got, gold, q.ordered)


@pytest.mark.parametrize(
    "q", [q for q in QUESTIONS if q.family == "topk"], ids=lambda q: q.id
)
def test_topk_boundary_has_no_tie(ds_con, q):
    # Arrange: the ranking value is the last output column, the first ORDER BY key
    k = int(re.search(r"LIMIT (\d+)\s*$", q.gold_sql).group(1))
    # Act
    boundary = topk_boundary(ds_con, q.gold_sql, k)
    # Assert
    assert q.ordered
    assert boundary is not None, "fewer than k + 1 candidates"
    kth, next_ = boundary
    assert kth[-1] != next_[-1]


def test_fk_orphans_none_on_sample(ds_con):
    # Arrange
    sample = sorted(FOREIGN_KEYS.items())[::4]
    # Act
    orphans = {c: fk_orphans(ds_con, c, p, sample=2000) for c, p in sample}
    # Assert
    assert len(FOREIGN_KEYS) >= 80
    assert all(n == 0 for n in orphans.values()), orphans


def test_fk_orphans_detects_dangling_reference():
    # Arrange
    con = duckdb.connect()
    con.execute("CREATE TABLE p (k INTEGER); INSERT INTO p VALUES (1)")
    con.execute("CREATE TABLE c (fk INTEGER); INSERT INTO c VALUES (1), (2), (NULL)")
    # Act
    n = fk_orphans(con, ("c", "fk"), ("p", "k"))
    # Assert
    assert n == 1
