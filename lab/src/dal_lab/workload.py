"""TPC-H information needs with gold answers, plus the coded workflows.

Each question carries:

- ``gold_sql``: the reference query; executing it on the store gives the gold
  answer at every scale factor.
- ``family`` and ``size_class``: ``const`` (answer size independent of scale),
  ``const-out`` (small answer computed over data that grows with scale),
  ``linear-small`` / ``linear`` (answer grows with scale).
- ``plan``: what a fine-grained (per-table, filter-only) interface lets an agent
  push down: per step a table and its filters. ``{"in_from": "t.col"}`` is a
  semi-join on keys fetched in an earlier step.
- ``workflow`` / ``workflow_params``: the coded intent that answers it, if one
  exists. About half the questions have none, which models the long tail that a
  workflow catalogue never covers.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

WORKLOAD_DIR = Path(__file__).resolve().parents[2] / "workloads"

WORKFLOWS: dict[str, dict] = {
    "customer_profile": {
        "description": "Name, address and phone of a customer.",
        "params": ["custkey"],
        "sql": "SELECT c_name, c_address, c_phone FROM customer WHERE c_custkey = $custkey::INTEGER"},
    "part_info": {
        "description": "Retail price of a part.",
        "params": ["partkey"],
        "sql": "SELECT p_retailprice FROM part WHERE p_partkey = $partkey::INTEGER"},
    "order_summary": {
        "description": "Status and total price of an order.",
        "params": ["orderkey"],
        "sql": "SELECT o_orderstatus, o_totalprice FROM orders WHERE o_orderkey = $orderkey::INTEGER"},
    "customer_orders": {
        "description": "Orders of a customer with dates and totals.",
        "params": ["custkey"],
        "sql": "SELECT o_orderkey, o_orderdate, o_totalprice FROM orders WHERE o_custkey = $custkey::INTEGER"},
    "order_lines": {
        "description": "Line items of an order.",
        "params": ["orderkey"],
        "sql": "SELECT l_linenumber, l_partkey, l_quantity, l_extendedprice FROM lineitem "
               "WHERE l_orderkey = $orderkey::INTEGER"},
    "orders_per_year": {
        "description": "Number of orders placed in a year.",
        "params": ["year"],
        "sql": "SELECT count(*) AS n FROM orders WHERE year(o_orderdate) = $year::INTEGER"},
    "customers_by_segment": {
        "description": "Number of customers per market segment.",
        "params": [],
        "sql": "SELECT c_mktsegment, count(*) AS n FROM customer GROUP BY c_mktsegment"},
    "top_customers": {
        "description": "Top-k customers by total order value in a year.",
        "params": ["year", "k"],
        "sql": "SELECT c_custkey, c_name, sum(o_totalprice) AS total FROM orders "
               "JOIN customer ON o_custkey = c_custkey WHERE year(o_orderdate) = $year::INTEGER "
               "GROUP BY c_custkey, c_name ORDER BY total DESC, c_custkey LIMIT $k::INTEGER"},
    "largest_orders": {
        "description": "The k largest orders by total price.",
        "params": ["k"],
        "sql": "SELECT o_orderkey, o_totalprice FROM orders "
               "ORDER BY o_totalprice DESC, o_orderkey LIMIT $k::INTEGER"},
    "part_suppliers": {
        "description": "Suppliers of a part with their supply cost.",
        "params": ["partkey"],
        "sql": "SELECT s_suppkey, s_name, ps_supplycost FROM partsupp "
               "JOIN supplier ON ps_suppkey = s_suppkey WHERE ps_partkey = $partkey::INTEGER"},
    "revenue_by_region": {
        "description": "Revenue per region for orders placed in a year.",
        "params": ["year"],
        "sql": "SELECT r_name, sum(l_extendedprice * (1 - l_discount)) AS revenue "
               "FROM lineitem JOIN orders ON l_orderkey = o_orderkey "
               "JOIN customer ON o_custkey = c_custkey JOIN nation ON c_nationkey = n_nationkey "
               "JOIN region ON n_regionkey = r_regionkey "
               "WHERE year(o_orderdate) = $year::INTEGER GROUP BY r_name"},
    "shipments_on": {
        "description": "Line items shipped on a day.",
        "params": ["day"],
        "sql": "SELECT l_orderkey, l_linenumber, l_partkey, l_quantity FROM lineitem "
               "WHERE l_shipdate = $day::DATE"},
    "orders_on": {
        "description": "Orders placed on a day.",
        "params": ["day"],
        "sql": "SELECT o_orderkey, o_custkey, o_totalprice FROM orders WHERE o_orderdate = $day::DATE"},
}


@dataclass
class Question:
    """One information need."""

    id: str
    family: str
    size_class: str
    question: str
    gold_sql: str
    plan: list[dict]
    ordered: bool = False
    workflow: str | None = None
    workflow_params: dict = field(default_factory=dict)


def load_questions(path: Path | None = None) -> list[Question]:
    """Load the question set.

    Args:
        path: JSONL file (defaults to ``workloads/tpch_questions.jsonl``).

    Returns:
        Questions in file order.
    """
    path = path or WORKLOAD_DIR / "tpch_questions.jsonl"
    with path.open() as fh:
        return [Question(**json.loads(line)) for line in fh if line.strip()]


def gold_rows(con: duckdb.DuckDBPyConnection, q: Question) -> list[tuple]:
    """Execute the gold SQL.

    Args:
        con: Connection to the store.
        q: The question.

    Returns:
        Gold answer rows as tuples.
    """
    return con.execute(q.gold_sql).fetchall()


def _norm(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()[:10]
    if isinstance(v, str):
        return v.strip()
    return v


def _row_key(row: tuple) -> tuple:
    return tuple(round(v, 2) if isinstance(v, float) else v for v in map(_norm, row))


def _close(a: Any, b: Any) -> bool:
    a, b = _norm(a), _norm(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-2)
    return a == b


def same_answer(got: list[tuple], gold: list[tuple], ordered: bool) -> bool:
    """Compare an answer to gold, tolerating float rounding.

    Args:
        got: Answer rows.
        gold: Gold rows.
        ordered: Whether row order matters (top-k).

    Returns:
        True if equal as lists (ordered) or multisets (unordered).
    """
    if len(got) != len(gold):
        return False
    if not ordered:
        got, gold = sorted(got, key=_row_key), sorted(gold, key=_row_key)
    return all(len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
               for a, b in zip(got, gold))


def row_scores(got: list[tuple], gold: list[tuple]) -> tuple[float, float]:
    """Row-level precision and recall of an answer against gold (as multisets).

    Separates missing rows (low recall: truncation, partial results) from extra or
    invented rows (low precision: fabrication). Rows match under the same float
    tolerance as :func:`same_answer`, via a rounded key.

    Returns:
        (precision, recall); an empty answer has precision 1 and recall 0 unless the
        gold is empty too.
    """
    from collections import Counter

    try:
        g = Counter(_row_key(tuple(r)) for r in gold)
        a = Counter(_row_key(tuple(r)) for r in got)
    except TypeError:
        return 0.0, 0.0
    hit = sum((g & a).values())
    precision = hit / sum(a.values()) if a else 1.0
    recall = hit / sum(g.values()) if g else 1.0
    return round(precision, 4), round(recall, 4)

