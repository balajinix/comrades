"""A renamed copy of TPC-H, to control for models having memorised its schema.

Models know ``lineitem.l_shipdate`` and ``customer.c_mktsegment`` from pretraining,
so on plain TPC-H a generic SQL tool may look better than it would on an unfamiliar
enterprise schema, and schema delivery (DDL in the prompt, skills, on-demand
description) looks free. This module builds the same data under plausible but
unfamiliar names: every table and column is renamed (globally unique, so joins stay
unambiguous) and the generated entity labels (``Customer#...``, ``Supplier#...``,
``Clerk#...``) get new prefixes. Question wording is unchanged, so the model has to
link user vocabulary ("market segment") to the store's names (``industry``), as it
would on a real enterprise schema.

Gold queries, fine-grained plans and coded workflows are rewritten mechanically, so
the gold answer is the same data under new names.
"""
from __future__ import annotations

import copy
import re
from dataclasses import replace
from pathlib import Path

import duckdb

from .tpch import DATA_DIR, ensure_tpch
from .workload import Question

TABLE_MAP = {
    "region": "zone", "nation": "country", "supplier": "vendor", "customer": "client",
    "part": "item", "partsupp": "sourcing", "orders": "purchase",
    "lineitem": "purchase_line",
}

COLUMN_MAP = {
    "r_regionkey": "zone_id", "r_name": "zone_label", "r_comment": "zone_notes",
    "n_nationkey": "country_id", "n_name": "country_label", "n_regionkey": "zone_ref",
    "n_comment": "country_notes",
    "s_suppkey": "vendor_id", "s_name": "vendor_label", "s_address": "vendor_street",
    "s_nationkey": "vendor_country_ref", "s_phone": "vendor_contact",
    "s_acctbal": "vendor_balance", "s_comment": "vendor_notes",
    "c_custkey": "client_id", "c_name": "client_label", "c_address": "client_street",
    "c_nationkey": "client_country_ref", "c_phone": "client_contact",
    "c_acctbal": "client_balance", "c_mktsegment": "industry", "c_comment": "client_notes",
    "p_partkey": "item_id", "p_name": "item_label", "p_mfgr": "maker",
    "p_brand": "item_line", "p_type": "item_kind", "p_size": "item_dim",
    "p_container": "packaging", "p_retailprice": "list_amount", "p_comment": "item_notes",
    "ps_partkey": "src_item_ref", "ps_suppkey": "src_vendor_ref",
    "ps_availqty": "stock_units", "ps_supplycost": "unit_cost",
    "ps_comment": "sourcing_notes",
    "o_orderkey": "purchase_id", "o_custkey": "buyer_ref", "o_orderstatus": "purchase_state",
    "o_totalprice": "purchase_amount", "o_orderdate": "placed_on",
    "o_orderpriority": "urgency", "o_clerk": "handled_by", "o_shippriority": "dispatch_rank",
    "o_comment": "purchase_notes",
    "l_orderkey": "line_purchase_ref", "l_partkey": "line_item_ref",
    "l_suppkey": "line_vendor_ref", "l_linenumber": "line_no", "l_quantity": "units",
    "l_extendedprice": "gross_amount", "l_discount": "rebate", "l_tax": "levy",
    "l_returnflag": "return_code", "l_linestatus": "line_state",
    "l_shipdate": "dispatched_on", "l_commitdate": "promised_on",
    "l_receiptdate": "delivered_on", "l_shipinstruct": "handling_note",
    "l_shipmode": "carrier_mode", "l_comment": "line_notes",
}

# Generated labels that identify TPC-H on sight; values are rewritten consistently.
LABEL_PREFIXES = {
    ("customer", "c_name"): ("Customer#", "Client#"),
    ("supplier", "s_name"): ("Supplier#", "Vendor#"),
    ("orders", "o_clerk"): ("Clerk#", "Agent#"),
}

_NAMES = {**TABLE_MAP, **COLUMN_MAP}
# Lowercase identifiers only (case-sensitive), so uppercase string literals such as
# 'GERMANY' or 'BUILDING' are never touched.
_IDENT = re.compile(r"\b(" + "|".join(sorted(_NAMES, key=len, reverse=True)) + r")\b")


def obf_path(sf: float) -> Path:
    """Return the DuckDB file path of the renamed store at ``sf``."""
    return DATA_DIR / f"tpch_sf{sf:g}_obf.duckdb"


def rename_sql(sql: str) -> str:
    """Rewrite TPC-H identifiers in ``sql`` to the renamed schema.

    Args:
        sql: A query over plain TPC-H.

    Returns:
        The same query over the renamed schema; literals are unchanged.
    """
    return _IDENT.sub(lambda m: _NAMES[m.group(1)], sql)


def _relabel(table: str, column: str, value: str) -> str:
    old, new = LABEL_PREFIXES.get((table, column), (None, None))
    return value.replace(old, new) if old else value


def rename_question(q: Question) -> Question:
    """Rewrite a question's gold SQL, plan and workflow binding for the renamed store.

    Literal filter values in the plan keep their data, except relabelled entity names.
    """
    plan = []
    for step in copy.deepcopy(q.plan):
        table = step["table"]
        filters = {}
        for col, val in step.get("filters", {}).items():
            if isinstance(val, dict) and "in_from" in val:
                val = {**val, "in_from": rename_sql(val["in_from"])}
            elif isinstance(val, str):
                val = _relabel(table, col, val)
            filters[COLUMN_MAP.get(col, col)] = val
        plan.append({**step, "table": TABLE_MAP[table], "filters": filters})
    return replace(q, gold_sql=rename_sql(q.gold_sql), plan=plan)


def rename_workflows(workflows: dict[str, dict]) -> dict[str, dict]:
    """Rewrite the coded workflows' SQL for the renamed store."""
    return {name: {**wf, "sql": rename_sql(wf["sql"])} for name, wf in workflows.items()}


def ensure_obfuscated(sf: float) -> Path:
    """Build the renamed store at ``sf`` once (from plain TPC-H) and return its path.

    Args:
        sf: TPC-H scale factor.

    Returns:
        Path to a DuckDB file with the renamed tables.
    """
    path = obf_path(sf)
    if path.exists():
        return path
    src = ensure_tpch(sf)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
        for table, new_table in TABLE_MAP.items():
            cols = [r[0] for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_catalog = 'src' AND table_name = ? ORDER BY ordinal_position",
                [table]).fetchall()]
            exprs = []
            for col in cols:
                expr = col
                if (table, col) in LABEL_PREFIXES:
                    old, new = LABEL_PREFIXES[(table, col)]
                    expr = f"replace({col}, '{old}', '{new}')"
                exprs.append(f"{expr} AS {COLUMN_MAP[col]}")
            con.execute(f"CREATE TABLE {new_table} AS SELECT {', '.join(exprs)} "
                        f"FROM src.{table}")
        con.execute("DETACH src")
    finally:
        con.close()
    tmp.rename(path)
    return path
