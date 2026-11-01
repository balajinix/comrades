"""Synthetically widened TPC-H: the width axis with everything else held fixed.

TPC-H (61 columns) and TPC-DS (425) differ in width, but also in tables, joins and
workload. Widening one schema isolates width: the same TPC-H SF 1 data and the same
questions, with irrelevant columns added until the schema has 250, 500, 1,000 or
2,000 columns. Padding columns have plausible business names (``c_loyalty_tier``,
``o_fulfilment_region_code``), not ``pad_17``, so they distract the way real
enterprise columns do. They hold only NULLs, so the stores stay small.
"""
from __future__ import annotations

import random
from pathlib import Path

import duckdb

from .tpch import db_path

WIDTHS = (250, 500, 1000, 2000)
_PREFIX = {"customer": "c", "orders": "o", "lineitem": "l", "part": "p",
           "supplier": "s", "partsupp": "ps", "nation": "n", "region": "r"}
_A = ["loyalty", "billing", "fulfilment", "risk", "credit", "audit", "legacy", "region",
      "channel", "promo", "tax", "warehouse", "carrier", "contract", "portal", "crm",
      "erp", "invoice", "survey", "segment", "routing", "compliance", "pricing", "vendor",
      "returns", "support", "marketing", "dispatch", "quality", "forecast"]
_B = ["tier", "code", "flag", "score", "status", "date", "ref", "id", "amount", "rate",
      "note", "class", "level", "count", "type", "owner", "source", "window", "zone",
      "version"]
_TYPES = ["VARCHAR", "INTEGER", "DATE", "DECIMAL(12,2)", "BOOLEAN"]


def wide_path(width: int, sf: float = 1.0) -> Path:
    """Store path for TPC-H at ``sf`` widened to ``width`` columns."""
    return db_path(sf).with_name(f"tpch_sf{sf:g}_w{width}.duckdb")


def padding(width: int, base_columns: dict[str, int], seed: int = 0) -> dict[str, list]:
    """Padding columns per table: ``{table: [(name, type), ...]}``.

    Extra columns are spread over tables in proportion to their current width, and
    every name is unique within its table and absent from TPC-H.
    """
    extra = width - sum(base_columns.values())
    if extra < 0:
        raise ValueError(f"width {width} is below the base schema's "
                         f"{sum(base_columns.values())} columns")
    rng = random.Random(f"widen:{seed}")
    total = sum(base_columns.values())
    alloc = {t: extra * n // total for t, n in base_columns.items()}
    for t in sorted(base_columns, key=base_columns.get, reverse=True):
        if sum(alloc.values()) >= extra:
            break
        alloc[t] += 1
    out: dict[str, list] = {}
    for table, n in alloc.items():
        names: set[str] = set()
        cols = []
        while len(cols) < n:
            name = f"{_PREFIX[table]}_{rng.choice(_A)}_{rng.choice(_B)}"
            if name in names:
                name = f"{name}_{len(cols)}"
            names.add(name)
            cols.append((name, rng.choice(_TYPES)))
        out[table] = cols
    return out


def ensure_wide(width: int, sf: float = 1.0) -> Path:
    """Build (once) TPC-H at ``sf`` widened to ``width`` columns; return its path."""
    path = wide_path(width, sf)
    if path.exists():
        return path
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute(f"ATTACH '{db_path(sf)}' AS src (READ_ONLY)")
    base = {t: n for t, n in con.execute(
        "SELECT table_name, count(*) FROM information_schema.columns "
        "WHERE table_catalog = 'src' GROUP BY 1").fetchall()}
    for table, cols in padding(width, base).items():
        pads = "".join(f", NULL::{ty} AS {name}" for name, ty in cols)
        con.execute(f"CREATE TABLE {table} AS SELECT *{pads} FROM src.{table}")
    con.execute("DETACH src")
    con.close()
    tmp.rename(path)
    return path
