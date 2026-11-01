"""TPC-DS data at several scale factors, stored as DuckDB files.

TPC-H (8 tables) is the narrow-schema base of the study; TPC-DS (24 tables:
7 fact tables over 17 dimensions, three sales channels plus returns and
inventory) is the next rung of the schema-breadth ladder. Data comes from
DuckDB's ``tpcds`` extension (``CALL dsdgen(sf=...)``), which is deterministic
and accepts fractional scale factors, so every run of the lab sees identical
rows.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

LAB_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = LAB_ROOT / "data"

FACT_TABLES = (
    "store_sales",
    "store_returns",
    "catalog_sales",
    "catalog_returns",
    "web_sales",
    "web_returns",
    "inventory",
)

DIMENSION_TABLES = (
    "call_center",
    "catalog_page",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "time_dim",
    "warehouse",
    "web_page",
    "web_site",
)

TABLES = FACT_TABLES + DIMENSION_TABLES

# (child table, child column) -> (parent table, parent column). Derived from the
# TPC-DS naming convention: a ``*_sk`` column references the dimension whose
# primary key carries the same suffix (``ss_customer_sk`` -> ``c_customer_sk``).
# Written out literally so the schema graph is reviewable and greppable.
FOREIGN_KEYS: dict[tuple[str, str], tuple[str, str]] = {
    # store_sales
    ("store_sales", "ss_sold_date_sk"): ("date_dim", "d_date_sk"),
    ("store_sales", "ss_sold_time_sk"): ("time_dim", "t_time_sk"),
    ("store_sales", "ss_item_sk"): ("item", "i_item_sk"),
    ("store_sales", "ss_customer_sk"): ("customer", "c_customer_sk"),
    ("store_sales", "ss_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("store_sales", "ss_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("store_sales", "ss_addr_sk"): ("customer_address", "ca_address_sk"),
    ("store_sales", "ss_store_sk"): ("store", "s_store_sk"),
    ("store_sales", "ss_promo_sk"): ("promotion", "p_promo_sk"),
    # store_returns
    ("store_returns", "sr_returned_date_sk"): ("date_dim", "d_date_sk"),
    ("store_returns", "sr_return_time_sk"): ("time_dim", "t_time_sk"),
    ("store_returns", "sr_item_sk"): ("item", "i_item_sk"),
    ("store_returns", "sr_customer_sk"): ("customer", "c_customer_sk"),
    ("store_returns", "sr_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("store_returns", "sr_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("store_returns", "sr_addr_sk"): ("customer_address", "ca_address_sk"),
    ("store_returns", "sr_store_sk"): ("store", "s_store_sk"),
    ("store_returns", "sr_reason_sk"): ("reason", "r_reason_sk"),
    # catalog_sales
    ("catalog_sales", "cs_sold_date_sk"): ("date_dim", "d_date_sk"),
    ("catalog_sales", "cs_sold_time_sk"): ("time_dim", "t_time_sk"),
    ("catalog_sales", "cs_ship_date_sk"): ("date_dim", "d_date_sk"),
    ("catalog_sales", "cs_bill_customer_sk"): ("customer", "c_customer_sk"),
    ("catalog_sales", "cs_bill_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("catalog_sales", "cs_bill_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("catalog_sales", "cs_bill_addr_sk"): ("customer_address", "ca_address_sk"),
    ("catalog_sales", "cs_ship_customer_sk"): ("customer", "c_customer_sk"),
    ("catalog_sales", "cs_ship_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("catalog_sales", "cs_ship_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("catalog_sales", "cs_ship_addr_sk"): ("customer_address", "ca_address_sk"),
    ("catalog_sales", "cs_call_center_sk"): ("call_center", "cc_call_center_sk"),
    ("catalog_sales", "cs_catalog_page_sk"): ("catalog_page", "cp_catalog_page_sk"),
    ("catalog_sales", "cs_ship_mode_sk"): ("ship_mode", "sm_ship_mode_sk"),
    ("catalog_sales", "cs_warehouse_sk"): ("warehouse", "w_warehouse_sk"),
    ("catalog_sales", "cs_item_sk"): ("item", "i_item_sk"),
    ("catalog_sales", "cs_promo_sk"): ("promotion", "p_promo_sk"),
    # catalog_returns
    ("catalog_returns", "cr_returned_date_sk"): ("date_dim", "d_date_sk"),
    ("catalog_returns", "cr_returned_time_sk"): ("time_dim", "t_time_sk"),
    ("catalog_returns", "cr_item_sk"): ("item", "i_item_sk"),
    ("catalog_returns", "cr_refunded_customer_sk"): ("customer", "c_customer_sk"),
    ("catalog_returns", "cr_refunded_cdemo_sk"): (
        "customer_demographics",
        "cd_demo_sk",
    ),
    ("catalog_returns", "cr_refunded_hdemo_sk"): (
        "household_demographics",
        "hd_demo_sk",
    ),
    ("catalog_returns", "cr_refunded_addr_sk"): ("customer_address", "ca_address_sk"),
    ("catalog_returns", "cr_returning_customer_sk"): ("customer", "c_customer_sk"),
    ("catalog_returns", "cr_returning_cdemo_sk"): (
        "customer_demographics",
        "cd_demo_sk",
    ),
    ("catalog_returns", "cr_returning_hdemo_sk"): (
        "household_demographics",
        "hd_demo_sk",
    ),
    ("catalog_returns", "cr_returning_addr_sk"): (
        "customer_address",
        "ca_address_sk",
    ),
    ("catalog_returns", "cr_call_center_sk"): ("call_center", "cc_call_center_sk"),
    ("catalog_returns", "cr_catalog_page_sk"): ("catalog_page", "cp_catalog_page_sk"),
    ("catalog_returns", "cr_ship_mode_sk"): ("ship_mode", "sm_ship_mode_sk"),
    ("catalog_returns", "cr_warehouse_sk"): ("warehouse", "w_warehouse_sk"),
    ("catalog_returns", "cr_reason_sk"): ("reason", "r_reason_sk"),
    # web_sales
    ("web_sales", "ws_sold_date_sk"): ("date_dim", "d_date_sk"),
    ("web_sales", "ws_sold_time_sk"): ("time_dim", "t_time_sk"),
    ("web_sales", "ws_ship_date_sk"): ("date_dim", "d_date_sk"),
    ("web_sales", "ws_item_sk"): ("item", "i_item_sk"),
    ("web_sales", "ws_bill_customer_sk"): ("customer", "c_customer_sk"),
    ("web_sales", "ws_bill_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("web_sales", "ws_bill_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("web_sales", "ws_bill_addr_sk"): ("customer_address", "ca_address_sk"),
    ("web_sales", "ws_ship_customer_sk"): ("customer", "c_customer_sk"),
    ("web_sales", "ws_ship_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("web_sales", "ws_ship_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("web_sales", "ws_ship_addr_sk"): ("customer_address", "ca_address_sk"),
    ("web_sales", "ws_web_page_sk"): ("web_page", "wp_web_page_sk"),
    ("web_sales", "ws_web_site_sk"): ("web_site", "web_site_sk"),
    ("web_sales", "ws_ship_mode_sk"): ("ship_mode", "sm_ship_mode_sk"),
    ("web_sales", "ws_warehouse_sk"): ("warehouse", "w_warehouse_sk"),
    ("web_sales", "ws_promo_sk"): ("promotion", "p_promo_sk"),
    # web_returns
    ("web_returns", "wr_returned_date_sk"): ("date_dim", "d_date_sk"),
    ("web_returns", "wr_returned_time_sk"): ("time_dim", "t_time_sk"),
    ("web_returns", "wr_item_sk"): ("item", "i_item_sk"),
    ("web_returns", "wr_refunded_customer_sk"): ("customer", "c_customer_sk"),
    ("web_returns", "wr_refunded_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("web_returns", "wr_refunded_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("web_returns", "wr_refunded_addr_sk"): ("customer_address", "ca_address_sk"),
    ("web_returns", "wr_returning_customer_sk"): ("customer", "c_customer_sk"),
    ("web_returns", "wr_returning_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("web_returns", "wr_returning_hdemo_sk"): (
        "household_demographics",
        "hd_demo_sk",
    ),
    ("web_returns", "wr_returning_addr_sk"): ("customer_address", "ca_address_sk"),
    ("web_returns", "wr_web_page_sk"): ("web_page", "wp_web_page_sk"),
    ("web_returns", "wr_reason_sk"): ("reason", "r_reason_sk"),
    # inventory
    ("inventory", "inv_date_sk"): ("date_dim", "d_date_sk"),
    ("inventory", "inv_item_sk"): ("item", "i_item_sk"),
    ("inventory", "inv_warehouse_sk"): ("warehouse", "w_warehouse_sk"),
    # dimension -> dimension
    ("customer", "c_current_addr_sk"): ("customer_address", "ca_address_sk"),
    ("customer", "c_current_cdemo_sk"): ("customer_demographics", "cd_demo_sk"),
    ("customer", "c_current_hdemo_sk"): ("household_demographics", "hd_demo_sk"),
    ("household_demographics", "hd_income_band_sk"): (
        "income_band",
        "ib_income_band_sk",
    ),
}


def db_path(sf: float) -> Path:
    """Return the DuckDB file path for a scale factor.

    Args:
        sf: TPC-DS scale factor.

    Returns:
        Path under ``lab/data``.
    """
    return DATA_DIR / f"tpcds_sf{sf:g}.duckdb"


def ensure_tpcds(sf: float) -> Path:
    """Generate TPC-DS at ``sf`` once and return its DuckDB file.

    The file is written under a ``.tmp`` name and renamed on success, so an
    interrupted generation never leaves a half-built store behind.

    Args:
        sf: TPC-DS scale factor (0.01, 0.1, 1, 10; fractional values work).

    Returns:
        Path to a DuckDB file holding the 24 TPC-DS tables.

    Raises:
        RuntimeError: If the tpcds extension cannot be installed (no network).
    """
    path = db_path(sf)
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        try:
            con.execute("INSTALL tpcds; LOAD tpcds")
        except duckdb.Error as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "DuckDB tpcds extension unavailable; generate data on a networked "
                f"machine and copy {path.name} into {DATA_DIR}"
            ) from exc
        start = time.time()
        con.execute(f"CALL dsdgen(sf={sf})")
        con.execute("CHECKPOINT")
        logger.info("generated TPC-DS sf=%s in %.1fs", sf, time.time() - start)
    finally:
        con.close()
    tmp.rename(path)
    return path


def connect(sf: float, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a connection to the TPC-DS database at ``sf``, generating it if needed.

    Args:
        sf: TPC-DS scale factor.
        read_only: Open read-only (the default, as an agent-facing store would be).

    Returns:
        A DuckDB connection.
    """
    return duckdb.connect(str(ensure_tpcds(sf)), read_only=read_only)


def fk_orphans(
    con: duckdb.DuckDBPyConnection,
    child: tuple[str, str],
    parent: tuple[str, str],
    sample: int | None = None,
) -> int:
    """Count non-null child keys that have no matching parent row.

    Args:
        con: Store connection.
        child: ``(table, column)`` holding the reference.
        parent: ``(table, column)`` holding the referenced key.
        sample: If given, check only this many child rows (reservoir sample,
            fixed seed) instead of the whole table.

    Returns:
        Number of orphaned child values (0 means the FK holds).
    """
    ct, cc = child
    pt, pc = parent
    src = f'"{ct}"'
    if sample is not None:
        src = (
            f'(SELECT "{cc}" FROM "{ct}" '
            f"USING SAMPLE reservoir({int(sample)} ROWS) REPEATABLE (42))"
        )
    sql = (
        f'SELECT count(*) FROM {src} c WHERE c."{cc}" IS NOT NULL '
        f'AND NOT EXISTS (SELECT 1 FROM "{pt}" p WHERE p."{pc}" = c."{cc}")'
    )
    return con.execute(sql).fetchone()[0]


def topk_boundary(
    con: duckdb.DuckDBPyConnection, gold_sql: str, k: int
) -> tuple[tuple, tuple] | None:
    """Return the k-th and (k+1)-th rows of a top-k query without its LIMIT.

    The gold SQL must end in ``LIMIT <k>``; the limit is raised to ``k + 1`` so
    the caller can check that the ranking value strictly separates the two.

    Args:
        con: Store connection.
        gold_sql: Top-k SQL ending in ``LIMIT k``.
        k: The k of the question.

    Returns:
        ``(row_k, row_k_plus_1)``, or None if fewer than ``k + 1`` rows exist.
    """
    head, sep, _ = gold_sql.rpartition("LIMIT")
    if not sep:
        raise ValueError("top-k gold SQL must end in LIMIT k")
    rows = con.execute(f"{head}LIMIT {k + 1}").fetchall()
    if len(rows) <= k:
        return None
    return rows[k - 1], rows[k]


def plan_answer(store: Path, gold_sql: str, plan: list[dict]) -> list[tuple]:
    """Answer a question from only the rows its fine-grained plan fetches.

    Mirrors what the ``fine_grained`` policy does: each plan step copies the
    filtered rows of one table into a scratch database (``in_from`` filters are
    semi-joins on keys already copied), then the gold SQL runs over the copies.
    If this differs from the gold answer, the plan under-fetches.

    Args:
        store: DuckDB file of the store.
        gold_sql: Reference SQL of the question.
        plan: The question's per-table filter plan.

    Returns:
        Rows of ``gold_sql`` evaluated over the plan-fetched rows.
    """
    sym = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
    local = duckdb.connect()
    try:
        local.execute(f"ATTACH '{store}' AS src (READ_ONLY)")
        for table in TABLES:
            local.execute(f'CREATE TABLE "{table}" AS FROM src."{table}" LIMIT 0')
        for step in plan:
            table = step["table"]
            clauses = []
            for col, spec in step["filters"].items():
                items = spec.items() if isinstance(spec, dict) else [("eq", spec)]
                for op, val in items:
                    if op == "in_from":
                        st, sc = val.split(".")
                        clauses.append(f'"{col}" IN (SELECT "{sc}" FROM main."{st}")')
                    else:
                        lit = f"'{val}'" if isinstance(val, str) else str(val)
                        clauses.append(f'"{col}" {sym[op]} {lit}')
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            local.execute(f'INSERT INTO main."{table}" FROM src."{table}"{where}')
        local.execute("USE memory.main")
        return local.execute(gold_sql).fetchall()
    finally:
        local.close()
