"""TPC-H data at several scale factors, stored as DuckDB files.

The data-volume axis of the study holds the schema fixed and grows the rows, so
TPC-H at scale factors 0.01 -> 1 (-> 10) is the relational control. Data comes
from DuckDB's ``tpch`` extension (``CALL dbgen(sf=...)``), which is deterministic,
so every run of the lab sees identical rows.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

LAB_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = LAB_ROOT / "data"

TABLES = ("region", "nation", "supplier", "customer", "part", "partsupp",
          "orders", "lineitem")


def db_path(sf: float) -> Path:
    """Return the DuckDB file path for a scale factor.

    Args:
        sf: TPC-H scale factor.

    Returns:
        Path under ``lab/data``.
    """
    return DATA_DIR / f"tpch_sf{sf:g}.duckdb"


def ensure_tpch(sf: float) -> Path:
    """Generate TPC-H at ``sf`` once and return its DuckDB file.

    Args:
        sf: TPC-H scale factor (0.01, 0.1, 1, 10).

    Returns:
        Path to a DuckDB file holding the eight TPC-H tables.

    Raises:
        RuntimeError: If the tpch extension cannot be installed (no network).
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
            con.execute("INSTALL tpch; LOAD tpch")
        except duckdb.Error as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                "DuckDB tpch extension unavailable; generate data on a networked "
                f"machine and copy {path.name} into {DATA_DIR}") from exc
        start = time.time()
        con.execute(f"CALL dbgen(sf={sf})")
        logger.info("generated TPC-H sf=%s in %.1fs", sf, time.time() - start)
    finally:
        con.close()
    tmp.rename(path)
    return path


def connect(sf: float, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open a connection to the TPC-H database at ``sf``, generating it if needed.

    Args:
        sf: TPC-H scale factor.
        read_only: Open read-only (the default, as an agent-facing store would be).

    Returns:
        A DuckDB connection.
    """
    return duckdb.connect(str(ensure_tpch(sf)), read_only=read_only)
