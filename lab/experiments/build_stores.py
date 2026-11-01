"""Build every DuckDB store the main grid uses, on a fresh machine.

Builds TPC-H at SF 0.01, 1 and 10, TPC-DS at SF 0.01 and 1, TPC-H SF 1 widened to
250-2,000 columns, and ITBench at 1 and 35 scenarios (downloading the pinned
ITBench-Lite revision first if it is missing). Each step is skipped when its store
already exists, so the script is safe to re-run.

Usage: .venv/bin/python experiments/build_stores.py [--sf100]
"""
from __future__ import annotations

import argparse
import logging
import time

from dal_lab import itbench, tpcds, tpch, widen

log = logging.getLogger("build_stores")


def main() -> None:
    """Build the stores in dependency order."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf100", action="store_true", help="also build TPC-H SF 100 (25 GB)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    steps = [(f"tpch sf{sf:g}", lambda sf=sf: tpch.ensure_tpch(sf))
             for sf in (0.01, 1.0, 10.0) + ((100.0,) if args.sf100 else ())]
    steps += [(f"tpcds sf{sf:g}", lambda sf=sf: tpcds.ensure_tpcds(sf)) for sf in (0.01, 1.0)]
    steps += [(f"tpch sf1 w{w}", lambda w=w: widen.ensure_wide(w)) for w in widen.WIDTHS]
    if 1 not in itbench.scenario_dirs():
        steps.append(("itbench download", itbench.download))
    steps += [(f"itbench {s}", lambda s=s: itbench.ensure_itbench(s)) for s in ("s1", "s35")]
    for name, fn in steps:
        t0 = time.monotonic()
        path = fn()
        log.info("%s: %s (%.0f s)", name, path, time.monotonic() - t0)


if __name__ == "__main__":
    main()
