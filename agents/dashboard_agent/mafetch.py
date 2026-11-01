#!/usr/bin/env python3
"""Multi-agent failure measurement: a consumption agent (dashboard) fetching from
the DAL agent over A2A, under a fault rate set on the DAL service.

Reports, per consumption-agent retry policy, the inter-agent message count
(a2a_calls) and task success. With retries=0 the consumption agent is fail-fast
(under-communicates: gives up when the DAL agent fails); with retries>0 it
retries the DAL agent (over-communicates: inter-agent amplification).

Run from the dashboard venv, with mcp_dal + dal_agent up (dal_agent started with
COMRADES_FAULT_RATE set to the desired rate):
    dashboard_agent/.venv/bin/python mafetch.py --retries 0,2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from app.dal_client import fetch_from_dal  # noqa: E402

SAMPLE = [
    {"id": "s1", "prompt": "list up to 3 ITSM incidents"},
    {"id": "s2", "prompt": "what are the top cloud costs this month"},
    {"id": "s3", "prompt": "show the application topology"},
]


def load_requests(path: str | None) -> list[dict]:
    if not path:
        return SAMPLE
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def main_async(retries_list, requests, out_csv):
    rate = os.getenv("COMRADES_FAULT_RATE", "0")
    print(f"DAL fault rate (service-side) = {rate}; {len(requests)} requests")
    rows = []
    for retries in retries_list:
        calls, succ = [], []
        for item in requests:
            _, meta = await fetch_from_dal(item["prompt"], retries=retries)
            calls.append(meta["a2a_calls"]); succ.append(1 if meta["success"] else 0)
            rows.append({"id": item.get("id"), "family": item.get("family"),
                         "fault_rate": rate, "retries": retries,
                         "a2a_calls": meta["a2a_calls"], "success": int(meta["success"])})
        n = len(requests)
        print(f"=> retries={retries}: mean_a2a_calls={sum(calls)/n:.2f} "
              f"task_success={sum(succ)/n:.2f}")
    if out_csv and rows:
        import csv
        from pathlib import Path
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "a", newline="") as f:   # append: caller sweeps rates
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retries", default="0,2", help="comma-separated retry policies")
    ap.add_argument("--requests", default=None, help="path to requests.jsonl (default: 3-item sample)")
    ap.add_argument("--out", default=None, help="CSV to append per-run rows to")
    args = ap.parse_args()
    retries_list = [int(r) for r in args.retries.split(",")]
    asyncio.run(main_async(retries_list, load_requests(args.requests), args.out))


if __name__ == "__main__":
    main()
