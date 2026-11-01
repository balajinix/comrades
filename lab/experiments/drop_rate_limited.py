"""Remove grid rows whose run ended on a provider rate limit, so a resume redoes them.

A 429 from the provider is an infrastructure failure, not model behaviour, but before
the client retried rate limits such runs were scored as ``llm_error``. This drops
each such row from the CSV and its trace from the ``_traces.jsonl`` file next to it;
re-running the same grid command then repeats exactly those runs. The removed rows
are appended to ``<stem>.rate_limited.csv`` for the record.

Usage: .venv/bin/python experiments/drop_rate_limited.py findings/grid/*.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _key(rec: dict) -> tuple[str, str, str]:
    return (f"{float(rec['sf']):g}", str(rec["qid"]), str(rec["facade"]))


def drop(csv_path: Path) -> int:
    """Drop rate-limited runs from one grid CSV and its traces; return how many."""
    traces = csv_path.with_name(csv_path.stem + "_traces.jsonl")
    if not traces.exists():
        return 0
    lines = traces.read_text().splitlines()
    limited = set()
    for line in lines:
        rec = json.loads(line)
        if rec["status"] == "llm_error" and any(
                "RateLimitError" in t.get("error", "") for t in rec["transcript"]):
            limited.add(_key(rec))
    if not limited:
        return 0
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0])
    keep = [r for r in rows if not (r["status"] == "llm_error" and _key(r) in limited)]
    gone = [r for r in rows if r not in keep]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    side = csv_path.with_name(csv_path.stem + ".rate_limited.csv")
    new = not side.exists()
    with open(side, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(gone)
    traces.write_text("".join(
        line + "\n" for line in lines
        if not (json.loads(line)["status"] == "llm_error"
                and _key(json.loads(line)) in limited)))
    return len(gone)


def main() -> None:
    """Drop rate-limited rows from every CSV given."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.csvs:
        if path.name.endswith(".rate_limited.csv"):
            continue
        n = drop(path)
        if n:
            print(f"{path}: dropped {n} rate-limited runs")


if __name__ == "__main__":
    main()
