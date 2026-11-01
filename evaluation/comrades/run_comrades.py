#!/usr/bin/env python3
"""COMRADES driver: sweep mechanism x model x dataset, score, write a tidy CSV.

Run with the dal_agent venv (has a2a-sdk, litellm, mcp, the vendored kg_agent):

    cd evaluation/comrades
    ../../agents/dal_agent/.venv/bin/python run_comrades.py \
        --dataset sample --mechanisms mcp_only,a2a \
        --model provider/meta-llama/llama-3-3-70b-instruct

Prereqs: mcp_dal on :8201 (always); for A2A, dal_agent on :8005.
Provider creds in env (or sourced from agents/dal_agent/.env).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

# Benchmarking MUST bypass the read-through cache, or latency/tokens are garbage
# (a repeated request returns instantly from a prior run). Set before any
# kg_agent import so config picks it up. Override with --cache to study caching.
os.environ.setdefault("KG_AGENT_CACHE_ENABLED", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make runner/metrics importable

from metrics import aggregate  # noqa: E402
from runner import RunResult  # noqa: E402
from runners.mcp_only import MCPOnlyRunner  # noqa: E402
from runners.a2a import A2ARunner  # noqa: E402
from runners.direct import DirectRunner  # noqa: E402

RUNNERS = {"direct": DirectRunner, "mcp_only": MCPOnlyRunner, "a2a": A2ARunner}
# acp to be added (stretch)

# A tiny built-in sample so the harness is runnable before the full dataset
# annotation lands (task #96). Each item: id, prompt, task, backends, hops.
SAMPLE = [
    {"id": "s1", "prompt": "List up to 3 ITSM incidents",
     "task": "query", "backends": ["itsm"], "hops": 1},
    {"id": "s2", "prompt": "What are the top cloud costs this month?",
     "task": "query", "backends": ["finops"], "hops": 1},
    {"id": "s3", "prompt": "Show the application topology from the observability platform",
     "task": "query", "backends": ["apm"], "hops": 1},
]


def load_dataset(name: str) -> list[dict]:
    if name == "sample":
        return SAMPLE
    path = Path(name)
    if not path.exists():
        path = Path(__file__).resolve().parents[1] / name
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("id", f"r{i}")
            d.setdefault("prompt", d.get("utterance") or d.get("query") or "")
            items.append(d)
    return items


async def run_all(dataset: list[dict], mechanisms: list[str], model: str | None) -> list[RunResult]:
    results: list[RunResult] = []
    for mech in mechanisms:
        runner = RUNNERS[mech](model=model)
        try:
            for item in dataset:
                r = await runner.arun(item["id"], item["prompt"])
                # carry dataset labels for later per-task / per-hop slicing
                r.raw.update({"task": item.get("task"), "hops": item.get("hops")})
                print(f"[{mech}] {item['id']}: success={r.success} "
                      f"latency={r.latency_s}s msgs={r.messages} "
                      f"tokens={r.tokens_total} tool={r.tool_invoked}")
                results.append(r)
        finally:
            await runner.aclose()
    return results


def write_csv(results: list[RunResult], out: Path) -> None:
    rows = [r.to_row() for r in results]
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sample")
    ap.add_argument("--mechanisms", default="mcp_only,a2a")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="results/comrades_runs.csv")
    args = ap.parse_args()

    mechanisms = [m.strip() for m in args.mechanisms.split(",") if m.strip()]
    for m in mechanisms:
        if m not in RUNNERS:
            sys.exit(f"unknown mechanism: {m} (have: {', '.join(RUNNERS)})")

    dataset = load_dataset(args.dataset)
    print(f"Running {len(dataset)} requests x {len(mechanisms)} mechanisms "
          f"({', '.join(mechanisms)}) model={args.model or '<env>'}")

    results = asyncio.run(run_all(dataset, mechanisms, args.model))

    out = Path(args.out)
    write_csv(results, out)
    print(f"\nWrote {len(results)} rows -> {out}")

    print("\n=== Aggregate (RQ1 shape) ===")
    for s in aggregate(results):
        print(s)


if __name__ == "__main__":
    main()
