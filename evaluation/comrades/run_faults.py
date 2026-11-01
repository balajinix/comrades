#!/usr/bin/env python3
"""Failure-driven communication amplification experiment.

For each request we run the agent failure-free (baseline) and again under
injected backend faults, and report how much the failures amplify communication
(messages, tokens, latency) and how much of that is redundant thrash.

Run with the dal_agent venv (mcp_dal on :8201; Provider creds in env):
    cd evaluation/comrades
    ../../agents/dal_agent/.venv/bin/python run_faults.py --rate 0.3 --seed 1
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("KG_AGENT_CACHE_ENABLED", "false")  # never cache during measurement
sys.path.insert(0, str(Path(__file__).resolve().parent))
_DAL_AGENT = Path(__file__).resolve().parents[2] / "agents" / "dal_agent"
if str(_DAL_AGENT) not in sys.path:
    sys.path.insert(0, str(_DAL_AGENT))

from kg_agent import run_agent  # noqa: E402
from metrics import capture_tokens  # noqa: E402
from faults import inject_faults  # noqa: E402
from redundancy import classify_trace  # noqa: E402

SAMPLE = [
    {"id": "s1", "prompt": "List up to 3 ITSM incidents"},
    {"id": "s2", "prompt": "What are the top cloud costs this month?"},
    {"id": "s3", "prompt": "Show the application topology from the observability platform"},
]
# Remote-friendly: set MCP_DAL_URL (e.g. http://internal-host:8201/mcp)
# to run against a remote mcp_dal; falls back to DAL_MCP_URL, then localhost.
MCP = [("mcp_dal", os.getenv("MCP_DAL_URL") or os.getenv("DAL_MCP_URL")
        or "http://localhost:8201/mcp")]


def load_requests(path: str | None) -> list[dict]:
    """Load the request set (jsonl with id/prompt/family); fall back to SAMPLE."""
    if not path:
        return SAMPLE
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _ci95(values: list[float]) -> float | None:
    """95% CI half-width across seeds (normal approx); None if <2 values."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    return round(1.96 * statistics.stdev(vals) / math.sqrt(len(vals)), 3)


def _data_obtained(trace: list[dict]) -> bool:
    """Task-success proxy: at least one tool call returned non-error data, i.e.
    the agent actually retrieved something rather than giving up. A fail-fast
    agent that aborts after a fault obtains no data -> task failure."""
    for step in trace or []:
        if not step.get("action"):
            continue
        tr = step.get("tool_result")
        if isinstance(tr, dict) and str(tr.get("status", "")).lower() != "error":
            return True
    return False


async def one_run(prompt: str, model: str | None, max_steps: int) -> dict:
    with capture_tokens() as tok:
        out = await run_agent(prompt, max_steps=max_steps, mcp_servers=MCP, model=model)
    raw = out.raw_result or {}
    trace = raw.get("trace", [])
    cls = classify_trace(trace)
    return {
        "data_obtained": _data_obtained(trace),
        "latency_s": round(float(out.elapsed_seconds or 0.0), 2),
        "tokens": tok.get("total") or 0,
        "messages": int(raw.get("steps_used") or 0),
        **cls,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


async def main_async(args) -> None:
    rates = [float(r) for r in args.rates.split(",")]
    seeds = [int(s) for s in str(args.seeds).split(",")]
    requests = load_requests(args.requests)
    # COMRADES_REPLAY=0 -> live: non-faulted calls hit the real mcp_dal (set MCP_DAL_URL).
    replay = os.getenv("COMRADES_REPLAY", "1") == "1"
    rows = []
    print(f"RQ1 fault-rate sweep: {len(requests)} requests x rates={rates} x seeds={seeds} "
          f"({'replay' if replay else 'LIVE'} backend; fail-fast agent)\n", flush=True)
    # Incremental checkpoint: write each row as it completes so a long live run
    # survives interruption with partial results intact.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "family", "fault_rate", "seed", "injected", "data_obtained",
            "messages", "tokens", "latency_s", "tool_steps", "errored_steps",
            "necessary", "redundant", "redundant_frac"]
    fcsv = open(out, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=cols, extrasaction="ignore")
    writer.writeheader(); fcsv.flush()
    for rate in rates:
        per_seed_success = []   # task-success per seed -> CI across seeds
        rate_rows = []
        for seed in seeds:
            seed_rows = []
            for idx, item in enumerate(requests):
                # Each request draws an INDEPENDENT, reproducible fault stream
                # derived from (seed, request index). Seeding every request with
                # the bare cell seed (the old behavior) made all requests share
                # one stream, so under fail-fast the whole cell collapsed to a
                # step function of that seed's first draw (draw[0] < rate) instead
                # of averaging over independent Bernoulli(rate) faults -- which
                # made the injected rate diverge from nominal and the per-seed
                # curves disagree wildly. A per-request sub-seed restores the
                # intended per-call fault model while staying deterministic.
                req_seed = seed * 100_003 + idx
                # Non-faulted calls: canned in replay mode; real backend when live.
                with inject_faults(rate=rate, seed=req_seed, replay=replay) as fstats:
                    res = await one_run(item["prompt"], args.model, args.max_steps)
                res.update(id=item["id"], family=item.get("family"),
                           fault_rate=rate, seed=seed,
                           injected=f"{fstats['faulted']}/{fstats['attempted']}")
                seed_rows.append(res); rate_rows.append(res); rows.append(res)
                writer.writerow(res); fcsv.flush()
            per_seed_success.append(sum(r["data_obtained"] for r in seed_rows) / len(seed_rows))

        succ = sum(r["data_obtained"] for r in rate_rows) / len(rate_rows)
        print(f"  rate={rate}: task_success={round(succ,3)} (95%CI +/-{_ci95(per_seed_success)}) "
              f"mean_msgs={_mean([r['messages'] for r in rate_rows])} "
              f"mean_tokens={_mean([float(r['tokens']) for r in rate_rows])} "
              f"mean_redundant={_mean([r['redundant_frac'] for r in rate_rows])}")

    fcsv.close()
    print(f"\nWrote {len(rows)} rows -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.0,0.2,0.4,0.6", help="comma-separated fault rates")
    ap.add_argument("--seeds", default="1", help="comma-separated seeds (>=2 for CIs)")
    ap.add_argument("--requests", default=None, help="path to requests.jsonl (default: 3-item sample)")
    ap.add_argument("--model", default=os.getenv("PROVIDER_MODEL_NAME"))
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", default="results/faults.csv")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
