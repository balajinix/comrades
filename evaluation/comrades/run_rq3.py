#!/usr/bin/env python3
"""RQ3: communication architecture x model sweep under fault injection.

Sweeps mechanism x model x fault rate x seed over the request set, recording
communication volume (messages, tokens, latency), task success, recovery, and the
redundant fraction. In-process mechanisms (direct, mcp_only) inject faults via
faults.inject_faults; the a2a mechanism restarts the dal_agent server with the
fault rate/seed/model per cell so the A2A path sees the same faults.

Run from the dal_agent venv. Remote mcp_dal: export MCP_DAL_URL.
    cd evaluation/comrades
    ../../agents/dal_agent/.venv/bin/python run_rq3.py \
      --mechanisms mcp_only \
      --models meta-llama/llama-3-3-70b-instruct,mistralai/mistral-medium-2505,ibm/granite-4-h-small \
      --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 --requests requests.jsonl --out results/rq3.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KG_AGENT_CACHE_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
_DAL = Path(__file__).resolve().parents[2] / "agents" / "dal_agent"
if str(_DAL) not in sys.path:
    sys.path.insert(0, str(_DAL))

from faults import inject_faults          # noqa: E402
from redundancy import classify_trace     # noqa: E402
from runners.mcp_only import MCPOnlyRunner  # noqa: E402
from runners.direct import DirectRunner   # noqa: E402
from runners.a2a import A2ARunner         # noqa: E402

RUNNERS = {"direct": DirectRunner, "mcp_only": MCPOnlyRunner, "a2a": A2ARunner}
INPROC = {"direct", "mcp_only"}
SAMPLE = [
    {"id": "s1", "prompt": "List up to 3 ITSM incidents"},
    {"id": "s2", "prompt": "What are the top cloud costs this month?"},
    {"id": "s3", "prompt": "Show the application topology from the observability platform"},
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


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _redundant_frac(res):
    tr = (res.raw or {}).get("trace")
    if not tr:
        return None
    return classify_trace(tr).get("redundant_frac")


def _restart_dal(rate: float, seed: int, model: str | None, mcp_url: str) -> bool:
    """Restart the dal_agent A2A server with the given fault rate/seed/model."""
    subprocess.run("kill -9 $(lsof -nP -tiTCP:8005 2>/dev/null) 2>/dev/null; sleep 1",
                   shell=True)
    env = dict(os.environ, KG_AGENT_CACHE_ENABLED="false",
               COMRADES_FAULT_RATE=str(rate), COMRADES_FAULT_SEED=str(seed),
               COMRADES_REPLAY=os.environ.get("COMRADES_REPLAY", "1"),
               DAL_MCP_URL=mcp_url, DAL_AGENT_MODEL=(model or ""))
    log = open("/tmp/dal_agent_rq3.log", "w")
    subprocess.Popen([str(_DAL / ".venv/bin/python"), "-m", "app"],
                     cwd=str(_DAL), env=env, stdout=log, stderr=log)
    for _ in range(40):
        if subprocess.run("lsof -nP -iTCP:8005 -sTCP:LISTEN >/dev/null 2>&1",
                          shell=True).returncode == 0:
            return True
        time.sleep(1)
    return False


def _row(r, mech, model, rate, seed, item, fstats) -> dict:
    return {
        "id": item["id"], "family": item.get("family"), "mechanism": mech,
        "model": model, "fault_rate": rate, "seed": seed,
        "success": int(bool(r.success)), "messages": r.messages,
        "tokens": r.tokens_total or 0, "latency_s": round(r.latency_s, 2),
        "recovered": int(bool(r.recovered)), "redundant_frac": _redundant_frac(r),
        "injected": (f"{fstats['faulted']}/{fstats['attempted']}" if fstats else ""),
    }


async def run_cell(mech, model, rate, seed, requests, max_steps, mcp_url) -> list[dict]:
    rows = []
    if mech in INPROC:
        runner = RUNNERS[mech](model=model, max_steps=max_steps)
        try:
            for idx, item in enumerate(requests):
                # Per-request sub-seed: seeding every request with the bare cell
                # seed made all requests share one fault stream, so under fail-fast
                # each cell collapsed to a step function of the seed's first draw
                # instead of averaging over independent Bernoulli(rate) faults.
                # (Same fix as run_faults.py; the a2a path is unaffected because its
                # server-side injector seeds once per server/cell restart.)
                req_seed = seed * 100_003 + idx
                with inject_faults(rate=rate, seed=req_seed,
                                   replay=os.getenv("COMRADES_REPLAY", "1") == "1") as fstats:
                    r = await runner.arun(item["id"], item["prompt"])
                rows.append(_row(r, mech, model, rate, seed, item, fstats))
        finally:
            await runner.aclose()
    else:  # a2a: faults live in the restarted server
        if not _restart_dal(rate, seed, model, mcp_url):
            print(f"  [a2a] dal_agent failed to start (rate={rate}, seed={seed})")
            return rows
        runner = RUNNERS[mech](model=model)
        try:
            for item in requests:
                r = await runner.arun(item["id"], item["prompt"])
                rows.append(_row(r, mech, model, rate, seed, item, None))
        finally:
            await runner.aclose()
    return rows


async def main_async(args) -> None:
    requests = load_requests(args.requests)
    mechs = [m.strip() for m in args.mechanisms.split(",") if m.strip()]
    models = [m.strip() for m in args.models.split(",")] if args.models else [None]
    rates = [float(x) for x in args.rates.split(",")]
    seeds = [int(x) for x in str(args.seeds).split(",")]
    mcp_url = (os.getenv("MCP_DAL_URL") or os.getenv("DAL_MCP_URL")
               or "http://localhost:8201/mcp")
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "family", "mechanism", "model", "fault_rate", "seed", "success",
            "messages", "tokens", "latency_s", "recovered", "redundant_frac", "injected"]
    # Checkpoint incrementally: write the header up front, then flush each cell as it
    # completes so a long sweep survives an interruption with partial results intact.
    n_written = 0
    print(f"RQ3: mechs={mechs} models={[m or '<env>' for m in models]} rates={rates} "
          f"seeds={seeds} x {len(requests)} requests  (mcp_dal={mcp_url})\n")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        f.flush()
        for model in models:
            for mech in mechs:
                for rate in rates:
                    cell = []
                    for seed in seeds:
                        cell += await run_cell(mech, model, rate, seed, requests,
                                               args.max_steps, mcp_url)
                    w.writerows(cell)
                    f.flush()
                    n_written += len(cell)
                    print(f"  {mech} | {model or '<env>'} | rate={rate}: "
                          f"success={_mean([r['success'] for r in cell])} "
                          f"msgs={_mean([r['messages'] for r in cell])} "
                          f"tokens={_mean([float(r['tokens']) for r in cell])} "
                          f"redundant={_mean([r['redundant_frac'] for r in cell])} "
                          f"[{n_written} rows]")
    print(f"\nWrote {n_written} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanisms", default="mcp_only",
                    help="comma list of: direct, mcp_only, a2a")
    ap.add_argument("--models", default=os.getenv("PROVIDER_MODEL_NAME"),
                    help="comma-separated model ids (default: env)")
    ap.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--requests", default=None)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--out", default="results/rq3.csv")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
