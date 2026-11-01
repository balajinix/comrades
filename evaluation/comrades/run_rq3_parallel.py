#!/usr/bin/env python3
"""Process-sharded driver for the RQ3 in-process sweep (direct, mcp_only).

Same rationale as ``run_faults_parallel.py``: ``faults.inject_faults`` monkeypatches
a process-global, so each (mechanism, model, rate, seed) cell runs in its own
``run_rq3.py`` process and merges at the end. ONLY the in-process mechanisms are
sharded here -- the ``a2a`` mechanism restarts a fixed-port server and must not run
concurrently (and its server-side seeding was never affected by the reseed bug).

Run from ``evaluation/comrades`` with the RQ3 env (PROVIDER_* creds, MCP_DAL_URL,
COMRADES_REPLAY=1).
"""
import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


async def run_cell(mech, model, rate, seed, requests, shard_dir, sem, results):
    safe_model = model.replace("/", "-")
    out = shard_dir / f"shard_{mech}_{safe_model}_r{rate}_s{seed}.csv"
    async with sem:
        t0 = time.monotonic()
        print(f"[start ] {mech} {model} rate={rate} seed={seed}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(HERE / "run_rq3.py"),
            "--mechanisms", mech, "--models", model,
            "--rates", str(rate), "--seeds", str(seed),
            "--requests", requests, "--out", str(out),
            cwd=str(HERE), env=os.environ.copy(),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        dt = time.monotonic() - t0
        n = (sum(1 for _ in open(out)) - 1) if out.exists() else 0
        ok = proc.returncode == 0
        results.append((mech, model, rate, seed, out, ok, n))
        tail = (err.decode(errors="replace").strip().splitlines()[-1:] or [""])[0] if err else ""
        print(f"[{'done ' if ok else 'FAIL '}] {mech} {model} rate={rate} seed={seed} "
              f"rows={n} {dt/60:.1f}min{'' if ok else '  ERR: ' + tail[:200]}", flush=True)


def merge(results, out):
    header, rows = None, []
    for *_ , src, _ok, _n in [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in results]:
        if not src.exists():
            continue
        with open(src, newline="") as f:
            rd = csv.reader(f)
            h = next(rd, None)
            if h is None:
                continue
            header = header or h
            rows.extend(row for row in rd if row)
    if header is None:
        return 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanisms", default="direct,mcp_only")
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--requests", default="requests.jsonl")
    ap.add_argument("--out", default="results/rq3_inproc_fixed.csv")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    mechs = [m.strip() for m in args.mechanisms.split(",") if m.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rates = [float(r) for r in args.rates.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    shard_dir = Path("results/shards_rq3")
    shard_dir.mkdir(parents=True, exist_ok=True)

    cells = [(m, mdl, r, s) for m in mechs for mdl in models for r in rates for s in seeds]
    print(f"sharded RQ3 in-proc: {len(cells)} cells x {args.requests} "
          f"(concurrency={args.concurrency}, replay={os.getenv('COMRADES_REPLAY')})\n",
          flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    await asyncio.gather(*[
        run_cell(m, mdl, r, s, args.requests, shard_dir, sem, results)
        for m, mdl, r, s in cells
    ])
    total = merge(results, Path(args.out))
    failed = [(m, mdl, r, s) for m, mdl, r, s, _, ok, _ in results if not ok]
    print(f"\nmerged {total} rows -> {args.out}", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} cells failed: {failed}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
