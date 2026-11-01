#!/usr/bin/env python3
"""Process-sharded driver for the RQ1 fault sweep.

``faults.inject_faults`` monkeypatches a *process-global* (``react.call_tool``)
for the duration of each run, so async concurrency within one process is unsafe
(peers clobber each other's patch). Instead we run each ``(rate, seed)`` cell in
its own OS process via ``run_faults.py`` -- each process keeps its own global
monkeypatch, so per-cell results are byte-for-byte identical to the sequential
run, just many cells at once. A semaphore caps concurrent provider load.

Run from ``evaluation/comrades`` with the same env as ``run_faults.py``
(PROVIDER_* creds, MCP_DAL_URL, COMRADES_REPLAY). Example::

    COMRADES_REPLAY=1 MCP_DAL_URL=... \
      python run_faults_parallel.py --rates 0.0,0.2,0.4,0.6 --seeds 1,2,3 \
        --requests requests.jsonl --out results/rq1_recorded_seeds.csv \
        --concurrency 6 --skip 0.0:1,0.0:2
"""
import argparse
import asyncio
import csv
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RETRY = "none"  # set from --retry; passed to every cell


async def run_cell(rate: float, seed: int, requests: str, shard_dir: Path,
                   sem: asyncio.Semaphore, results: list) -> None:
    out = shard_dir / f"shard_r{rate}_s{seed}.csv"
    async with sem:
        t0 = time.monotonic()
        print(f"[start ] rate={rate} seed={seed}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(HERE / "run_faults.py"),
            "--rates", str(rate), "--seeds", str(seed),
            "--requests", requests, "--out", str(out),
            *(["--retry", RETRY] if RETRY != "none" else []),
            cwd=str(HERE), env=os.environ.copy(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        dt = time.monotonic() - t0
        n = 0
        if out.exists():
            with open(out) as f:
                n = max(sum(1 for _ in f) - 1, 0)
        ok = proc.returncode == 0
        results.append((rate, seed, out, ok, n))
        tail = (err.decode(errors="replace").strip().splitlines()[-1:] or [""])[0] if err else ""
        print(f"[{'done ' if ok else 'FAIL '}] rate={rate} seed={seed} "
              f"rows={n} {dt/60:.1f}min {('' if ok else '  ERR: ' + tail[:200])}",
              flush=True)


def merge(shards: list, salvage: Path | None, out: Path) -> int:
    """Concatenate shard CSVs (+ optional salvaged rows) under one header."""
    header = None
    rows = []
    sources = [s[2] for s in shards if s[2].exists()]
    if salvage and salvage.exists():
        sources.insert(0, salvage)
    for src in sources:
        with open(src, newline="") as f:
            r = csv.reader(f)
            h = next(r, None)
            if h is None:
                continue
            header = header or h
            rows.extend(row for row in r if row)
    if header is None:
        return 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--requests", default="requests.jsonl")
    ap.add_argument("--out", default="results/rq1_recorded_seeds.csv")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--retry", choices=("none", "shared"), default="none",
                    help="harness retry policy for every cell (default fail-fast)")
    ap.add_argument("--shard-dir", default="results/shards",
                    help="per-cell shard folder; use a new one so existing shards stay intact")
    ap.add_argument("--skip", default="",
                    help="comma list of rate:seed cells to skip, e.g. 0.0:1,0.0:2")
    ap.add_argument("--salvage", default="",
                    help="CSV of already-completed rows to fold into the merge "
                         "(must only contain the skipped cells)")
    args = ap.parse_args()

    rates = [float(r) for r in args.rates.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    skip = {tuple(c.split(":")) for c in args.skip.split(",") if c}
    global RETRY
    RETRY = args.retry
    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    cells = [(r, s) for r in rates for s in seeds
             if (str(r), str(s)) not in skip]
    print(f"sharded RQ1: {len(cells)} cells x {args.requests} "
          f"(concurrency={args.concurrency}, replay={os.getenv('COMRADES_REPLAY')})\n"
          f"skipping {sorted(skip)}\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list = []
    await asyncio.gather(*[
        run_cell(r, s, args.requests, shard_dir, sem, results) for r, s in cells
    ])

    salvage = Path(args.salvage) if args.salvage else None
    total = merge(results, salvage, Path(args.out))
    failed = [(r, s) for r, s, _, ok, _ in results if not ok]
    print(f"\nmerged {total} rows -> {args.out}", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} cells failed: {failed}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
