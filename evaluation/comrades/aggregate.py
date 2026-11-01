#!/usr/bin/env python3
"""Aggregate COMRADES result CSVs into LaTeX table rows for the paper.

- RQ1 (run_faults.py -> results/faults.csv): per fault rate, task success (with a
  95% normal-approx CI on the success proportion over all runs at that rate), mean
  messages, mean tokens, mean redundant fraction.

Usage:
    python aggregate.py --rq1 results/faults.csv --rq3 results/rq3.csv
Prints LaTeX \\midrule-separated rows ready to paste into tab:rq1 / tab:rq3.
No third-party deps (stdlib csv/statistics/math).
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _ci95(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def _fmt_tokens(t):
    if t is None:
        return "--"
    return f"{t/1000:.1f}k" if t >= 1000 else f"{t:.0f}"


def rq1(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    # group by rate; within a rate, per-seed success rate -> CI across seeds
    by_rate = defaultdict(list)
    for r in rows:
        by_rate[_f(r["fault_rate"])].append(r)

    print("% RQ1 (tab:rq1): fail-fast agent vs fault rate")
    print("% n = runs per rate (seeds x requests). CI = 95% normal-approx on the success proportion.")
    print("% Fault rate & Task success & Msgs & Tokens & Redundant frac. \\\\")
    for rate in sorted(by_rate):
        rr = by_rate[rate]
        outcomes = [1 if str(r["data_obtained"]).lower() == "true" else 0 for r in rr]
        n = len(outcomes)
        p = _mean(outcomes)
        # 95% normal-approximation CI on a proportion over ALL runs at this rate
        ci = 1.96 * math.sqrt(p * (1 - p) / n) if n else None
        ci_s = f"$\\pm${ci:.2f}" if ci is not None else ""
        msgs = _mean([_f(r["messages"]) for r in rr])
        toks = _mean([_f(r["tokens"]) for r in rr])
        red = _mean([_f(r["redundant_frac"]) for r in rr])
        print(f"{rate:.1f} & {p:.2f}{ci_s} & {msgs:.1f} & {_fmt_tokens(toks)} & {red:.2f} \\\\  % n={n}")
    print()


def _short_model(m: str) -> str:
    """Strip provider prefix for display: meta-llama/llama-3-3-70b-instruct -> llama-3-3-70b."""
    base = (m or "").split("/")[-1]
    return base.replace("-instruct", "").replace("provider/", "")


def rq3(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    by = defaultdict(list)
    for r in rows:
        by[(r["mechanism"], r.get("model", ""), _f(r["fault_rate"]))].append(r)

    print("% RQ3 (tab:rq3): architecture x model x fault rate")
    print("% Arch & Model & Fault rate & Success & Msgs & Tokens & Redundant frac. \\\\")
    for (mech, model, rate) in sorted(by):
        rr = by[(mech, model, rate)]
        succ = _mean([1 if str(r["success"]).lower() in ("1", "true") else 0 for r in rr])
        msgs = _mean([_f(r["messages"]) for r in rr])
        toks = _mean([_f(r["tokens"]) for r in rr])
        red = _mean([_f(r["redundant_frac"]) for r in rr if r.get("redundant_frac") not in ("", None)])
        red_s = f"{red:.2f}" if red is not None else "--"
        print(f"{mech} & {_short_model(model)} & {rate:.1f} & {succ:.2f} & "
              f"{msgs:.1f} & {_fmt_tokens(toks)} & {red_s} \\\\")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1", default=None, help="results/faults.csv")
    ap.add_argument("--rq3", default=None, help="results/rq3.csv")
    args = ap.parse_args()
    if not args.rq1 and not args.rq3:
        ap.error("give --rq1 and/or --rq3")
    if args.rq1:
        rq1(args.rq1)
    if args.rq3:
        rq3(args.rq3)


if __name__ == "__main__":
    main()
