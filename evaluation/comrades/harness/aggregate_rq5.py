"""Aggregate RQ5 harness runs into per-(harness, fault rate) LaTeX rows.

Reports COMMUNICATION as a function of injected fault rate: mean tool calls to the
data layer (the amplification signal), mean turns and output tokens, and the
recover-vs-give-up split (how the harness behaves when it hits a fault). No safety /
enforcement metric -- that is the a related project paper's variable.

Usage:  python harness/aggregate_rq5.py harness/results/rq5.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else 0.0


def _ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n) if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="harness/results/rq5.csv")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    by = defaultdict(list)
    for r in rows:
        by[(r["harness"], _f(r["fault_rate"]))].append(r)

    print("% RQ5 (tab:rq5): coding-harness communication vs backend fault rate")
    print("% Harness & Fault rate & Tool calls & Turns & Out tokens & Recover & Give-up \\\\")
    for (h, rate) in sorted(by):
        rr = by[(h, rate)]
        n = len(rr)
        calls = _mean([_f(r["tool_calls"]) for r in rr])
        turns = _mean([_f(r["num_turns"]) for r in rr])
        tok = _mean([_f(r["tokens_out"]) for r in rr])
        # recover / give-up are defined only on runs that actually hit a fault
        faulted_runs = [r for r in rr if _f(r["faulted"]) > 0]
        nf = len(faulted_runs)
        rec = sum(int(_f(r["recovered"])) for r in faulted_runs)
        rec_p = rec / nf if nf else 0.0
        giveup_p = sum(int(_f(r["gave_up"])) for r in faulted_runs) / nf if nf else 0.0
        rec_s = f"{rec_p:.2f}$\\pm${_ci95(rec_p, nf):.2f}" if nf else "--"
        giveup_s = f"{giveup_p:.2f}" if nf else "--"
        print(f"{h} & {rate:.1f} & {calls:.1f} & {turns:.1f} & {tok:.0f} & "
              f"{rec_s} & {giveup_s} \\\\  % n={n}, faulted_runs={nf}")
    print()


if __name__ == "__main__":
    main()
