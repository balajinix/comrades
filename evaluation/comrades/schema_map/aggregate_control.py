"""Aggregate the control-discipline sweep into the comparison the paper reports.

Two views, per (mechanism, transport, policy):
  1. Per-rate table — success, gold F1, communication (proxy tool-calls, redundant
     fraction), give-up rate, latency.
  2. Slope decomposition — communication is split into the rate-0 INTERCEPT
     (architectural: a coded workflow is cheaper than ReAct by construction) and
     the SLOPE (Δ from rate 0 = the fault-INDUCED communication, the robustness
     signal). We also report comms-per-recovered-task so a mechanism that gives up
     cheaply does not look "efficient."

A recovery/deviation read comes from the proxy log: for each faulted intent, was
it re-issued (a retry) or abandoned? A coded workflow retries retryable faults by
construction; an agent's loop-detector tends to BLOCK the re-issue, so it abandons
— the under-communication the control axis exposes.

Usage:
    ../../agents/dal_agent/.venv/bin/python schema_map/aggregate_control.py \
        --csv results/control.csv --proxy-log ../harness/proxy_calls.jsonl
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from score import aggregate as gold_aggregate  # noqa: E402


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_rows(csv_path: Path) -> list[dict]:
    return list(csv.DictReader(csv_path.open()))


def proxy_recovery(proxy_log: Path) -> dict[str, dict]:
    """Per run_id: faulted intents, how many were re-issued (retried) vs abandoned.

    An intent = (tool, args_fp). A faulted intent that later reappears in the log
    was retried; one that never reappears was abandoned at first fault."""
    by_run: dict[str, list[dict]] = collections.defaultdict(list)
    if proxy_log.exists():
        for line in proxy_log.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                by_run[r["run_id"]].append(r)
    out = {}
    for run_id, recs in by_run.items():
        counts = collections.Counter((r["tool"], r["args_fp"]) for r in recs)
        faulted_intents = {(r["tool"], r["args_fp"]) for r in recs if r.get("faulted")}
        retried = sum(1 for k in faulted_intents if counts[k] > 1)
        out[run_id] = {"faulted_intents": len(faulted_intents),
                       "retried": retried,
                       "abandoned": len(faulted_intents) - retried}
    return out


def _cell_key(r: dict) -> tuple:
    return (r["mechanism"], r.get("transport", "mcp"), r.get("policy", "shared"), _f(r["fault_rate"]))


def summarize(rows: list[dict], recovery: dict[str, dict]) -> dict[tuple, dict]:
    cells: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        cells[_cell_key(r)].append(r)
    out = {}
    for key, rs in cells.items():
        n = len(rs)
        gold = gold_aggregate([{k: (_f(r[k]) if k in ("precision", "recall", "f1") else int(_f(r[k])))
                                for k in ("precision", "recall", "f1", "tp", "n_pred", "n_gold")} for r in rs])
        succ = sum(int(_f(r["success"])) for r in rs) / n
        calls = statistics.mean(_f(r["proxy_calls"]) for r in rs)
        faulted = statistics.mean(_f(r["proxy_faulted"]) for r in rs)
        rfrac = [_f(r["redundant_frac"]) for r in rs if r["redundant_frac"] not in ("", "None")]
        giveup = sum(1 for r in rs if r["gave_up_at"]) / n
        lat = statistics.mean(_f(r["latency_s"]) for r in rs)
        # recovery roll-up from the proxy log
        rec = [recovery.get(_runid(r), {}) for r in rs]
        fi = sum(x.get("faulted_intents", 0) for x in rec)
        rt = sum(x.get("retried", 0) for x in rec)
        out[key] = {
            "n": n, "success": round(succ, 3),
            "micro_f1": round(gold.get("micro_f1", 0), 3),
            "macro_f1": round(gold.get("macro_f1", 0), 3),
            "mean_calls": round(calls, 2), "mean_faulted": round(faulted, 2),
            "redundant_frac": round(statistics.mean(rfrac), 3) if rfrac else 0.0,
            "giveup_rate": round(giveup, 3), "mean_latency_s": round(lat, 2),
            "faulted_intents": fi, "retried_intents": rt,
            "retry_rate": round(rt / fi, 3) if fi else None,
            "calls_per_recovered": round(calls / succ, 2) if succ > 0 else None,
        }
    return out


def _runid(r: dict) -> str:
    return f"{r['mechanism']}:{r.get('transport','mcp')}:{_f(r['fault_rate'])}:{r['seed']}:{r['id']}"


def print_report(summary: dict[tuple, dict]) -> None:
    arms = sorted({k[:3] for k in summary})
    cols = ["rate", "n", "success", "micro_f1", "calls", "faulted", "redund", "giveup",
            "retry_rate", "calls/recov", "lat_s"]
    for arm in arms:
        mech, transport, policy = arm
        print(f"\n=== {mech} / {transport} / {policy} ===")
        print("  " + " ".join(f"{c:>10}" for c in cols))
        rates = sorted(r for (m, t, p, r) in summary if (m, t, p) == arm)
        base = summary.get((mech, transport, policy, 0.0), {})
        for rate in rates:
            s = summary[(mech, transport, policy, rate)]
            row = [f"{rate:>10}", f"{s['n']:>10}", f"{s['success']:>10.3f}",
                   f"{s['micro_f1']:>10.3f}", f"{s['mean_calls']:>10.2f}",
                   f"{s['mean_faulted']:>10.2f}", f"{s['redundant_frac']:>10.3f}",
                   f"{s['giveup_rate']:>10.3f}",
                   f"{(s['retry_rate'] if s['retry_rate'] is not None else 0):>10.3f}",
                   f"{(s['calls_per_recovered'] if s['calls_per_recovered'] is not None else 0):>10.2f}",
                   f"{s['mean_latency_s']:>10.2f}"]
            print("  " + " ".join(row))
        # slope decomposition vs rate-0 intercept
        if base:
            print(f"  intercept (rate0): calls={base['mean_calls']:.2f} "
                  f"redund={base['redundant_frac']:.3f} success={base['success']:.3f}")
            for rate in rates:
                if rate == 0.0:
                    continue
                s = summary[(mech, transport, policy, rate)]
                print(f"  slope @{rate}: Δcalls={s['mean_calls']-base['mean_calls']:+.2f} "
                      f"Δredund={s['redundant_frac']-base['redundant_frac']:+.3f} "
                      f"Δsuccess={s['success']-base['success']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/control.csv")
    ap.add_argument("--proxy-log", dest="proxy_log",
                    default=str(_HERE.parent / "harness" / "proxy_calls.jsonl"))
    ap.add_argument("--out", default=None, help="optional tidy CSV for plotting")
    args = ap.parse_args()

    csv_path = _HERE / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    rows = load_rows(csv_path)
    recovery = proxy_recovery(Path(args.proxy_log))
    summary = summarize(rows, recovery)
    print_report(summary)

    if args.out:
        out = _HERE / args.out
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["mechanism", "transport", "policy", "fault_rate", "n", "success",
                        "micro_f1", "macro_f1", "mean_calls", "mean_faulted", "redundant_frac",
                        "giveup_rate", "retry_rate", "calls_per_recovered", "mean_latency_s"])
            for (m, t, p, rate), s in sorted(summary.items()):
                w.writerow([m, t, p, rate, s["n"], s["success"], s["micro_f1"], s["macro_f1"],
                            s["mean_calls"], s["mean_faulted"], s["redundant_frac"],
                            s["giveup_rate"], s["retry_rate"], s["calls_per_recovered"],
                            s["mean_latency_s"]])
        print(f"\nwrote tidy summary -> {out}")


if __name__ == "__main__":
    main()
