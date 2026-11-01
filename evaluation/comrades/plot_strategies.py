#!/usr/bin/env python3
"""Figures and per-cell summaries for the agent-strategy reruns (react, cot, reflect).

Reads results/strategies/ (written by run_agent_strategies.sh) plus the paper's react
baseline for rq1 (results/rq1_recorded_seeds.csv). Writes into results/strategies/
only, so nothing under paper/ changes; copy a PDF into the paper by hand once its
numbers are checked.

    agents/dal_agent/.venv/bin/python evaluation/comrades/plot_strategies.py

Outputs (each skipped while its inputs are missing):
  rq1_strategies.pdf  success vs fault rate (95% CI across seeds) and tokens per task
  rq1_strategies.csv  per (strategy, fault rate): n, seeds, success, CI, messages, tokens
  rq2_strategies.csv  per (strategy, fault rate, retries): n, A2A calls, success, CI
  rq3_strategies.pdf  tokens per successful task by strategy, model and transport (replay)
  rq3_strategies.csv  per (substrate, strategy, model, transport): n, success, CI, latency,
                      tokens, messages, redundant fraction; the paper's live react rows sit
                      beside the replay rows so the two substrates can be compared
Style, palette and fonts come from plots.py.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from plots import OI, _ci95_prop, _style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STRAT = os.path.join(RESULTS, "strategies")
ORDER = ("react", "cot", "reflect")
RQ1_ORDER = ORDER + ("react_retry",)  # rq1 also carries the harness-retry arm
LABEL = {"react": "ReAct", "cot": "Chain of thought", "reflect": "Self-reflection",
         "react_retry": "ReAct, harness retry"}
COLOR = {"react": OI["blue"], "cot": OI["orange"], "reflect": OI["green"],
         "react_retry": OI["purple"]}
MARK = {"react": "o", "cot": "s", "reflect": "^", "react_retry": "D"}
LINE = {"react": "-", "cot": "--", "reflect": ":", "react_retry": "-."}
MODEL = {"meta-llama/llama-3-3-70b-instruct": "Llama-70B",  # Table rq3's short names
         "mistralai/mistral-medium-2505": "Mistral-M",
         "ibm/granite-4-h-small": "Granite-S"}
LIVE_RQ3 = os.path.join(RESULTS, "shards_rq3")  # the paper's live Table rq3 runs


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1"))


def _ci_seeds(values: list[float]) -> float | None:
    """95% CI half-width across seeds (normal approx), as in run_faults.py."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return None
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def _save(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name + ".pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, HERE)}")


def _rq1_frames(react_csv: str) -> dict[str, pd.DataFrame]:
    paths = {"react": react_csv, "cot": os.path.join(STRAT, "rq1_cot.csv"),
             "reflect": os.path.join(STRAT, "rq1_reflect.csv"),
             "react_retry": os.path.join(STRAT, "rq1_react_retry.csv")}
    return {s: pd.read_csv(p) for s, p in paths.items() if os.path.exists(p)}


def rq1(out_dir: str, react_csv: str) -> None:
    frames = _rq1_frames(react_csv)
    if len(frames) < 2:
        print("skip rq1: need react plus at least one other strategy")
        return
    rows = []
    for s in RQ1_ORDER:
        if s not in frames:
            continue
        df = frames[s].copy()
        df["success"] = _truthy(df["data_obtained"])
        for rate, g in df.groupby("fault_rate"):
            per_seed = g.groupby("seed")["success"].mean().tolist()
            rows.append({"strategy": s, "fault_rate": rate, "n": len(g),
                         "seeds": g["seed"].nunique(),
                         "success": round(g["success"].mean(), 3),
                         "ci95_seeds": None if (c := _ci_seeds(per_seed)) is None else round(c, 3),
                         "ci95_runs": round(_ci95_prop(g["success"].mean(), len(g)), 3),
                         "messages": round(g["messages"].mean(), 2),
                         "tokens": round(g["tokens"].mean()),
                         "redundant_frac": round(g["redundant_frac"].mean(), 3),
                         "tool_steps": round(g["tool_steps"].mean(), 2),
                         "attempts": round(g["attempts"].mean(), 2)
                         if "attempts" in g else None})
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "rq1_strategies.csv"), index=False)
    print(f"wrote {os.path.relpath(os.path.join(out_dir, 'rq1_strategies.csv'), HERE)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.4, 3.1),
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
    for s in RQ1_ORDER:
        d = summary[summary.strategy == s].sort_values("fault_rate")
        if d.empty:
            continue
        x, y = d.fault_rate.values, d.success.values
        ci = d.ci95_seeds.fillna(0).values
        ax1.fill_between(x, y - ci, y + ci, color=COLOR[s], alpha=0.15, linewidth=0)
        ax1.plot(x, y, color=COLOR[s], marker=MARK[s], linestyle=LINE[s], label=LABEL[s])
        ax2.plot(x, d.tokens.values / 1000, color=COLOR[s], marker=MARK[s],
                 linestyle=LINE[s], label=LABEL[s])
    ax1.set_ylabel("Task success")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y")
    ax1.legend(loc="lower left", frameon=False)
    ax2.set_ylabel("Tokens per task (k)")
    ax2.set_ylim(0, summary.tokens.max() / 1000 * 1.2)
    ax2.grid(True, axis="y")
    ax2.set_xlabel("Backend fault rate")
    ax2.set_xticks(sorted(summary.fault_rate.unique()))
    _save(fig, out_dir, "rq1_strategies")


def rq2(out_dir: str) -> None:
    rows = []
    for s in ("cot", "reflect"):
        p = os.path.join(STRAT, f"multiagent_{s}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "success" not in df.columns:  # e.g. an in-progress placeholder
            continue
        for (rate, retries), g in df.groupby(["fault_rate", "retries"]):
            p_ok = g["success"].mean()
            rows.append({"strategy": s, "fault_rate": rate, "retries": retries, "n": len(g),
                         "a2a_calls": round(g["a2a_calls"].mean(), 2),
                         "success": round(p_ok, 3),
                         "ci95_runs": round(_ci95_prop(p_ok, len(g)), 3)})
    if not rows:
        print("skip rq2: no strategy results yet")
        return
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "rq2_strategies.csv"), index=False)
    print(f"wrote {os.path.relpath(os.path.join(out_dir, 'rq2_strategies.csv'), HERE)}")


def _rq3_rows(df: pd.DataFrame, substrate: str, strategy: str) -> list[dict]:
    rows = []
    for (model, mech), g in df.groupby(["model", "mechanism"]):
        p_ok = g["success"].mean()
        red = pd.to_numeric(g["redundant_frac"], errors="coerce")
        ok = g[g["success"].astype(int) == 1]
        rows.append({"substrate": substrate, "strategy": strategy, "model": model,
                     "model_short": MODEL.get(model, model), "mechanism": mech,
                     "n": len(g), "seeds": g["seed"].nunique(),
                     "success": round(p_ok, 3),
                     "ci95_runs": round(_ci95_prop(p_ok, len(g)), 3),
                     "latency_s": round(g["latency_s"].mean(), 1),
                     "tokens": round(g["tokens"].mean()),
                     # runners/direct.py logs a failed run as 0 tokens and 0 s while
                     # mcp_only records them, so compare transports on these columns.
                     "latency_s_ok": round(ok["latency_s"].mean(), 1) if len(ok) else None,
                     "tokens_ok": round(ok["tokens"].mean()) if len(ok) else None,
                     "messages": round(g["messages"].mean(), 2),
                     # Table rq3 counts runs without a tool call as 0; both are given.
                     "redundant_frac": round(red.fillna(0).mean(), 3),
                     "redundant_frac_tool_runs": round(red.mean(), 3)
                     if red.notna().any() else None})
    return rows


def rq3(out_dir: str) -> None:
    frames = {s: pd.read_csv(p) for s in ORDER
              if os.path.exists(p := os.path.join(STRAT, f"rq3_{s}.csv"))}
    if not frames:
        print("skip rq3: no strategy results yet")
        return
    rows = []
    live = sorted(Path(LIVE_RQ3).glob("shard_*.csv")) if os.path.isdir(LIVE_RQ3) else []
    if live:
        rows += _rq3_rows(pd.concat(pd.read_csv(f) for f in live), "live", "react")
    for s, df in frames.items():
        rows += _rq3_rows(df, "replay", s)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "rq3_strategies.csv"), index=False)
    print(f"wrote {os.path.relpath(os.path.join(out_dir, 'rq3_strategies.csv'), HERE)}")

    summary = summary[summary.substrate == "replay"]  # the figure shows replay only
    models = [m for m in MODEL if m in set(summary.model)]
    mechs = [m for m in ("direct", "mcp_only") if m in set(summary.mechanism)]
    strategies = [s for s in ORDER if s in frames]
    fig, axes = plt.subplots(1, len(mechs), sharey=True, figsize=(3.4, 2.0), squeeze=False)
    width = 0.8 / len(strategies)
    for ax, mech in zip(axes[0], mechs):
        for i, s in enumerate(strategies):
            d = summary[(summary.strategy == s) & (summary.mechanism == mech)].set_index("model")
            ys = [d.tokens_ok.get(m, float("nan")) / 1000 for m in models]  # successful runs
            ax.bar([j + (i - (len(strategies) - 1) / 2) * width for j in range(len(models))],
                   ys, width, color=COLOR[s], label=LABEL[s])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL[m] for m in models], fontsize=6)
        ax.set_xlabel("in-process (replay)" if mech == "direct" else "MCP (replay)")
        ax.grid(True, axis="y")
    axes[0][0].set_ylabel("Tokens per task (k)")
    axes[0][0].legend(loc="upper left", frameon=False)
    _save(fig, out_dir, "rq3_strategies")


def main() -> None:
    global STRAT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-dir", default=STRAT, help="where the strategy CSVs are")
    ap.add_argument("--out-dir", default=STRAT)
    ap.add_argument("--rq1-react", default=None,
                    help="react baseline for rq1; default the same-day "
                         "strategies/rq1_react.csv if present, else the paper's June file")
    args = ap.parse_args()
    STRAT = args.in_dir
    os.makedirs(args.out_dir, exist_ok=True)
    _style()
    same_day = os.path.join(STRAT, "rq1_react.csv")
    react_csv = args.rq1_react or (same_day if os.path.exists(same_day)
                                   else os.path.join(RESULTS, "rq1_recorded_seeds.csv"))
    print(f"rq1 react baseline: {os.path.relpath(react_csv, HERE)}")
    rq1(args.out_dir, react_csv)
    rq2(args.out_dir)
    rq3(args.out_dir)


if __name__ == "__main__":
    main()
