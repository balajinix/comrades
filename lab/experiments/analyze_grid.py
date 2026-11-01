"""Summarise the main grid (E2/E5): metrics, paired tests, oracle and regret.

Reads every ``findings/grid/*.csv`` and writes ``findings/grid_summary.md`` with:

1. Per dataset, scale, model and interface: n, accuracy with a Wilson 95 % CI,
   median and mean prompt tokens, cost per correct answer (all prompt and completion
   tokens spent, including failed runs, divided by correct answers), silent-failure
   rate (answered but wrong), abstention rate (no answer: step, token or context
   limit, or parse failure), mean row precision and recall of answered runs, and
   database time and temp-table memory.
2. Paired exact McNemar tests between selected interfaces on identical questions.
3. Selection: the per-question oracle (the cheapest correct interface), the best
   static interface (highest accuracy, then lowest cost), and every interface's gap to
   both, as accuracy and as cost per correct answer.

Usage: .venv/bin/python experiments/analyze_grid.py [--glob 'findings/grid/*.csv']
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("handles", "generic"), ("sql+auto", "generic"), ("sql+auto", "handles"),
         ("router", "generic"), ("fine_grained+ops", "fine_grained"),
         ("code+prov", "code"), ("sql+schema", "generic"), ("skills", "sql+schema")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load(pattern: str) -> pd.DataFrame:
    """All grid rows, with dataset labels and total tokens."""
    # Side files of dropped rate-limited runs are a record, not results.
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / pattern)))
              if not f.endswith(".rate_limited.csv")]
    if not frames:
        raise SystemExit(f"no files match {pattern}")
    d = pd.concat(frames, ignore_index=True)
    d["correct"] = d["correct"].astype(str) == "True"
    d["answered"] = d["status"] == "answered"
    d["tokens"] = d["prompt_tokens"].fillna(0) + d["completion_tokens"].fillna(0)
    d["model"] = d["model"].str.split("/").str[-1]
    d["dataset"] = d["schema"].fillna("tpch")
    return d


def metrics(d: pd.DataFrame) -> pd.DataFrame:
    """Per-cell metrics."""
    rows = []
    for key, g in d.groupby(["dataset", "sf", "model", "facade"]):
        n, k = len(g), int(g["correct"].sum())
        lo, hi = wilson(k, n)
        ans = g[g["answered"]]
        rows.append(dict(zip(["dataset", "sf", "model", "facade"], key), n=n, correct=k,
                         acc=k / n, acc_lo=lo, acc_hi=hi,
                         ptok_med=g["prompt_tokens"].median(),
                         ptok_mean=g["prompt_tokens"].mean(),
                         cost_per_correct=g["tokens"].sum() / k if k else math.inf,
                         silent_fail=((g["answered"]) & (~g["correct"])).mean(),
                         abstain=(~g["answered"]).mean(),
                         row_p=ans["row_precision"].mean() if "row_precision" in g else math.nan,
                         row_r=ans["row_recall"].mean() if "row_recall" in g else math.nan,
                         db_s=g.get("db_time_s", pd.Series(dtype=float)).mean(),
                         temp_mb=g.get("db_temp_table_peak_bytes",
                                       pd.Series(dtype=float)).mean() / 2 ** 20))
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame) -> pd.DataFrame:
    """Exact McNemar tests for PAIRS on identical (question, seed) runs."""
    rows = []
    for key, g in d.groupby(["dataset", "sf", "model"]):
        wide = g.pivot_table(index=["qid", "seed"], columns="facade", values="correct",
                             aggfunc="first")
        for a, b in PAIRS:
            if a not in wide or b not in wide:
                continue
            both = wide[[a, b]].dropna()
            ab = int(((both[a] == True) & (both[b] == False)).sum())  # noqa: E712
            ba = int(((both[a] == False) & (both[b] == True)).sum())  # noqa: E712
            rows.append(dict(zip(["dataset", "sf", "model"], key), a=a, b=b,
                             n=len(both), a_only=ab, b_only=ba,
                             p=mcnemar_exact(ab, ba)))
    return pd.DataFrame(rows)


def selection(d: pd.DataFrame) -> pd.DataFrame:
    """Oracle and best-static comparison per dataset, scale and model."""
    rows = []
    for key, g in d.groupby(["dataset", "sf", "model"]):
        per_q = g.groupby(["qid", "seed"])
        oracle_ok, oracle_tok = 0, 0.0
        for _, qg in per_q:
            ok = qg[qg["correct"]]
            if len(ok):
                oracle_ok += 1
                oracle_tok += ok["tokens"].min()
            else:
                oracle_tok += qg["tokens"].min()
        nq = per_q.ngroups
        cells = metrics(g)
        best = cells.sort_values(["acc", "cost_per_correct"],
                                 ascending=[False, True]).iloc[0]
        oracle_cpc = oracle_tok / oracle_ok if oracle_ok else math.inf
        for _, c in cells.iterrows():
            rows.append(dict(zip(["dataset", "sf", "model"], key), facade=c["facade"],
                             acc=c["acc"], oracle_acc=oracle_ok / nq,
                             best_static=best["facade"], best_acc=best["acc"],
                             cost_per_correct=c["cost_per_correct"],
                             oracle_cpc=oracle_cpc,
                             cost_vs_oracle=c["cost_per_correct"] / oracle_cpc
                             if oracle_cpc else math.nan))
    return pd.DataFrame(rows)


def main() -> None:
    """Write the summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="findings/grid/*.csv")
    ap.add_argument("--out", default="findings/grid_summary.md")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="runs to include; the paper's per-cell tables use seed 0, and "
                         "experiments/repeats.py covers the repeats")
    args = ap.parse_args()
    d = load(args.glob)
    d = d[d["seed"].isin(args.seeds)]
    m, p, s = metrics(d), paired(d), selection(d)
    fmt = {"acc": "{:.2f}", "acc_lo": "{:.2f}", "acc_hi": "{:.2f}", "ptok_med": "{:.0f}",
           "ptok_mean": "{:.0f}", "cost_per_correct": "{:.0f}", "silent_fail": "{:.2f}",
           "abstain": "{:.2f}", "row_p": "{:.2f}", "row_r": "{:.2f}", "db_s": "{:.2f}",
           "temp_mb": "{:.1f}", "p": "{:.3f}", "oracle_acc": "{:.2f}", "best_acc": "{:.2f}",
           "oracle_cpc": "{:.0f}", "cost_vs_oracle": "{:.2f}"}

    def table(df: pd.DataFrame) -> str:
        out = df.copy()
        for c, f in fmt.items():
            if c in out:
                out[c] = out[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
        return out.to_markdown(index=False)

    text = ["# Main grid summary", "",
            f"{len(d)} runs; generated by `experiments/analyze_grid.py`.", "",
            "## Per-cell metrics", "", table(m), "", "## Paired tests (exact McNemar)", "",
            table(p) if len(p) else "(none yet)", "", "## Selection: oracle and best static",
            "", table(s), ""]
    (ROOT / args.out).write_text("\n".join(text))
    print(f"wrote {args.out}: {len(m)} cells, {len(p)} pairs")


if __name__ == "__main__":
    main()
