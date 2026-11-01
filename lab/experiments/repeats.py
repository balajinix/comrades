"""How robust are the reference model's results to repeat runs? For the paper.

Reads the Llama-3.3-70B grid at seed 0 (temperature 0) and seeds 1-2 (temperature
0.7) and writes ``findings/repeats.md`` with:

1. Spread: per cell (interface, dataset, scale), the range of accuracy across the
   three runs; its median and maximum, and the share of cells within 0.10.
2. Accuracy averaged over the three runs, per interface and column (Table 1's layout).
3. The paper's claims, re-tested per run:
   a. large-result questions at scale: correct of 22 per interface;
   b. handles vs SQL on TPC-H SF 1 and 10 (paired exact McNemar);
   c. router gap to the best interface, per column, per run and on the 3-run mean;
   d. code+prov vs code on TPC-DS: silent-failure rate and accuracy at SF 1.

Usage: .venv/bin/python experiments/repeats.py
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "findings" / "grid"
MODEL = "llama-3-3-70b-instruct"
LARGE_SCALE = {"tpch": [1.0, 10.0], "tpcds": [1.0], "itbench": [35.0]}
FACADES = ["generic", "handles", "sql+auto", "router", "sql+schema", "skills", "code",
           "code+prov", "fine_grained", "fine_grained+ops", "graphql+agg"]
COLUMNS = [("itbench", 1.0), ("itbench", 35.0), ("tpcds", 0.01), ("tpcds", 1.0),
           ("tpch", 0.01), ("tpch", 1.0), ("tpch", 10.0)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def md(df: pd.DataFrame) -> str:
    """A DataFrame as a Markdown table."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |"
              for i, r in df.iterrows()]
    return "\n".join(lines)


def main() -> None:
    """Compute every table and write the Markdown summary."""
    files = sorted(glob.glob(str(GRID / f"*{MODEL}_s[012].csv")))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["ds"] = d["schema"].fillna("tpch")
    d["ok"] = d["correct"].astype(str) == "True"
    d["answered"] = d["status"] == "answered"
    runs = sorted(d["seed"].unique())
    out = ["# Repeat runs: Llama-3.3-70B", "",
           "Seed 0 at temperature 0; seeds 1-2 at temperature 0.7. Runs per seed: "
           + ", ".join(f"{s}: {n}" for s, n in d.groupby("seed").size().items()), ""]

    acc = d.groupby(["facade", "ds", "sf", "seed"])["ok"].mean().unstack("seed")
    rng = acc.max(axis=1) - acc.min(axis=1)
    out += ["## 1. Spread across runs, per cell", "",
            f"- cells: {len(rng)}",
            f"- median range: {rng.median():.2f}; max: {rng.max():.2f} "
            f"({' '.join(map(str, rng.idxmax()))})",
            f"- share of cells with range <= 0.10: {(rng <= 0.10).mean():.2f}", ""]

    mean = acc.mean(axis=1).unstack(["ds", "sf"])[COLUMNS].reindex(FACADES).round(2)
    mean.columns = [f"{ds} {sf:g}" for ds, sf in mean.columns]
    mean.index.name = "interface"
    out += ["## 2. Accuracy, mean of 3 runs", "", md(mean), ""]

    at_scale = pd.Series(False, index=d.index)
    for ds, sfs in LARGE_SCALE.items():
        at_scale |= (d["ds"] == ds) & d["sf"].isin(sfs)
    big = d[at_scale & d["family"].str.contains("large")]
    t = big.groupby(["facade", "seed"])["ok"].sum().unstack("seed").reindex(FACADES)
    t.columns = [f"run {s}" for s in t.columns]
    t.index.name = "interface"
    out += ["## 3a. Large-result questions at scale: correct of 22", "",
            md(t.astype("Int64")), ""]

    rows = []
    for sf in (1.0, 10.0):
        for s in runs:
            g = d[(d["ds"] == "tpch") & (d["sf"] == sf) & (d["seed"] == s)]
            p = g.pivot_table(index="qid", columns="facade", values="ok", aggfunc="first")
            b = int((p["handles"] & ~p["generic"]).sum())
            c = int((~p["handles"] & p["generic"]).sum())
            rows.append({"TPC-H SF": f"{sf:g}", "run": s, "handles only": b,
                         "SQL only": c, "p": f"{mcnemar_exact(b, c):.3g}"})
    out += ["## 3b. handles vs SQL on TPC-H, paired exact McNemar", "",
            md(pd.DataFrame(rows).set_index("TPC-H SF")), ""]

    rows = []
    for ds, sf in COLUMNS:
        row = {"column": f"{ds} {sf:g}"}
        for s in runs + ["mean"]:
            a = acc.loc[(slice(None), ds, sf), :]
            a = (a.mean(axis=1) if s == "mean" else a[s]).droplevel([1, 2])
            row[f"gap {s}"] = round(a.max() - a["router"], 2)
            row[f"best {s}"] = a.idxmax()
        rows.append(row)
    t = pd.DataFrame(rows).set_index("column")
    out += ["## 3c. Router gap to the best interface", "", md(t), "",
            f"Max gap per run: "
            + ", ".join(f"{s}: {t[f'gap {s}'].max():.2f}" for s in runs + ["mean"]), ""]

    rows = []
    for s in runs:
        for f in ("code", "code+prov"):
            g = d[(d["ds"] == "tpcds") & (d["seed"] == s) & (d["facade"] == f)]
            sf1 = g[g["sf"] == 1.0]
            rows.append({"run": s, "interface": f,
                         "silent failures": f"{(g['answered'] & ~g['ok']).mean():.2f}",
                         "accuracy SF 1": f"{sf1['ok'].mean():.2f}"})
    out += ["## 3d. code vs code+prov on TPC-DS", "",
            md(pd.DataFrame(rows).set_index("run")), ""]

    (ROOT / "findings" / "repeats.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
