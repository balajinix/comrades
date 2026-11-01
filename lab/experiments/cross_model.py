"""Cross-model summary of the main grid (seed 0), for the paper.

Reads ``findings/grid/*_s0.csv`` for every model and writes ``findings/cross_model.md``:

1. Accuracy per model, interface and dataset (scales pooled).
2. Large-result questions at the larger scale of each dataset: correct of n and
   median tokens, per model and interface.
3. Every other question: correct of n, plain SQL against the by-reference interfaces.
4. Per column (dataset and scale): the best interface's accuracy, the router's gap to
   it, and the per-question oracle (any interface correct).
5. Paired exact McNemar tests, handles against SQL, per model and dataset.
6. The schema-statement fix: gpt-oss router and sql+schema before and after, with
   rate-limited runs left out of the "before" rows.

Usage: .venv/bin/python experiments/cross_model.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "findings" / "grid"
BEFORE = ROOT / "findings" / "grid_v1_schema_probe"
MODELS = {"llama-3-3-70b-instruct": "Llama-3.3-70B", "gpt-oss-120b": "gpt-oss-120b",
          "granite-4-h-small": "Granite-4-H-Small"}
DATASETS = {"itbench": "ITBench", "tpcds": "TPC-DS", "tpch": "TPC-H"}
FACADES = ["generic", "handles", "sql+auto", "router", "sql+schema", "skills", "code",
           "code+prov", "fine_grained", "fine_grained+ops", "graphql+agg"]
BY_REF = ["handles", "sql+auto", "router"]
LARGE_SCALE = {"tpch": [1.0, 10.0], "tpcds": [1.0], "itbench": [35.0]}


def load(pattern: str) -> pd.DataFrame:
    """Grid rows with the model short name, dataset, success flag and total tokens."""
    files = sorted(glob.glob(str(pattern)))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["m"] = d["model"].str.split("/").str[-1]
    d["ds"] = d["schema"].fillna("tpch")
    d["ok"] = d["correct"].astype(str) == "True"
    d["tok"] = d["prompt_tokens"].fillna(0) + d["completion_tokens"].fillna(0)
    d["large"] = d["family"].str.contains("large")
    return d[d["m"].isin(MODELS)]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def md(df: pd.DataFrame) -> str:
    """A DataFrame as a Markdown table."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def rate_limited(traces: Path) -> set[tuple[str, str, str]]:
    """Keys of runs in a trace file that ended on a provider rate limit."""
    out = set()
    if traces.exists():
        for line in traces.read_text().splitlines():
            rec = json.loads(line)
            if any("RateLimitError" in t.get("error", "") for t in rec["transcript"]):
                out.add((f"{float(rec['sf']):g}", rec["qid"], rec["facade"]))
    return out


def main() -> None:
    """Compute every table and write the Markdown summary."""
    d = load(GRID / "*_s0.csv")
    out = ["# Cross-model grid (seed 0, temperature 0)", "",
           f"Runs: {len(d)}; per model: "
           + ", ".join(f"{MODELS[m]} {n}" for m, n in d.groupby("m").size().items()), ""]

    acc = d.groupby(["facade", "m", "ds"])["ok"].mean().unstack(["m", "ds"]).round(2)
    acc = acc.reindex(FACADES)[[(m, ds) for m in MODELS for ds in DATASETS]]
    acc.columns = [f"{MODELS[m]} {DATASETS[ds]}" for m, ds in acc.columns]
    acc.index.name = "interface"
    out += ["## 1. Accuracy (scales pooled)", "", md(acc), ""]

    at_scale = pd.Series(False, index=d.index)
    for ds, sfs in LARGE_SCALE.items():
        at_scale |= (d["ds"] == ds) & d["sf"].isin(sfs)
    big = d[at_scale & d["large"]]
    rows = {}
    for f in FACADES:
        rows[f] = {MODELS[m]: f"{int(g['ok'].sum())}/{len(g)}, {g['tok'].median() / 1e3:.1f}k"
                   for m, g in big[big["facade"] == f].groupby("m")}
    t = pd.DataFrame(rows).T[list(MODELS.values())]
    t.index.name = "interface"
    out += ["## 2. Large-result questions at scale (correct/n, median tokens)", "",
            md(t), ""]

    other = d[~d["large"]]
    rows = {}
    for m, g in other.groupby("m"):
        rows[MODELS[m]] = {f: f"{int(g[g['facade'] == f]['ok'].sum())}/"
                              f"{int((g['facade'] == f).sum())}"
                           for f in ["generic"] + BY_REF}
    t = pd.DataFrame(rows).T
    t.index.name = "model"
    out += ["## 3. Other questions (all scales): correct/n", "", md(t), ""]

    rows = []
    for (m, ds, sf), g in d.groupby(["m", "ds", "sf"]):
        a = g.groupby("facade")["ok"].mean()
        oracle = g.groupby("qid")["ok"].any().mean()
        rows.append({"model": MODELS[m], "column": f"{DATASETS[ds]} {sf:g}",
                     "best": f"{a.idxmax()} {a.max():.2f}",
                     "router": round(a.get("router", float("nan")), 2),
                     "router gap": round(a.max() - a.get("router", float("nan")), 2),
                     "oracle": round(oracle, 2)})
    t = pd.DataFrame(rows).set_index("model")
    out += ["## 4. Router gap to the best interface, and the oracle, per column", "",
            md(t), ""]

    rows = []
    for (m, ds), g in d.groupby(["m", "ds"]):
        p = g.pivot_table(index=["sf", "qid"], columns="facade", values="ok",
                          aggfunc="first")
        b = int((p["handles"] & ~p["generic"]).sum())
        c = int((~p["handles"] & p["generic"]).sum())
        rows.append({"model": MODELS[m], "dataset": DATASETS[ds],
                     "handles only": b, "SQL only": c,
                     "p": f"{mcnemar_exact(b, c):.3g}"})
    t = pd.DataFrame(rows).set_index("model")
    out += ["## 5. handles vs SQL, paired exact McNemar", "", md(t), ""]

    if BEFORE.exists():
        rows = []
        for ds_file in sorted(BEFORE.glob("*gpt-oss-120b_s0.csv")):
            old = pd.read_csv(ds_file)
            limited = rate_limited(ds_file.with_name(ds_file.stem + "_traces.jsonl"))
            key = old["sf"].map(lambda s: f"{float(s):g}") + "|" + old["qid"] + "|" \
                + old["facade"]
            old = old[~key.isin({"|".join(k) for k in limited})]
            new = pd.read_csv(GRID / ds_file.name)
            for f in ["router", "sql+schema"]:
                o = old[old["facade"] == f]
                n = new[new["facade"] == f]
                rows.append({"dataset": ds_file.stem.split("_gpt")[0], "interface": f,
                             "before": f"{(o['correct'].astype(str) == 'True').mean():.2f}"
                                       f" (n={len(o)})",
                             "after": f"{(n['correct'].astype(str) == 'True').mean():.2f}"
                                      f" (n={len(n)})"})
        t = pd.DataFrame(rows).set_index("dataset")
        out += ["## 6. gpt-oss: serving SHOW TABLES / DESCRIBE in on-demand schema SQL",
                "", "Before excludes runs that ended on a provider rate limit.", "",
                md(t), ""]

    (ROOT / "findings" / "cross_model.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
