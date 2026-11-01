"""Question-level tests for the paper's paired comparisons. For the paper.

Pooled cells repeat a question at two or three scales, and the retry experiment
repeats each question over seven fault seeds, so runs are not independent. Here every
comparison is made per question: a question's score under an interface is its mean
correctness over the scales (or seeds) in the cell, and an exact two-sided sign test
counts the questions on which one interface scored higher. Holm's correction is
applied twice: over the eight headline tests (F1 at both TPC-H scales and on the large-result questions, handles vs SQL
per model over all 100 questions, F6 against both other arms), and over every test
in the table. Also writes the counts the paper cites: runs per
model, questions served by a coded workflow, and latency on the large-result pairs
and over all runs.

Usage: .venv/bin/python experiments/question_level.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "findings" / "grid"
MODELS = {"llama-3-3-70b-instruct": "Llama-3.3-70B", "gpt-oss-120b": "gpt-oss-120b",
          "granite-4-h-small": "Granite-4-H-Small"}
DATASETS = {"itbench": "ITBench", "tpcds": "TPC-DS", "tpch": "TPC-H"}
LARGE_SCALE = {"tpch": [1.0, 10.0], "tpcds": [1.0], "itbench": [35.0]}
TOP = ["generic", "handles", "sql+auto", "router"]


def sign_test(b: int, c: int) -> float:
    """Two-sided exact sign test (binomial, p = 1/2) on b wins against c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=ps.__getitem__)
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def load() -> pd.DataFrame:
    """Seed-0 grid rows of every model, without rate-limit side files."""
    files = [f for f in glob.glob(str(GRID / "*_s0.csv")) if ".rate_limited" not in f]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["ds"] = d["schema"].fillna("tpch")
    d["ok"] = d["correct"].astype(str) == "True"
    d["m"] = d["model"].map(lambda s: next(v for k, v in MODELS.items() if k in s))
    return d


def compare(g: pd.DataFrame, a: str, b: str) -> tuple[int, int]:
    """Questions on which interface ``a`` scored higher than ``b``, and the reverse."""
    s = g[g["facade"].isin([a, b])].groupby(["qid", "facade"])["ok"].mean().unstack()
    return int((s[a] > s[b]).sum()), int((s[a] < s[b]).sum())


def md(rows: list[dict]) -> str:
    """Rows of dicts as a Markdown table."""
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])


def main() -> None:
    """Run every test and write ``findings/question_level.md``."""
    d = load()
    tests = []  # (finding, comparison, a wins, b wins)
    llama = d[d["m"] == "Llama-3.3-70B"]
    for sf in (1.0, 10.0):
        g = llama[(llama["ds"] == "tpch") & (llama["sf"] == sf)]
        tests.append(("F1", f"Llama TPC-H SF {sf:g}: handles vs SQL",
                      *compare(g, "handles", "generic")))
    at_scale = pd.Series(False, index=llama.index)
    for ds, sfs in LARGE_SCALE.items():
        at_scale |= (llama["ds"] == ds) & llama["sf"].isin(sfs)
    big = llama[at_scale & llama["family"].str.contains("large")]
    tests.append(("F1", "Llama large-result questions at scale: handles vs SQL",
                  *compare(big.assign(qid=big["ds"] + ":" + big["qid"]), "handles", "generic")))
    for m in MODELS.values():
        g = d[d["m"] == m].assign(qid=lambda x: x["ds"] + ":" + x["qid"])
        tests.append(("F3", f"{m} all 100 questions: handles vs SQL",
                      *compare(g, "handles", "generic")))
        for ds, name in DATASETS.items():
            g = d[(d["m"] == m) & (d["ds"] == ds)]
            tests.append(("F3", f"{m} {name}: handles vs SQL", *compare(g, "handles", "generic")))
    for m in MODELS.values():
        for ds, name in DATASETS.items():
            g = d[(d["m"] == m) & (d["ds"] == ds)]
            acc = g[g["facade"].isin(TOP)].groupby("facade")["ok"].mean()
            best = acc.idxmax()
            for f in ("sql+auto", "router"):
                if f != best:
                    tests.append(("F2", f"{m} {name}: {f} vs best ({best})",
                                  *compare(g, f, best)))
    retry = pd.read_csv(ROOT / "findings" / "llm_faults_v2.csv")
    bursty = retry[retry["process"] != "bernoulli"].assign(ok=lambda x: x["outcome"] == "correct")
    for other in ("native", "prose"):
        tests.append(("F6", f"bursty faults: coded vs {other} (per question, 7 seeds)",
                      *compare(bursty.rename(columns={"arm": "facade"}), "coded", other)))

    ps = [sign_test(b, c) for *_, b, c in tests]
    adj = holm(ps)
    head = [i for i, (f, c, *_) in enumerate(tests)
            if f in ("F1", "F6") or "all 100" in c]
    head_adj = dict(zip(head, holm([ps[i] for i in head])))
    rows = [{"finding": f, "comparison": c, "first higher": b, "second higher": w,
             "p": f"{p:.3g}",
             "Holm p (headline)": f"{head_adj[i]:.3g}" if i in head_adj else "",
             "Holm p (all)": f"{a:.3g}"}
            for i, ((f, c, b, w), p, a) in enumerate(zip(tests, ps, adj))]
    out = ["# Question-level tests", "",
           "Per question: mean correctness over the scales (grid) or fault seeds (F6) in "
           f"the cell; exact two-sided sign test; Holm over the {len(head)} headline "
           f"tests and over all {len(tests)}.", "", md(rows), ""]

    runs = d.groupby("m").size()
    expected = d.groupby("m").apply(lambda g: g.groupby(["ds", "sf", "qid"]).ngroups
                                    * g["facade"].nunique(), include_groups=False)
    out += ["## Runs per model (seed 0)", "",
            md([{"model": m, "runs": int(runs[m]), "question-scale-interface cells":
                 int(expected[m])} for m in runs.index]), ""]

    wf = []
    for ds, path in (("tpch", "tpch_questions.jsonl"), ("tpcds", "tpcds_questions.jsonl"),
                     ("itbench", "itbench_questions.jsonl")):
        qs = [json.loads(line) for line in (ROOT / "workloads" / path).open()]
        wf.append({"dataset": DATASETS[ds], "questions": len(qs),
                   "with a coded workflow": sum(bool(q.get("workflow")) for q in qs)})
    out += ["## Questions a router workflow serves", "", md(wf), ""]

    lat = big.groupby("facade")[["wall_s", "llm_latency_s", "db_time_s"]].median()
    lat = lat.reindex([f for f in TOP + ["code+prov", "skills"] if f in lat.index]).round(2)
    out += ["## Llama, 22 large-result pairs: median seconds per run", "",
            md([{"interface": f, **r.to_dict()} for f, r in lat.iterrows()]), ""]
    med = llama.groupby(["facade", "ds"])["wall_s"].median().unstack()
    med["all"] = llama.groupby("facade")["wall_s"].median()
    out += ["## Llama, all runs: median wall-clock seconds per run", "",
            md([{"interface": f, **{DATASETS.get(c, c): f"{v:.1f}" for c, v in r.items()}}
                for f, r in med.iterrows()]), ""]
    tr = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "findings" / "truncation"
                                                            / "*_s0.csv"))])
    tr["ok"] = tr["correct"].astype(str) == "True"
    tr["silent"] = (tr["status"] == "answered") & ~tr["ok"]
    tr["empty"] = tr["status"] == "parse_error"
    tab = tr.groupby("facade")[["ok", "silent", "empty"]].sum()
    out += ["## Truncation ablation, gpt-oss-120b (second provider), 22 large-result pairs",
            "", md([{"arm": f, "correct": int(r["ok"]), "confident wrong": int(r["silent"]),
                     "no parseable answer": int(r["empty"])} for f, r in tab.iterrows()]), ""]
    wx = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "findings"
                                                            / "truncation_provider" / "*_s0.csv"))])
    both = pd.concat([wx, d[d["family"].str.contains("large")
                                        & d.apply(lambda r: r["sf"] in LARGE_SCALE[r["ds"]], axis=1)
                                        & (d["facade"] == "generic")]])
    both["ok"] = both["correct"].astype(str) == "True"
    both["silent"] = (both["status"] == "answered") & ~both["ok"]
    both["m"] = both["model"].map(lambda s: next(v for k, v in MODELS.items() if k in s))
    tab = both.groupby(["m", "facade"])[["ok", "silent"]].sum()
    out += ["## Truncation ablation, same provider as the grid (flag arm = grid SQL)", "",
            md([{"model": m, "arm": f, "correct": int(r["ok"]),
                 "confident wrong": int(r["silent"])} for (m, f), r in tab.iterrows()]), ""]
    (ROOT / "findings" / "question_level.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
