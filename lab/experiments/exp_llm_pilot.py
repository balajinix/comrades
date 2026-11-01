"""LLM pilot: the same questions through each data-access facade, with a real model.

12 questions (two per family, with and without coded workflows) × 5 facades
(generic SQL, fine-grained tools, handles, workflows, generic + unannotated
compaction) × SF 0.01 and SF 1, one seed at temperature 0. Resume-safe: rows
already in the CSV are skipped.

Usage (credentials from the environment, e.g. the dal_agent .env):
    set -a; source ../../comrades/agents/dal_agent/.env; set +a
    .venv/bin/python experiments/exp_llm_pilot.py [--model M] [--sf 0.01 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dal_lab.agent import STRATEGIES, run_agent
from dal_lab.delegation import DelegationFacade
from dal_lab.facades import open_store
from dal_lab.obfuscate import ensure_obfuscated, rename_question, rename_workflows
from dal_lab.llm import LLMGateway
from dal_lab.meter import Meter, duckdb_probe, tool_description_tokens
from dal_lab.policies import Lab
from dal_lab.tpch import db_path
from dal_lab.workload import (WORKFLOWS, WORKLOAD_DIR, gold_rows, load_questions, row_scores,
                              same_answer)

OUT = Path(__file__).resolve().parents[1] / "findings"
QIDS = ("L1", "L3", "F1", "F3", "A2", "A3", "T1", "T3", "J1", "J3", "G1", "G3")
FACADES = ("generic", "fine_grained", "handles", "workflow", "workflow+sql", "generic+compact",
           "graphql", "graphql+agg", "skills", "code")
FIELDS = ["model", "schema", "sf", "qid", "family", "has_workflow", "facade", "status", "correct",
          "steps", "tool_errors", "turns", "prompt_tokens", "completion_tokens",
          "max_prompt_tokens", "llm_latency_s", "schema_tokens", "calls",
          "response_tokens", "rows", "response_bytes", "by_reference", "wall_s",
          "delegate_turns", "delegate_prompt_tokens", "delegate_completion_tokens",
          "seed", "temperature", "db_time_s", "db_mem_peak_bytes",
          "db_temp_table_peak_bytes", "db_spill_peak_bytes", "row_precision",
          "row_recall", "agent", "reflections", "revised"]


# Interfaces that can return a handle, so the agent may answer by reference.
BY_REFERENCE = ("handles", "handles+lazy", "code", "code+prov", "sql+auto", "router",
                "sql+answer_query")


def _grade(con, lab: Lab, q, run) -> tuple[bool, float | None, float | None]:
    """(correct, row precision, row recall) of the run's answer; None when no answer."""
    gold = gold_rows(con, q)
    if run.answer_handle:
        table = next((h[run.answer_handle] for fac in (lab.handle_facade,
                                                      *lab.facades.values())
                      if isinstance(h := getattr(fac, "handles", None), dict)
                      and run.answer_handle in h), None)
        if table is None:  # an unknown handle is simply wrong
            return False, 0.0, 0.0
        cols = [d[0] for d in con.execute(q.gold_sql).description]
        have = [d[0] for d in con.execute(f'SELECT * FROM {table} LIMIT 0').description]
        sel = ", ".join(f'"{c}"' for c in cols) if set(cols) <= set(have) else "*"
        got = con.execute(f"SELECT {sel} FROM {table}").fetchall()
    elif isinstance(run.answer, list):
        try:
            got = [tuple(r) if isinstance(r, (list, tuple)) else (r,) for r in run.answer]
        except TypeError:
            return False, 0.0, 0.0
    else:
        return False, None, None
    try:
        ok = same_answer(got, gold, q.ordered)
    except TypeError:
        ok = False
    return (ok, *row_scores(got, gold))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="provider/meta-llama/llama-3-3-70b-instruct")
    ap.add_argument("--sf", nargs="+", type=float, default=[0.01, 1.0])
    ap.add_argument("--out", default=str(OUT / "llm_pilot.csv"))
    ap.add_argument("--qids", nargs="+", default=list(QIDS))
    ap.add_argument("--all-questions", action="store_true",
                    help="every question in the workload, not the 12-question TPC-H pilot set")
    ap.add_argument("--seed", type=int, default=None,
                    help="repeat index / sampling seed; part of the resume key")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="completion cap per turn; reasoning models may need more")
    ap.add_argument("--agent", choices=STRATEGIES, default="react",
                    help="agent strategy: react, chain of thought (cot) or self-reflection")
    ap.add_argument("--max-reflections", type=int, default=1,
                    help="reflection rounds for --agent reflect")
    ap.add_argument("--facades", nargs="+", default=list(FACADES))
    ap.add_argument("--schema", choices=("tpch", "obf", "tpcds", "itbench", "wide"), default="tpch",
                    help="obf: renamed TPC-H, the memorisation control; tpcds: the "
                         "broad-schema rung (all its questions unless --qids is given)")
    args = ap.parse_args()
    out = Path(args.out)
    traces = out.with_name(out.stem + "_traces.jsonl")
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["model"], r.get("schema") or "tpch", float(r["sf"]), r["qid"], r["facade"],
                     r.get("seed") or None, r.get("agent") or "react")
                    for r in csv.DictReader(fh)}
    new_file = not out.exists()
    # Append with the file's own header, so adding columns never misaligns old rows.
    fields = FIELDS
    if not new_file:
        with out.open() as fh:
            fields = next(csv.reader(fh))
        if args.agent != "react" and "agent" not in fields:
            raise SystemExit(f"{out} predates the agent column; write {args.agent} runs to a "
                             "new --out file so they are not mistaken for react runs")
    obf, ds = args.schema == "obf", args.schema == "tpcds"
    itb = args.schema == "itbench"
    if itb:  # --sf is the number of scenarios loaded: 1, 5 or 35
        from dal_lab import itbench  # noqa: PLC0415 - optional dataset

        questions = {q.id: q for q in load_questions(WORKLOAD_DIR / "itbench_questions.jsonl")}
        itb_workflows = json.loads((WORKLOAD_DIR / "itbench_workflows.json").read_text())
        if args.qids == list(QIDS):
            args.qids = list(questions)
    elif ds:
        from dal_lab import tpcds  # noqa: PLC0415 - optional dataset

        questions = {q.id: q for q in load_questions(WORKLOAD_DIR / "tpcds_questions.jsonl")}
        if args.qids == list(QIDS):
            args.qids = list(questions)
    else:
        questions = {q.id: rename_question(q) if obf else q for q in load_questions()}
    if args.all_questions:
        args.qids = list(questions)
    llm = LLMGateway(args.model, max_tokens=args.max_tokens, temperature=args.temperature,
                     seed=args.seed)
    with out.open("a", newline="") as fh, traces.open("a") as tf:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for sf in args.sf:
            if itb:
                con = open_store(itbench.ensure_itbench(f"s{int(sf)}"))
                lab = Lab(con, workflows=itb_workflows)
            elif args.schema == "wide":  # --sf is the width in columns; data is SF 1
                from dal_lab import widen  # noqa: PLC0415 - optional store

                con = open_store(widen.ensure_wide(int(sf)))
                lab = Lab(con)
            elif ds:
                con = open_store(tpcds.ensure_tpcds(sf))
                lab = Lab(con, workflows={})
            else:
                con = open_store(ensure_obfuscated(sf) if obf else db_path(sf))
                lab = Lab(con, workflows=rename_workflows(WORKFLOWS) if obf else None)
            for qid in args.qids:
                q = questions[qid]
                cols = [d[0] for d in con.execute(q.gold_sql).description]
                for fname in args.facades:
                    if (args.model, args.schema, sf, qid, fname,
                            None if args.seed is None else str(args.seed), args.agent) in done:
                        continue
                    if fname.startswith("a2a:"):  # delegate to a data agent over a facade
                        fac = DelegationFacade(lab.facades[fname[4:]], args.model,
                                               allow_handles=False)
                    else:
                        fac = lab.facades[fname]
                    meter = Meter(db_probe=duckdb_probe(con))
                    tools = fac.bind(meter)
                    t0 = time.time()
                    run = run_agent(q.question, cols, fac.descriptors, tools, meter, llm,
                                    allow_handles=fname in BY_REFERENCE,
                                    strategy=args.agent,
                                    max_reflections=args.max_reflections)
                    ok, row_p, row_r = (_grade(con, lab, q, run)
                                        if run.status == "answered" else (False, None, None))
                    for f in (lab.handle_facade, *lab.facades.values()):
                        if hasattr(f, "release"):
                            f.release()
                    row = {"model": args.model, "schema": args.schema, "sf": sf, "qid": qid,
                           "family": q.family,
                           "has_workflow": bool(q.workflow), "facade": fname,
                           "status": run.status, "correct": ok, "steps": run.steps,
                           "tool_errors": run.tool_errors,
                           "schema_tokens": tool_description_tokens(fac.descriptors),
                           "calls": run.data["calls"],
                           "response_tokens": run.data["response_tokens"],
                           "rows": run.data["rows"],
                           "response_bytes": run.data["response_bytes"],
                           "by_reference": bool(run.answer_handle),
                           "wall_s": round(time.time() - t0, 1),
                           "seed": args.seed, "temperature": args.temperature,
                           "row_precision": row_p, "row_recall": row_r,
                           "agent": args.agent, "reflections": run.reflections,
                           "revised": run.revised,
                           "db_time_s": round(run.data["latency_s"], 3),
                           "db_mem_peak_bytes": run.data["db_mem_peak_bytes"],
                           "db_temp_table_peak_bytes": run.data["db_temp_table_peak_bytes"],
                           "db_spill_peak_bytes": run.data["db_spill_peak_bytes"],
                           **run.llm}
                    if isinstance(fac, DelegationFacade):
                        row.update(delegate_turns=fac.cost.turns,
                                   delegate_prompt_tokens=fac.cost.prompt_tokens,
                                   delegate_completion_tokens=fac.cost.completion_tokens)
                    row.pop("llm_latency_s", None)
                    row["llm_latency_s"] = run.llm.get("llm_latency_s")
                    w.writerow({k: row.get(k) for k in FIELDS})
                    fh.flush()
                    tf.write(json.dumps({"model": args.model, "schema": args.schema,
                                         "sf": sf, "qid": qid,
                                         "facade": fname, "agent": args.agent,
                                         "status": run.status,
                                         "correct": ok, "transcript": run.transcript}) + "\n")
                    tf.flush()
                    print(f"sf={sf} {qid} {fname:16s} {args.agent:7s} {run.status:11s} "
                          f"correct={ok} "
                          f"steps={run.steps} prompt={run.llm['prompt_tokens']}", flush=True)
            con.close()


if __name__ == "__main__":
    main()
