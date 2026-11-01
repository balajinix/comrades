"""Where should retry live? An LLM agent under independent vs bursty faults.

Arms (see dal_lab.faulty): native (no policy, no wait), prose (written policy plus
a wait tool), coded (DAL-side exponential backoff with jitter). Fault processes:
Bernoulli 0.3 and Gilbert–Elliott with 10 s mean bursts at the same marginal rate.
Six questions via the generic SQL facade at SF 0.01, 3 seeds, temperature 0.
Each model turn costs 1 s of virtual time; deadline 120 s.

Outcomes: correct; empty (answered "no rows" although gold has rows: a silent
failure); wrong (other wrong rows); gave_up (answered null); or deadline/other.
No arm is told how to give up, as deployed agents are not. Resume-safe.

Usage:
    set -a; source ../../comrades/agents/dal_agent/.env; set +a
    .venv/bin/python experiments/exp_llm_faults.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dal_lab.agent import run_agent
from dal_lab.faults import Bernoulli, GilbertElliott
from dal_lab.faulty import (PROSE_POLICY, WAIT_DESCRIPTOR, FaultStats, VirtualClock, inject,
                            wait_tool)
from dal_lab.facades import open_store
from dal_lab.llm import LLMGateway
from dal_lab.meter import Meter
from dal_lab.policies import Lab
from dal_lab.retry import ARMS
from dal_lab.tpch import db_path
from dal_lab.workload import gold_rows, load_questions, same_answer

OUT = Path(__file__).resolve().parents[1] / "findings"
QIDS = ("L1", "F1", "A2", "A3", "T3", "J3")
THINK_S, DEADLINE_S = 1.0, 120.0
FIELDS = ["model", "process", "arm", "seed", "qid", "outcome", "status", "steps",
          "tool_errors", "turns", "prompt_tokens", "completion_tokens", "calls", "attempts", "faults",
          "faults_in_burst", "waits", "waited_s", "dal_retries", "virtual_s"]


def _process(name: str, seed: int):
    if name == "bernoulli":
        return Bernoulli(0.3, seed=seed)
    return GilbertElliott.matched(0.3, mean_bad=10.0, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="provider/meta-llama/llama-3-3-70b-instruct")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(OUT / "llm_faults.csv"))
    ap.add_argument("--qids", nargs="+", default=list(QIDS))
    args = ap.parse_args()
    out = Path(args.out)
    traces = out.with_name(out.stem + "_traces.jsonl")
    done = set()
    if out.exists():
        with out.open() as fh:
            done = {(r["process"], r["arm"], int(r["seed"]), r["qid"]) for r in csv.DictReader(fh)}
    new_file = not out.exists()
    con = open_store(db_path(0.01))
    lab = Lab(con)
    fac = lab.facades["generic"]
    questions = {q.id: q for q in load_questions()}
    llm = LLMGateway(args.model, max_tokens=2048)
    with out.open("a", newline="") as fh, traces.open("a") as tf:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for proc_name in ("bernoulli", "ge10"):
            for arm in ("native", "prose", "coded"):
                for seed in args.seeds:
                    for qid in args.qids:
                        if (proc_name, arm, seed, qid) in done:
                            continue
                        q = questions[qid]
                        cols = [d[0] for d in con.execute(q.gold_sql).description]
                        clock, stats, meter = VirtualClock(), FaultStats(), Meter()
                        # Offset each run into the fault schedule so runs are not all
                        # aligned with the chain's start.
                        clock.advance(200.0 * (QIDS.index(qid) + 1))
                        start = clock.now
                        process = _process(proc_name, seed)
                        retry = ARMS["exp_jitter_eqwait"] if arm == "coded" else None
                        tools = inject(fac.bind(meter), process, clock, stats, retry=retry,
                                       seed=seed)
                        descriptors = list(fac.descriptors)
                        # No give-up hint: deployed agents have none, and the realistic
                        # failure is an empty "no data" answer, which we classify.
                        extra = ""
                        if arm == "prose":
                            tools["wait"] = wait_tool(clock, stats)
                            descriptors.append(WAIT_DESCRIPTOR)
                            extra = PROSE_POLICY

                        def on_turn(clock=clock, start=start):
                            clock.advance(THINK_S)
                            return "deadline" if clock.now - start > DEADLINE_S else None

                        run = run_agent(q.question, cols, descriptors, tools, meter, llm,
                                        extra_instructions=extra, on_turn=on_turn)
                        if run.status != "answered":
                            outcome = run.status
                        elif run.answer is None:
                            outcome = "gave_up"
                        else:
                            got = [tuple(r) if isinstance(r, (list, tuple)) else (r,)
                                   for r in run.answer] if isinstance(run.answer, list) else None
                            ok = got is not None and same_answer(got, gold_rows(con, q), q.ordered)
                            if ok:
                                outcome = "correct"
                            elif got == []:
                                outcome = "empty"  # silent "no data": a confident "none" answer
                            else:
                                outcome = "wrong"
                        row = {"model": args.model, "process": proc_name, "arm": arm,
                               "seed": seed, "qid": qid, "outcome": outcome,
                               "status": run.status, "steps": run.steps,
                               "tool_errors": run.tool_errors, "turns": run.llm["turns"],
                               "prompt_tokens": run.llm["prompt_tokens"],
                               "completion_tokens": run.llm["completion_tokens"],
                               "calls": run.data["calls"], "attempts": stats.attempts,
                               "faults": stats.faults,
                               "faults_in_burst": stats.faults_in_burst, "waits": stats.waits,
                               "waited_s": round(stats.waited_s, 1),
                               "dal_retries": stats.dal_retries,
                               "virtual_s": round(clock.now - start, 1)}
                        w.writerow(row)
                        fh.flush()
                        tf.write(json.dumps({**row, "fault_log": stats.log,
                                             "transcript": run.transcript}) + "\n")
                        tf.flush()
                        print(f"{proc_name} {arm:6s} s{seed} {qid} -> {outcome} "
                              f"faults={stats.faults} waits={stats.waits} turns={run.llm['turns']}",
                              flush=True)
    con.close()


if __name__ == "__main__":
    main()
