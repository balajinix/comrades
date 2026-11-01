"""Scale-crossover experiment: modality x data volume, scripted agents, no LLM.

For every TPC-H scale factor, question and facade this runs the scripted policy,
records the communication cost, and compares the analytic cost model
(calibrated at the smallest scale factor) with what is measured at larger ones.

Usage:
    .venv/bin/python experiments/exp_scale_crossover.py [--sf 0.01 0.1 1] [--budget 128000]

Writes ``findings/scale_crossover.csv`` and ``findings/scale_crossover.md``.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from dal_lab.costmodel import ShippingModel, handle_crossover_sf
from dal_lab.facades import open_store
from dal_lab.policies import DEFAULT_BUDGET, Lab, RunResult, plan_rows, tokens_per_row
from dal_lab.tpch import ensure_tpch
from dal_lab.workload import load_questions

logger = logging.getLogger("exp_scale_crossover")
FINDINGS = Path(__file__).resolve().parents[1] / "findings"
FIELDS = ["sf", "qid", "family", "facade", "status", "calls", "turns", "schema_tokens",
          "base_tokens", "context_peak_tokens", "cumulative_input_tokens", "output_tokens",
          "total_tokens", "response_bytes", "rows_moved", "db_latency_s", "by_reference",
          "detail"]


def total(r: RunResult) -> int:
    """Billed tokens of a run: cumulative input plus output."""
    return r.cumulative_input_tokens + r.output_tokens


def main() -> None:
    """Run the sweep and write findings."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    FINDINGS.mkdir(exist_ok=True)
    questions = load_questions()
    results: dict[float, list[RunResult]] = {}
    plan_info: dict[float, dict[str, dict[str, int]]] = {}
    tpr: dict[str, float] = {}
    timings: dict[float, float] = {}
    for sf in args.sf:
        con = open_store(ensure_tpch(sf))
        lab = Lab(con)
        if not tpr:
            tpr = {t: tokens_per_row(con, t) for t in lab.catalog.tables}
        start = time.time()
        runs = []
        for q in questions:
            for fac in lab.scripted:
                runs.append(lab.run(q, fac, budget=args.budget))
        timings[sf] = time.time() - start
        plan_info[sf] = {q.id: plan_rows(con, q) for q in questions}
        results[sf] = runs
        logger.info("sf=%g: %d runs in %.1fs", sf, len(runs), timings[sf])
        con.close()

    with (FINDINGS / "scale_crossover.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for sf, runs in results.items():
            for r in runs:
                row = {k: getattr(r, k) for k in FIELDS if hasattr(r, k)}
                row.update(sf=sf, total_tokens=total(r), db_latency_s=round(r.db_latency_s, 4))
                w.writerow(row)

    (FINDINGS / "scale_crossover.md").write_text(
        report(args, questions, results, plan_info, tpr, timings))
    logger.info("wrote %s", FINDINGS)


def report(args, questions, results, plan_info, tpr, timings) -> str:  # noqa: ANN001
    """Render the markdown summary."""
    sfs = sorted(results)
    facades = list(dict.fromkeys(r.facade for r in results[sfs[0]]))
    out = ["# Scale crossover: modality x data volume (scripted agents)", "",
           f"TPC-H at SF {', '.join(f'{s:g}' for s in sfs)}; {len(questions)} questions; "
           f"context budget {args.budget:,} tokens; tokens are cl100k_base. Scripted "
           "agents assume perfect query/plan writing, so differences come from the "
           "interface alone. Total = cumulative input over all turns + output.", "",
           "## 1. Outcome and cost per facade and scale", "",
           "| SF | facade | correct | wrong | overflow | no workflow | median calls | "
           "median total tokens (correct runs) | total tokens per correct answer | "
           "runs needing > 8 calls |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for sf in sfs:
        by = defaultdict(list)
        for r in results[sf]:
            by[r.facade].append(r)
        for fac in facades:
            rs = by[fac]
            cnt = defaultdict(int)
            for r in rs:
                cnt[r.status] += 1
            ok = [r for r in rs if r.correct]
            med_tok = statistics.median(total(r) for r in ok) if ok else float("nan")
            per_correct = sum(total(r) for r in rs) / len(ok) if ok else float("inf")
            out.append(f"| {sf:g} | {fac} | {cnt['correct']}/{len(rs)} | {cnt['wrong']} | "
                       f"{cnt['overflow']} | {cnt['no_workflow']} | "
                       f"{statistics.median(r.calls for r in rs):g} | {med_tok:,.0f} | "
                       f"{per_correct:,.0f} | {sum(r.calls > 8 for r in rs)} |")
    out += ["", "## 2. Interface delivery cost (tool descriptions, per turn)", "",
            "| facade | tool-description tokens |", "|---|---|"]
    seen = {}
    for r in results[sfs[0]]:
        seen.setdefault(r.facade, r.schema_tokens)
    out += [f"| {f} | {t:,} |" for f, t in seen.items()]

    # --- fine-grained overflow: predicted (calibrated at smallest sf) vs measured
    cal = sfs[0]
    fine_cal = {r.qid: r for r in results[cal] if r.facade == "fine_grained"}
    out += ["", f"## 3. Fine-grained overflow: model calibrated at SF {cal:g} vs measured", "",
            "| question | predicted overflow SF | " + " | ".join(f"SF {s:g}" for s in sfs)
            + " | agree |", "|---|---|" + "---|" * len(sfs) + "---|"]
    agree = n = 0
    for q in questions:
        model = ShippingModel.calibrate(fine_cal[q.id].base_tokens, plan_info[cal][q.id],
                                        tpr, cal)
        sf_star = model.overflow_sf(args.budget)
        cells, ok_all = [], True
        for s in sfs:
            r = next(x for x in results[s] if x.qid == q.id and x.facade == "fine_grained")
            pred_over = s >= sf_star
            meas_over = r.status == "overflow"
            ok_all &= pred_over == meas_over
            cells.append(f"{r.status}{'' if pred_over == meas_over else ' ✗'}")
            agree += pred_over == meas_over
            n += 1
        out.append(f"| {q.id} | {'never' if math.isinf(sf_star) else f'{sf_star:.3g}'} | "
                   + " | ".join(cells) + f" | {'yes' if ok_all else 'no'} |")
    out += ["", f"Prediction agreement: {agree}/{n} (question, SF) cells. ✗ marks a miss. "
            "The one-point model assumes every scale-dependent step grows linearly with SF; "
            "it misses steps selected by key (a customer's orders, a part's line items), "
            "whose size is constant."]
    if len(sfs) >= 3:
        s1, s2, tgt = sfs[0], sfs[1], sfs[-1]
        out += ["", f"### 3b. Two-point model: per-step scaling exponent fitted on SF {s1:g} "
                f"and {s2:g}, predicting SF {tgt:g}", "",
                "| question | predicted context at target | measured | agree |",
                "|---|---|---|---|"]
        agree2 = 0
        for q in questions:
            pred_rows = {}
            for key, r2 in plan_info[s2][q.id].items():
                r1 = plan_info[s1][q.id][key]
                beta = math.log(r2 / r1) / math.log(s2 / s1) if r1 > 0 and r2 > 0 else 0.0
                pred_rows[key] = r2 * (tgt / s2) ** beta
            m = ShippingModel.calibrate(fine_cal[q.id].base_tokens, pred_rows, tpr, tgt)
            pred_ctx = m.context_tokens(tgt)
            r = next(x for x in results[tgt] if x.qid == q.id and x.facade == "fine_grained")
            ok = (pred_ctx > args.budget) == (r.status == "overflow")
            agree2 += ok
            out.append(f"| {q.id} | {pred_ctx:,.0f} ({'overflow' if pred_ctx > args.budget else 'fits'}) "
                       f"| {r.status} | {'yes' if ok else 'no'} |")
        out += ["", f"Two-point agreement at SF {tgt:g}: {agree2}/{len(questions)}."]

    # --- handles vs generic for growing answers
    out += ["", "## 4. Handles vs in-context results for growing answers", "",
            "Predicted crossover (max of the cost crossover and the 50-row inline threshold) from SF "
            f"{cal:g} calibration: handles win once the in-context answer (read once as an "
            "observation, written once as output) costs more than the handle path.", "",
            "| question | predicted crossover SF | "
            + " | ".join(f"SF {s:g} generic / handles total" for s in sfs) + " |",
            "|---|---|" + "---|" * len(sfs)]
    for q in questions:
        if q.size_class not in ("linear", "linear-small"):
            continue
        g = next(r for r in results[cal] if r.qid == q.id and r.facade == "generic")
        h = next(r for r in results[cal] if r.qid == q.id and r.facade == "handles")
        first = h.trace[0]
        h_ref = 2 * h.base_tokens + 2 * first["request_tokens"] + first["response_tokens"] + 30
        g_fixed = 2 * g.base_tokens + 2 * g.trace[0]["request_tokens"]
        g_var_per_sf = (total(g) - g_fixed) / cal
        # handles only answer by reference above the inline threshold (50 rows)
        rows_cal = max(1, g.rows_moved)
        sf_inline = 50 * cal / rows_cal
        sf_x = max(handle_crossover_sf(max(0.0, h_ref - g_fixed), g_var_per_sf), sf_inline)
        cells = []
        for s in sfs:
            gg = next(r for r in results[s] if r.qid == q.id and r.facade == "generic")
            hh = next(r for r in results[s] if r.qid == q.id and r.facade == "handles")
            win = "handles" if total(hh) < total(gg) else "generic"
            pred = "handles" if s >= sf_x else "generic"
            cells.append(f"{total(gg):,} / {total(hh):,} ({win}{'' if win == pred else ' ✗'})")
        out.append(f"| {q.id} | {sf_x:.3g} | " + " | ".join(cells) + " |")

    # --- compaction
    out += ["", "## 5. Observation compaction (4,000-char cap, unannotated)", "",
            "| SF | facade | wrong answers | uncompacted got it right | uncompacted overflowed "
            "(a visible failure turned silent) |",
            "|---|---|---|---|---|"]
    for sf in sfs:
        idx = {(r.qid, r.facade): r for r in results[sf]}
        for fac in ("generic+compact", "fine_grained+compact"):
            base = fac.split("+")[0]
            wrong = [r for r in results[sf] if r.facade == fac and r.status == "wrong"]
            rescued = sum(idx[(r.qid, base)].correct for r in wrong)
            silent = sum(idx[(r.qid, base)].status == "overflow" for r in wrong)
            out.append(f"| {sf:g} | {fac} | {len(wrong)} | {rescued} | {silent} |")

    out += ["", "## 6. Wall-clock", ""]
    out += [f"- SF {s:g}: {t:.1f} s for {len(results[s])} runs" for s, t in timings.items()]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
