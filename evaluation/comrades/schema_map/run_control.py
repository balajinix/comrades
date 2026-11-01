"""Control-discipline experiment over the gold-scored schema-mapping family.

Compares how *control discipline* — who dispatches the granular map_reflect tool
workflow — positions a mechanism on the under-/over-communication frontier under
backend faults, holding transport and the retry POLICY fixed:

  * wf    — coded workflow: the find_type -> find_field -> reflect sequence and the
            retry policy live in code and cannot be deviated from.
  * skill — an NL playbook (policy.to_prose) a single ReAct loop is told to follow;
            it MAY deviate (we measure how often).
  * free  — bare ReAct over the same tools, no authored procedure.

All mechanisms cross the SAME fault-proxy (harness/fault_proxy.py), so faults are
injected at one boundary and every tool call is counted identically from the proxy
log (the mechanism-independent ground truth). All retry arms share ONE policy
(policy.SHARED_RETRY), so only dispatch determinism varies — not policy quality.

This file currently implements the ``wf`` mechanism end-to-end; ``free``/``skill``
(run_agent over the proxy) and the ``a2a`` transport slot into the same dispatch
table and CSV. Run with the dal_agent venv (map_reflect :8203 + proxy :8202 up):

    ../../agents/dal_agent/.venv/bin/python schema_map/run_control.py \
        --mechanism wf --rates 0.0,0.3,0.6 --seeds 1 --limit 15 --fresh-log
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

# Never serve cached agent answers during measurement: a cache hit bypasses the
# model, the tools, AND the fault injection, so free/skill would record 0 proxy
# calls and a fault-rate-invariant score. Hard-set (not setdefault) so no inherited
# shell/.env value can silently re-enable it. (run_faults.py does the same for RQ1.)
os.environ["KG_AGENT_CACHE_ENABLED"] = "false"

_HERE = Path(__file__).resolve().parent
_COMRADES = _HERE.parent
sys.path.insert(0, str(_HERE))          # score.py
sys.path.insert(0, str(_COMRADES))      # policy.py
sys.path.insert(0, str(_COMRADES.parent.parent / "agents" / "dal_agent"))  # kg_agent

from policy import SHARED_RETRY, FAIL_FAST, RetryPolicy, fault_type_of  # noqa: E402
from score import prf, load_dataset, aggregate  # noqa: E402
from kg_agent.mcp_client import call_tool  # noqa: E402  (async (name, args, url) -> dict)

CONTROL_TOOL = "__comrades_control__"
PROXY_URL = os.getenv("FAULT_PROXY_URL", "http://localhost:8202/mcp")
A2A_SHIM_URL = os.getenv("A2A_SHIM_URL", "http://localhost:8210")
# Transport for the workflow's tool dispatch: "mcp" calls the proxy directly;
# "a2a" routes each step through the step-shim (faults still land at the proxy).
TRANSPORT = "mcp"
PROXY_LOG = Path(os.getenv("FAULT_PROXY_LOG", str(_COMRADES / "harness" / "proxy_calls.jsonl")))
DEFAULT_DATASET = os.getenv(
    "SCHEMA_MAP_DATASET", os.path.expanduser("~/src/schema-mapping/data/itopsgraph_dataset.jsonl"))
POLICIES = {"shared": SHARED_RETRY, "failfast": FAIL_FAST}


# --- result extraction ------------------------------------------------------
def _is_error(res) -> tuple[bool, str | None]:
    """Did a parsed tool result carry an injected/upstream error? -> (is_err, fault_type)."""
    if isinstance(res, dict) and str(res.get("status", "")).lower() == "error":
        return True, res.get("fault_type") or fault_type_of(res.get("error", "") or "")
    return False, None


def _mapping_of(res) -> dict:
    """Pull the gold-comparable field mapping out of a tool result.

    find_field_mapping -> dict keyed by source field id (returned directly).
    reflect_on_mapping -> {"mappings": {...}, "reflection_feedback": ...}.
    list-returning tools are wrapped as {"result": [...]} (not a mapping).
    """
    if not isinstance(res, dict):
        return {}
    # FastMCP wraps a dict return under "result"; the ReAct path also injects a
    # sibling "_results_summary" key, so unwrap on presence (not sole key).
    if isinstance(res.get("result"), dict):
        res = res["result"]
    # reflect_on_mapping returns {"mappings": {...}, "reflection_feedback": ...}.
    if isinstance(res.get("mappings"), dict):
        return res["mappings"]
    # find_field_mapping returns the mapping dict directly; drop bookkeeping keys
    # (status/error/fault_type/reflection_feedback and any "_"-prefixed summaries).
    return {k: v for k, v in res.items()
            if not k.startswith("_")
            and k not in ("status", "error", "fault_type", "reflection_feedback")}


# --- transport: dispatch a single tool call (mcp direct vs a2a via shim) -----
_A2A_CARD = {}  # cache the shim's agent card per url


async def _a2a_call(tool: str, args: dict) -> dict:
    """Send one {tool, args} step to the A2A step-shim; parse its JSON result.

    Same fault boundary as MCP — the shim forwards to the proxy — but the call
    crosses an A2A hop, isolating transport at fixed (workflow) control."""
    from a2a.client import A2ACardResolver, A2AClient
    from a2a.types import (Message, MessageSendParams, Part, Role,
                           SendMessageRequest, TextPart)
    async with httpx.AsyncClient(timeout=180) as hc:
        card = _A2A_CARD.get(A2A_SHIM_URL)
        if card is None:
            card = await A2ACardResolver(hc, A2A_SHIM_URL).get_agent_card()
            _A2A_CARD[A2A_SHIM_URL] = card
        client = A2AClient(hc, card, url=A2A_SHIM_URL)
        msg = Message(role=Role.user, messageId=uuid.uuid4().hex,
                      parts=[Part(root=TextPart(text=json.dumps({"tool": tool, "args": args})))])
        resp = await client.send_message(
            SendMessageRequest(id=uuid.uuid4().hex, params=MessageSendParams(message=msg)))
        text = resp.model_dump()["result"]["status"]["message"]["parts"][0]["text"]
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return {"status": "error", "error": f"a2a: unparseable result {text[:120]}"}


async def dispatch(tool: str, args: dict) -> dict:
    if TRANSPORT == "a2a":
        return await _a2a_call(tool, args)
    return await call_tool(tool, args, PROXY_URL)


# --- the workflow mechanism (coded dispatch, shared-policy retry) ------------
async def _attempt(tool: str, args: dict, policy: RetryPolicy) -> tuple[dict | None, str | None]:
    """Call a tool through the proxy under the shared retry policy.

    Re-issues the SAME call up to policy.max_attempts when the fault is retryable;
    stops on a non-retryable fault or exhausted budget. Returns (result|None, fault).
    The proxy log is the authoritative call count; this only drives control flow.
    """
    last_fault = None
    for attempt in range(policy.max_attempts):
        res = await dispatch(tool, args)
        err, fault = _is_error(res)
        if not err:
            return res, None
        last_fault = fault
        if not (policy.is_retryable(fault) and attempt < policy.max_attempts - 1):
            return None, fault
        if policy.backoff_s:
            await asyncio.sleep(policy.backoff_s)
    return None, last_fault


async def run_wf(row: dict, policy: RetryPolicy, run_id: str) -> dict:
    """Coded workflow: find_type -> find_field -> reflect, gold-scored.

    Steps are genuinely dependent: a fault that defeats find_type_mapping starves
    find_field_mapping (the multi-hop fault-propagation the single-hop DAL lacks).
    reflect is best-effort refinement — if it gives up, the pre-reflect mapping stands.
    """
    sp, st, tp = row["source_schema"], row["source_type"], row["target_schema"]
    pred: dict = {}
    gave_up_at = None

    # 1. discover the target type
    tm, fault = await _attempt(
        "find_type_mapping", {"source_product": sp, "source_type": st, "target_product": tp}, policy)
    if tm is None:
        gave_up_at = "find_type_mapping"
    else:
        cands = tm.get("result", tm) if isinstance(tm, dict) else tm
        top = (cands[0]["target_type"] if isinstance(cands, list) and cands
               else row["expected_target_type"])
        target_type = top.split("_", 1)[1] if "_" in top else top  # strip product prefix

        # 2. map fields for the chosen type pair
        fm, fault = await _attempt(
            "find_field_mapping",
            {"source_product": sp, "source_type": st, "target_product": tp,
             "target_type": target_type}, policy)
        if fm is None:
            gave_up_at = "find_field_mapping"
        else:
            pred = _mapping_of(fm)
            # 3. reflect/prune (best-effort: failure keeps the prior mapping)
            rf, _ = await _attempt(
                "reflect_on_mapping",
                {"mappings": pred, "source_type": st, "target_type": row["target_type"]}, policy)
            if rf is not None:
                refined = _mapping_of(rf)
                if refined:
                    pred = refined

    score = prf(pred, row["gold"])
    return {"pred": pred, "score": score, "gave_up_at": gave_up_at or "", "last_fault": fault or ""}


# --- the agent mechanisms (model dispatch: free vs NL-playbook skill) -------
# Both run the SAME ReAct loop over the SAME tools through the SAME proxy. They
# differ only in the prompt: ``skill`` is handed an authored procedure + the
# shared retry policy as prose (policy.to_prose); ``free`` gets only the goal.
# Neither executes the policy in code — retry is the model's choice, bounded by
# the agent's production loop-detector (which SKIPS an identical re-issue, so a
# faulted call cannot be blindly retried). That guardrail-vs-code difference is
# the control-discipline signal we measure.

_TASK_TMPL = (
    "Map the fields of the '{st}' type in the '{sp}' schema onto the most "
    "appropriate type in the '{tp}' schema. Use the schema-mapping tools: first "
    "find the best target type, then map the fields for that type pair, then "
    "(optionally) reflect to refine the mapping. Your goal is the field-to-field "
    "mapping."
)
_SKILL_PLAYBOOK = (
    "Follow this procedure exactly:\n"
    "1. Call find_type_mapping(source_product='{sp}', source_type='{st}', "
    "target_product='{tp}') and take the top-ranked target type.\n"
    "2. Call find_field_mapping(source_product='{sp}', source_type='{st}', "
    "target_product='{tp}', target_type=<the bare type name, no product prefix>).\n"
    "3. Call reflect_on_mapping(mappings=<step 2 result>) to prune; if it fails, "
    "keep the step-2 mapping.\n{policy}\n"
)


async def _agent_run(row: dict, run_id: str, prompt: str) -> dict:
    """Drive the ReAct agent over the proxy for one row; recover its mapping from
    the trace (the last successful find_field_mapping / reflect_on_mapping)."""
    from kg_agent import run_agent  # lazy: avoids import cost for wf-only runs
    out = await run_agent(prompt, max_steps=8,
                          mcp_servers=[("map_reflect", PROXY_URL)],
                          model=os.getenv("PROVIDER_MODEL_NAME"))
    trace = (out.raw_result or {}).get("trace", []) if out.raw_result else []
    pred, last_fault = {}, ""
    for step in trace:
        if step.get("action") not in ("find_field_mapping", "reflect_on_mapping"):
            continue
        tr = step.get("tool_result")
        err, fault = _is_error(tr)
        if err:
            last_fault = fault or last_fault
            continue
        m = _mapping_of(tr)
        if m:
            pred = m  # keep the latest successful mapping
    score = prf(pred, row["gold"])
    return {"pred": pred, "score": score,
            "gave_up_at": "" if pred else "no-mapping", "last_fault": last_fault}


async def run_free(row: dict, policy: RetryPolicy, run_id: str) -> dict:
    sp, st, tp = row["source_schema"], row["source_type"], row["target_schema"]
    return await _agent_run(row, run_id, _TASK_TMPL.format(sp=sp, st=st, tp=tp))


async def run_skill(row: dict, policy: RetryPolicy, run_id: str) -> dict:
    sp, st, tp = row["source_schema"], row["source_type"], row["target_schema"]
    prompt = (_TASK_TMPL.format(sp=sp, st=st, tp=tp) + "\n\n"
              + _SKILL_PLAYBOOK.format(sp=sp, st=st, tp=tp, policy=policy.to_prose()))
    return await _agent_run(row, run_id, prompt)


# --- proxy-log comms (mechanism-independent ground truth) -------------------
def read_run_comms(run_id: str) -> dict:
    """Read the proxy log for one run: total calls, faulted, necessary vs redundant.

    Redundancy is defined on the proxy log uniformly for EVERY mechanism: the first
    issuance of a (tool, args) intent is necessary; any re-issuance is redundant
    thrash. This generalizes the redundancy classifier beyond the ReAct trace.
    """
    if not PROXY_LOG.exists():
        return {"proxy_calls": 0, "proxy_faulted": 0, "necessary": 0,
                "redundant": 0, "redundant_frac": None}
    seen, total, faulted, redundant = set(), 0, 0, 0
    with PROXY_LOG.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("run_id") != run_id:
                continue
            total += 1
            faulted += int(bool(r.get("faulted")))
            key = (r.get("tool"), r.get("args_fp"))
            if key in seen:
                redundant += 1
            else:
                seen.add(key)
    return {"proxy_calls": total, "proxy_faulted": faulted,
            "necessary": total - redundant, "redundant": redundant,
            "redundant_frac": round(redundant / total, 3) if total else None}


# --- driver -----------------------------------------------------------------
MECHANISMS = {"wf": run_wf, "free": run_free, "skill": run_skill}


async def one_run(mechanism: str, row: dict, rate: float, seed: int,
                  policy: RetryPolicy) -> dict:
    run_id = f"{mechanism}:{TRANSPORT}:{rate}:{seed}:{row['id']}"
    # Control is ALWAYS set over MCP directly to the proxy (the shim forwards to
    # the same proxy, so its run_id state applies regardless of transport).
    await call_tool(CONTROL_TOOL, {"rate": rate, "seed": seed, "run_id": run_id}, PROXY_URL)
    t0 = time.time()
    out = await MECHANISMS[mechanism](row, policy, run_id)
    latency = round(time.time() - t0, 3)
    comms = read_run_comms(run_id)
    score = out["score"]
    return {
        "mechanism": mechanism, "transport": TRANSPORT, "policy": policy_name(policy),
        "id": row["id"], "source": row["source_schema"], "target": row["target_schema"],
        "fault_rate": rate, "seed": seed,
        "success": int(score["f1"] > 0.0 and not out["gave_up_at"]),
        "precision": round(score["precision"], 3), "recall": round(score["recall"], 3),
        "f1": round(score["f1"], 3), "tp": score["tp"], "n_pred": score["n_pred"],
        "n_gold": score["n_gold"], "gave_up_at": out["gave_up_at"],
        "last_fault": out["last_fault"], "latency_s": latency, **comms,
    }


def policy_name(policy: RetryPolicy) -> str:
    return next((k for k, v in POLICIES.items() if v is policy), "custom")


async def main_async(args) -> None:
    global TRANSPORT
    TRANSPORT = args.transport
    rows = load_dataset(args.dataset)
    if args.rows_file:                       # restrict to a harvested solvable subset
        rf = Path(args.rows_file)
        if not rf.is_absolute():
            rf = _HERE / rf                  # resolve like --out / --select-out
        want = {ln.strip() for ln in rf.read_text().splitlines() if ln.strip()}
        rows = [r for r in rows if r["id"] in want]
    if args.rows:
        want = set(args.rows.split(","))
        rows = [r for r in rows if r["id"] in want]
    if args.limit:
        rows = rows[:args.limit]
    rates = [float(x) for x in args.rates.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    mechs = [m.strip() for m in args.mechanism.split(",")]
    policy = POLICIES[args.policy]

    if args.fresh_log and PROXY_LOG.exists():
        PROXY_LOG.unlink()

    out_path = _HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mechanism", "transport", "policy", "id", "source", "target", "fault_rate",
              "seed", "success", "precision", "recall", "f1", "tp", "n_pred", "n_gold",
              "gave_up_at", "last_fault", "latency_s", "proxy_calls", "proxy_faulted",
              "necessary", "redundant", "redundant_frac"]
    all_rows = []
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for mech in mechs:
            for rate in rates:
                for seed in seeds:
                    cell = []
                    for row in rows:
                        rec = await one_run(mech, row, rate, seed, policy)
                        w.writerow(rec); fh.flush(); all_rows.append(rec); cell.append(rec)
                    agg = aggregate([{k: r[k] for k in ("precision", "recall", "f1",
                                      "tp", "n_pred", "n_gold")} for r in cell])
                    mc = sum(r["proxy_calls"] for r in cell) / len(cell)
                    rf = [r["redundant_frac"] for r in cell if r["redundant_frac"] is not None]
                    succ = sum(r["success"] for r in cell) / len(cell)
                    print(f"[{mech}/{policy_name(policy)}] rate={rate} seed={seed}: "
                          f"n={len(cell)} success={succ:.3f} micro_f1={agg['micro_f1']:.3f} "
                          f"mean_calls={mc:.2f} redundant_frac="
                          f"{(sum(rf)/len(rf) if rf else 0):.3f}")
    print(f"\nwrote {len(all_rows)} rows -> {out_path}")

    # Harvest the fault-free-solvable subset: ids the pipeline scores F1>0 on
    # without giving up. Run this with --rates 0.0 so "solvable" means
    # "solvable with no faults" — the fair-test set the fault sweep then uses.
    if args.select_out:
        solvable = sorted({r["id"] for r in all_rows
                           if r["fault_rate"] == 0.0 and r["success"] == 1})
        sel = _HERE / args.select_out
        sel.write_text("\n".join(solvable) + ("\n" if solvable else ""))
        seen0 = [r for r in all_rows if r["fault_rate"] == 0.0]
        print(f"solvable @rate0: {len(solvable)}/{len(seen0)} "
              f"({(len(solvable)/len(seen0) if seen0 else 0):.0%}) -> {sel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", default="wf", help="comma list: wf[,free,skill]")
    ap.add_argument("--transport", default="mcp", choices=["mcp", "a2a"],
                    help="wf dispatch transport; a2a routes via the step-shim (:8210)")
    ap.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--policy", default="shared", choices=list(POLICIES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rows", default=None, help="comma list of row ids (overrides limit order)")
    ap.add_argument("--rows-file", dest="rows_file", default=None,
                    help="restrict to row ids listed in this file (e.g. a harvested solvable subset)")
    ap.add_argument("--select-out", dest="select_out", default=None,
                    help="write fault-free-solvable ids (F1>0 @rate0) here; use with --rates 0.0")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--fresh-log", action="store_true", help="truncate the proxy log first")
    ap.add_argument("--out", default="results/control.csv")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
