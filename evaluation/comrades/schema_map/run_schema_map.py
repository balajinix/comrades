#!/usr/bin/env python3
"""COMRADES schema-mapping task family: fault-injected runner over mcp_map_reflect.

For each gold row it drives the granular map_reflect tools as a multi-step workflow
(find_type_mapping -> find_field_mapping -> reflect_on_mapping), injecting faults at
the MCP tool-call boundary, and scores the produced mapping against gold (P/R/F1).
This is the schema-mapping analogue of run_faults.py (RQ1) -- a multi-hop, gold-scored
tool-calling task. See MCP_MAP_REFLECT_REQUIREMENTS.md for the tool contract.

Status: runnable end-to-end in --mock mode (validates orchestration/faults/scoring/CSV).
Real runs need the mcp_map_reflect server (port 8203); flip on by passing --url and
implementing _mcp_call with the `mcp` streamable-http client.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

from score import aggregate, load_dataset, prf

DEFAULT_DATASET = os.path.expanduser("~/src/schema-mapping/data/itopsgraph_dataset.jsonl")
# Remote-friendly: set MCP_MAP_REFLECT_URL (e.g. http://internal-host:8203/mcp).
DEFAULT_URL = os.getenv("MCP_MAP_REFLECT_URL", "http://localhost:8203/mcp")


class FaultInjector:
    """Seeded per-call fault injection at the tool boundary (COMRADES semantics)."""

    def __init__(self, rate: float, seed: int):
        self.rate = rate
        self.rng = random.Random(seed)

    def maybe_fault(self) -> bool:
        return self.rate > 0 and self.rng.random() < self.rate


class ToolError(RuntimeError):
    pass


# --- tool transport -------------------------------------------------------
def _mock_call(tool: str, args: dict, row: dict):
    """Mock mcp_map_reflect: returns gold-derived responses (plumbing test only)."""
    if tool == "find_type_mapping":
        return [{"target_type": row["expected_target_type"], "score": 1.0}]
    if tool == "find_field_mapping":
        # echo the gold mapping in the predicted shape
        pred = {}
        for src, tgt in (row["gold"] or {}).items():
            tgts = tgt if isinstance(tgt, list) else [tgt]
            pred[src] = [{"mapProperty": t, "confidence": 1.0} for t in tgts]
        return pred
    if tool == "reflect_on_mapping":
        return args.get("mappings", {})  # mock reflect is a no-op
    if tool in ("list_products", "list_types", "get_fields", "search_properties"):
        return []
    raise ToolError(f"unknown tool {tool}")


def _mcp_call(url: str, tool: str, args: dict):
    """Real call to mcp_map_reflect over streamable-http MCP.

    Lazy-imports `mcp` so --mock mode needs no extra deps. Run from a venv that has
    the `mcp` client (e.g. agents/dal_agent/.venv). Opens a fresh session per call; fine for
    a first run, optimize to a pooled session if throughput matters.
    """
    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _call():
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(tool, args or {})
                if getattr(res, "isError", False):
                    msg = " ".join(str(getattr(c, "text", "")) for c in (res.content or []))
                    raise ToolError(f"{tool} MCP error: {msg[:200]}")
                # Prefer structured output; FastMCP wraps a non-dict return under "result".
                sc = getattr(res, "structuredContent", None)
                if isinstance(sc, dict):
                    return sc.get("result", sc)
                if sc is not None:
                    return sc
                texts = [c.text for c in (res.content or [])
                         if getattr(c, "type", "") == "text"]
                blob = "\n".join(texts).strip()
                try:
                    return json.loads(blob) if blob else {}
                except (ValueError, TypeError):
                    return blob

    return anyio.run(_call)


def call_tool(tool, args, *, fault: FaultInjector, retries: int, mock_row, url):
    """Invoke a tool through the fault injector with a retry/give-up policy.

    Returns (result, n_calls, n_faults). Raises ToolError if it gives up.
    """
    calls = faults = 0
    for attempt in range(retries + 1):
        calls += 1
        if fault.maybe_fault():
            faults += 1
            continue  # faulted: retry if budget remains
        result = _mock_call(tool, args, mock_row) if mock_row is not None \
            else _mcp_call(url, tool, args)
        return result, calls, faults
    raise ToolError(f"{tool} gave up after {calls} attempts ({faults} faults)")


def run_row(row, fault, retries, mock, url) -> dict:
    """Drive the granular tool workflow for one gold row; score vs gold."""
    mock_row = row if mock else None
    total_calls = total_faults = 0
    gave_up_at = None
    pred = {}
    src_prod, tgt_prod = row["source_schema"], row["target_schema"]
    src_type, tgt_type = row["source_type"], row["target_type"]

    workflow = [
        ("find_type_mapping",
         {"source_product": src_prod, "source_type": src_type, "target_product": tgt_prod}),
        ("find_field_mapping",
         {"source_product": src_prod, "source_type": src_type,
          "target_product": tgt_prod, "target_type": tgt_type}),
        ("reflect_on_mapping", None),  # args filled from the previous step
    ]
    t0 = time.time()
    for tool, args in workflow:
        if tool == "reflect_on_mapping":
            args = {"mappings": pred, "source_type": src_type, "target_type": tgt_type}
        try:
            result, c, f = call_tool(tool, args, fault=fault, retries=retries,
                                     mock_row=mock_row, url=url)
            total_calls += c
            total_faults += f
            if tool == "find_field_mapping":
                pred = result
            elif tool == "reflect_on_mapping" and isinstance(result, dict):
                # reflect returns {"mappings": {...}, "reflection_feedback": "..."};
                # the pruned mapping lives under "mappings".
                pred = result.get("mappings", pred)
        except ToolError:
            gave_up_at = tool
            total_calls += retries + 1
            break

    score = prf(pred, row["gold"])
    return {
        "id": row["id"], "source": src_prod, "target": tgt_prod,
        "tool_calls": total_calls, "faults": total_faults,
        "gave_up_at": gave_up_at or "",
        "success": int(score["f1"] > 0.0 and gave_up_at is None),
        "precision": round(score["precision"], 3), "recall": round(score["recall"], 3),
        "f1": round(score["f1"], 3), "latency_s": round(time.time() - t0, 3),
        **{k: score[k] for k in ("tp", "n_pred", "n_gold")},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--url", default=DEFAULT_URL, help="mcp_map_reflect endpoint")
    ap.add_argument("--rates", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--retries", type=int, default=0, help="per-tool retry budget")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (for smoke tests)")
    ap.add_argument("--mock", action="store_true", help="use mock tools (no server)")
    ap.add_argument("--out", default="results/schema_map.csv")
    args = ap.parse_args()

    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    rates = [float(x) for x in args.rates.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(os.path.dirname(os.path.join(os.path.dirname(__file__), args.out)) or ".",
                exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    fields = ["id", "fault_rate", "seed", "retries", "source", "target", "tool_calls",
              "faults", "gave_up_at", "success", "precision", "recall", "f1",
              "latency_s", "tp", "n_pred", "n_gold"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for rate in rates:
            for seed in seeds:
                fault = FaultInjector(rate, seed)
                per = []
                for row in rows:
                    rec = run_row(row, fault, args.retries, args.mock, args.url)
                    rec.update({"fault_rate": rate, "seed": seed, "retries": args.retries})
                    w.writerow(rec)
                    per.append({k: rec[k] for k in ("precision", "recall", "f1",
                                                    "tp", "n_pred", "n_gold")})
                agg = aggregate(per)
                print(f"rate={rate} seed={seed} retries={args.retries}: "
                      f"rows={agg['n_rows']} micro_f1={agg['micro_f1']:.3f} "
                      f"macro_f1={agg['macro_f1']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
