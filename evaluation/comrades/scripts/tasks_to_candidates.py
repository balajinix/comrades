"""Convert already-gold-tagged task files into annotation candidates.

The expanded task sets (requests_large.jsonl 431, requests_read_v151.jsonl 97, and
the multi-hop harness/tasks_*.jsonl) already carry the prompt and the gold
tool(s) by construction, so we do NOT re-template prompts. This just maps each
task to a candidate row the human verifies, where the only required human input
is the ``result_assertion``. Enriches each row with the tool's backend +
description (from catalog_mcp_dal_v151.enriched.json) so the app's reference
panel shows the API surface.

    python tasks_to_candidates.py --tasks ../requests_large.jsonl \
        --out candidates/dataset_A_large.jsonl
    # multi-hop:
    python tasks_to_candidates.py --tasks ../harness/tasks_multi.jsonl \
        --out candidates/dataset_multi.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENRICHED = HERE / "catalog_mcp_dal_v151.enriched.json"
FAMILIES = {"cost", "apm", "itsm", "resilience", "infra"}
ASSERTION_PLACEHOLDER = ("<EDIT: checkable predicate on `result`, true iff answered, "
                         "e.g. len(result.get('results',[]))>=1>")


def load_enriched() -> dict:
    if not ENRICHED.exists():
        return {}
    return {t["name"]: t for t in json.loads(ENRICHED.read_text())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="gold-tagged task jsonl")
    ap.add_argument("--out", default="candidates/dataset_A_large.jsonl")
    args = ap.parse_args()

    enriched = load_enriched()
    tasks_path = Path(args.tasks)
    if not tasks_path.is_absolute():
        tasks_path = HERE / tasks_path
    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, multi = [], 0
    for line in tasks_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)
        gold_tools = t.get("gold_tools")
        gold_tool = t.get("gold_tool")
        # derive family + backend + description from the gold tool when single-hop
        fam = t.get("family")
        primary = gold_tool or (gold_tools[0] if gold_tools else None)
        e = enriched.get(primary, {}) if primary else {}
        if fam not in FAMILIES:
            fam = e.get("family_guess", fam or "other")
        row = {
            "id": t["id"],
            "family": fam,
            "prompt": t.get("prompt", ""),
            "gold_args": {},
            "result_assertion": ASSERTION_PLACEHOLDER,
            "difficulty": "",
            "hops": int(t.get("hops", 1) or 1),
            "reasoning_type": "",
            "source": t.get("source", "task-import"),
        }
        if gold_tools:
            row["gold_tools"] = gold_tools
            multi += 1
        else:
            row["gold_tool"] = gold_tool or ""
            if e.get("description"):
                row["tool_desc"] = e["description"][:160]
            if e.get("backend"):
                row["backend"] = e["backend"]
        rows.append(row)

    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    fams: dict[str, int] = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print(f"wrote {len(rows)} candidates ({multi} multi-hop) -> {out}")
    print(f"by family: {dict(sorted(fams.items()))}")
    miss = sum(1 for r in rows if r.get("gold_tool") and r["gold_tool"] not in enriched)
    if miss:
        print(f"note: {miss} rows reference a tool absent from the enriched catalog")


if __name__ == "__main__":
    main()
