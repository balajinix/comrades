"""Seed Dataset A annotation candidates from the live mcp_dal tool catalog.

For each of mcp_dal's ~105 tools this emits K candidate rows whose *gold tool is
known by construction* (the tool the template was built from), plus a draft
natural-language prompt, draft gold args (the tool's required params), and a draft
result assertion. A human then verifies/edits each in annotate.html — the machine
proposes the tedious parts; the human certifies naturalness, unambiguity, and the
assertion. Candidates are clearly marked ``source="templated"`` and ``verified``
is only set when a human accepts them in the UI.

Usage (mcp_dal on :8201):
    python gen_candidates.py --per-tool 3 --out candidates/dataset_A.jsonl
    # offline (reuse a cached catalog):
    python gen_candidates.py --catalog catalog_mcp_dal.json --per-tool 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MCP_DAL_URL = os.getenv("MCP_DAL_URL", "http://localhost:8201/mcp")

# Family inference from the tool name (priority order; first hit wins). The
# annotator can fix the family in the UI, so best-effort is fine.
FAMILY_RULES = [
    ("cost",       ("finops", "cost", "billing", "rightsizing", "anomal", "budget", "finops")),
    ("resilience", ("resilience", "resilien", "security", "cve", "complian", "vulnerab", "risk")),
    ("infra",      ("infra", "iql", "nl2iql", "infrastructure", "topology", "cmdb")),
    ("itsm",       ("incident", "change", "problem", "knowledge", "article", "catalog",
                    "story", "epic", "scrum", "sprint", "user", "group", "workflow", "approval")),
    ("apm",        ("application", "apm", "slo", "metric", "event", "trace", "website",
                    "mobile", "synthetic", "host", "release", "observ", "service")),
]


def infer_family(name: str) -> str:
    n = name.lower()
    for fam, kws in FAMILY_RULES:
        if any(k in n for k in kws):
            return fam
    return "other"


def base_phrase(desc: str) -> str:
    """First sentence of the tool description, lightly cleaned into a phrase."""
    if not desc:
        return ""
    first = re.split(r"(?<=[.!?])\s|\n", desc.strip())[0]
    first = re.sub(r"^(This tool|This)\s+", "", first, flags=re.I).strip()
    return first.rstrip(".")


def prompt_variants(name: str, desc: str, k: int) -> list[str]:
    """A few draft phrasings; the annotator rewrites these for naturalness."""
    phrase = base_phrase(desc) or name.replace("_", " ")
    low = phrase[0].lower() + phrase[1:] if phrase else phrase
    cands = [
        f"{phrase}.",
        f"Using the available data tools, {low}.",
        f"I need to {low} — can you pull that?",
        f"As an SRE, {low}. Summarize what you find.",
    ]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out[:max(1, k)]


def required_args(schema: dict) -> dict:
    req = (schema or {}).get("required") or []
    return {a: "<EDIT>" for a in req if a != "ctx"}


def fetch_catalog(url: str) -> list[dict]:
    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _go():
        async with streamablehttp_client(url) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tl = await s.list_tools()
                return [{"name": t.name, "description": t.description or "",
                         "inputSchema": t.inputSchema or {}} for t in tl.tools]
    return anyio.run(_go)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=MCP_DAL_URL)
    ap.add_argument("--catalog", default=None, help="use a cached catalog JSON instead of the live server")
    ap.add_argument("--per-tool", type=int, default=3, help="candidate prompts per tool")
    ap.add_argument("--out", default="candidates/dataset_A.jsonl")
    args = ap.parse_args()

    if args.catalog:
        tools = json.loads(Path(args.catalog).read_text())
    else:
        tools = fetch_catalog(args.url)
        (HERE / "catalog_mcp_dal.json").write_text(json.dumps(tools, indent=1))
        print(f"cached catalog -> {HERE/'catalog_mcp_dal.json'} ({len(tools)} tools)")

    # Prefer enrich_catalog.py output: more accurate family (from source location),
    # fuller descriptions, backend label, and API endpoint hints.
    enriched = {}
    ep = HERE / "catalog_mcp_dal.enriched.json"
    if ep.exists():
        enriched = {e["name"]: e for e in json.loads(ep.read_text())}

    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for t in tools:
        e = enriched.get(t["name"], {})
        fam = e.get("family_guess") or infer_family(t["name"])
        desc = e.get("description") or t.get("description", "")
        gargs = required_args(t["inputSchema"])
        for j, p in enumerate(prompt_variants(t["name"], desc, args.per_tool)):
            n += 1
            rows.append({
                "id": f"A-{t['name']}-{j+1}",
                "family": fam,
                "prompt": p,
                "gold_tool": t["name"],
                "gold_args": gargs,
                "result_assertion": "<EDIT: checkable predicate on the tool result, "
                                    "e.g. result['status']=='ok' and len(result.get('results',[]))>=1>",
                "difficulty": "",          # easy | medium | hard
                "hops": 1,
                "reasoning_type": "",       # lookup | filter | aggregate | compare | ...
                "source": "templated",
                "backend": e.get("backend", ""),
                "api_hints": e.get("api_hints", []),
                "tool_desc": base_phrase(desc)[:160],
            })
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    fams = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    print(f"wrote {len(rows)} candidates ({len(tools)} tools x ~{args.per_tool}) -> {out}")
    print(f"by family: {dict(sorted(fams.items()))}")
    print("Next: open annotate.html, Load candidates, pick this file.")


if __name__ == "__main__":
    main()
