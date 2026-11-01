"""Validate the gold_tools annotation against the real mcp_dal tool descriptions.

Fetches every tool's name + description from the fault proxy (which mirrors mcp_dal),
scores each tool's relevance to each task prompt, and prints the top candidates. This
turns the gold_tools mapping from a hand guess into something checked against the
ground-truth tool surface: for each task it shows the highest-ranked tools, marks the
ones currently annotated as gold, and flags any gold tool that does not rank near the
top (a sign the annotation -- or the task wording -- needs a second look).

Scoring uses sentence-transformer embeddings when available, else a transparent
lexical (token-overlap) fallback. Run from a venv with `mcp` (e.g. the
schema_mapping_agent venv, which also has sentence-transformers).

Usage:
    python harness/check_gold_tools.py --top 5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_URL = "http://localhost:8212/mcp"


def fetch_tools(url: str) -> dict:
    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def _call():
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tl = await session.list_tools()
                return {t.name: (t.description or "").strip() for t in tl.tools}

    return anyio.run(_call)


def load_tasks(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _toks(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def lexical_rank(prompt: str, tools: dict) -> list[tuple[str, float]]:
    pt = _toks(prompt)
    scored = []
    for name, desc in tools.items():
        tt = _toks(name.replace("_", " ")) | _toks(desc)
        overlap = len(pt & tt)
        scored.append((name, overlap / (len(tt) ** 0.5 + 1)))  # mild length norm
    return sorted(scored, key=lambda x: -x[1])


def embed_ranker(tasks: list[dict], tools: dict):
    """Return a function task_id -> ranked [(name, score)], or None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:  # noqa: BLE001  (offline / model not cached)
        return None
    names = list(tools)
    tdocs = [f"{n.replace('_', ' ')}: {tools[n]}" for n in names]
    tvec = model.encode(tdocs, convert_to_tensor=True, normalize_embeddings=True)
    pvec = model.encode([t["prompt"] for t in tasks], convert_to_tensor=True,
                        normalize_embeddings=True)
    sims = util.cos_sim(pvec, tvec)
    ranked = {}
    for i, t in enumerate(tasks):
        row = sorted(((names[j], float(sims[i][j])) for j in range(len(names))),
                     key=lambda x: -x[1])
        ranked[t["id"]] = row
    return lambda tid: ranked[tid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--tasks", default=str(HERE / "tasks.jsonl"))
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    tools = fetch_tools(args.url)
    tasks = load_tasks(args.tasks)
    print(f"{len(tools)} tools from {args.url}; {len(tasks)} tasks\n")

    embed = embed_ranker(tasks, tools)
    method = "embeddings (all-MiniLM-L6-v2)" if embed else "lexical token-overlap"
    print(f"scoring: {method}\n" + "=" * 72)

    misses = []
    for t in tasks:
        ranked = embed(t["id"]) if embed else lexical_rank(t["prompt"], tools)
        topk = ranked[:args.top]
        gold = set(t.get("gold_tools", []))
        top_names = [n for n, _ in topk]
        print(f"\n[{t['id']}] {t['prompt'][:78]}")
        print(f"  gold: {sorted(gold) or '(none)'}")
        for n, sc in topk:
            mark = " <== GOLD" if n in gold else ""
            print(f"    {sc:5.2f}  {n}{mark}")
        # A task is on-task-satisfiable if ANY gold tool ranks in the top-k (calling
        # any one of them counts). Flag only tasks where NO gold tool makes the cut.
        rank_of = {n: i for i, (n, _) in enumerate(ranked)}
        for g in sorted(gold - set(top_names)):
            r = rank_of.get(g, -1)
            print(f"  - gold '{g}' ranks #{r + 1}" if r >= 0
                  else f"  - gold '{g}' NOT in tool surface")
        if gold and not (gold & set(top_names)):
            misses.append((t["id"], sorted(gold)))

    print("\n" + "=" * 72)
    if misses:
        print(f"{len(misses)} task(s) with gold tools outside top-{args.top}:")
        for tid, gs in misses:
            print(f"  {tid}: {gs}")
    else:
        print(f"All annotated gold tools rank within top-{args.top}. ✓")


if __name__ == "__main__":
    main()
