"""Build a LARGER data-retrieval question set from the 96 read tools.

One-question-per-tool (gen_read_questions_v151.py) was an artificial cap. Here each read
tool seeds multiple questions via two independent multipliers, all keeping the same gold
tool:
  1. paraphrase frames  -- same intent, different surface wording (imperative vs. polite
     vs. tool-using; question-form handled separately so frames stay grammatical);
  2. slot-value variation -- swap concrete values grounded in the real environment
     (apps, time windows, cloud vendors, entity types) where the canonical prompt admits
     one.
De-duplicated and capped per tool for family balance. Reuses the curated base prompts and
backend tagging from gen_read_questions_v151.

Outputs (in evaluation/comrades/):
  - requests_large.jsonl        {id, family, prompt, gold_tool, source}
  - harness/tasks_large.jsonl   {id, family, hops, products, prompt, gold_tools, source}

Usage:  python annotation/gen_questions_large.py --per-tool 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from gen_read_questions_v151 import CURATED, backend, is_read

HERE = Path(__file__).resolve().parent
COMRADES = HERE.parent

# --- slot pools, grounded in the real environment (KG logs + live exploration) -------
APPS = ["demo-us-eks", "robot-shop", "var_tomcat_jvm_app", "var-scenarios-otel-demo",
        "checkout", "payments"]
WINDOWS = ["this month", "over the last 7 days", "over the last 30 days",
           "over the last quarter", "in the last 24 hours", "over the last 90 days"]
WINDOW_TOKENS = ["this month", "over the last 7 days", "in the last 7 days",
                 "over the last 30 days"]
VENDORS = ["AWS", "Azure", "GCP"]
ENTITY_TYPES = ["EC2 instances", "VPCs", "Kubernetes clusters", "RDS databases",
                "load balancers", "S3 buckets"]

IMPERATIVE_VERBS = {"list", "show", "generate", "analyze", "run", "find", "get", "fetch"}


def paraphrases(prompt: str) -> list[str]:
    """Same-intent rewordings that stay grammatical for the prompt's mood."""
    s = prompt.rstrip(" .?")
    low = s[0].lower() + s[1:]
    if prompt.rstrip().endswith("?") or s.split()[0].lower() in {
            "what", "which", "how", "are", "is", "does", "do"}:
        # keep the wh-question intact -- only prefix frames stay grammatical
        return [prompt, f"Using the available data tools, {low}?",
                f"Pull the data to answer: {low}?"]
    if s.split()[0].lower() in IMPERATIVE_VERBS:
        return [f"{s}.", f"Can you {low}?", f"Please {low}.",
                f"I need you to {low}.", f"Using the available data tools, {low}."]
    return [prompt, f"Using the available data tools, {low}."]


def slot_variants(prompt: str) -> list[str]:
    """Concrete-value swaps where the canonical prompt admits a slot."""
    out: list[str] = []
    for tok in WINDOW_TOKENS:
        if tok in prompt:
            out += [prompt.replace(tok, w) for w in WINDOWS if w != tok]
            break
    if "our applications" in prompt:
        out += [prompt.replace("our applications", f"application '{a}'") for a in APPS[:4]]
    if re.search(r"\ba specific\b.*application", prompt):
        out += [re.sub(r"a specific (\w+ )?application", f"application '{a}'", prompt)
                for a in APPS[:3]]
    if "AWS" in prompt:
        out += [prompt.replace("AWS", v) for v in VENDORS if v != "AWS"]
    if "all AWS EC2 instances" in prompt:
        out += [prompt.replace("all AWS EC2 instances", f"all {e}") for e in ENTITY_TYPES]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tool", type=int, default=8, help="max questions per read tool")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cat = json.load(open(HERE / "catalog_mcp_dal_v151.json"))
    req_rows, task_rows = [], []
    for t in cat:
        name = t["name"]
        if not is_read(name) or name not in CURATED:
            continue
        fam = backend(t)
        canonical = CURATED[name]
        # base prompts = canonical + its slot variants; each expanded by paraphrase frames
        bases = [canonical] + slot_variants(canonical)
        variants: list[str] = []
        for b in bases:
            variants += paraphrases(b)
        # dedupe preserving order, then sample down to the per-tool cap
        seen, uniq = set(), []
        for v in variants:
            if v not in seen:
                seen.add(v); uniq.append(v)
        if len(uniq) > args.per_tool:
            # always keep the canonical first, sample the rest
            rest = uniq[1:]
            rng.shuffle(rest)
            uniq = [uniq[0]] + rest[:args.per_tool - 1]
        for i, prompt in enumerate(uniq, start=1):
            rid = f"lg-{name}-{i}"
            req_rows.append({"id": rid, "family": fam, "prompt": prompt,
                             "gold_tool": name, "source": "templated-large"})
            task_rows.append({"id": rid, "family": fam, "hops": 1, "products": [fam],
                              "prompt": prompt, "gold_tools": [name],
                              "source": "templated-large"})

    with open(COMRADES / "requests_large.jsonl", "w") as f:
        for r in req_rows:
            f.write(json.dumps(r) + "\n")
    with open(COMRADES / "harness" / "tasks_large.jsonl", "w") as f:
        for r in task_rows:
            f.write(json.dumps(r) + "\n")

    import collections
    fam = collections.Counter(r["family"] for r in req_rows)
    per_tool = collections.Counter(r["gold_tool"] for r in req_rows)
    print(f"wrote {len(req_rows)} questions over {len(per_tool)} read tools "
          f"(avg {len(req_rows)/max(len(per_tool),1):.1f}/tool)")
    print("per-family:", dict(fam))


if __name__ == "__main__":
    main()
