# COMRADES annotation

Turn the proxy-scored tool-calling workload (Dataset A) into a **gold,
correctness-graded** set by human-verifying machine-proposed candidates. No
server, no ports: the annotator is a single static HTML page; the generator is a
one-shot script.

## Files
- `annotate.html` — the annotation UI (open in a browser; autosaves to localStorage).
- `gen_candidates.py` — seeds candidates from the live `mcp_dal` tool catalog.
- `candidates/dataset_A.jsonl` — generated candidates to annotate.
- `catalog_mcp_dal.json` — cached tool catalog (lets you regenerate offline).

## 1. Generate candidates (already done once)
```bash
# mcp_dal must be on :8201 (read-only tools/list; does NOT touch the :8202 proxy)
../../../agents/dal_agent/.venv/bin/python gen_candidates.py --per-tool 3 --out candidates/dataset_A.jsonl
# offline regen from the cached catalog:
../../../agents/dal_agent/.venv/bin/python gen_candidates.py --catalog catalog_mcp_dal.json --per-tool 3
```
The pool is **catalog-shaped** (ITSM/PM-heavy, because that is what `mcp_dal`
exposes) and the `family` guess is rough — you rebalance and fix family while
annotating (see Balance below).

## 2. Annotate
1. Open `annotate.html` in any browser (double-click, or `open annotate.html`).
2. Enter your initials when prompted (stored locally, for provenance).
3. Click **Load candidates** → pick `candidates/dataset_A.jsonl`.
4. For each row, verify/edit then **Accept** / **Reject** / **Skip**.

Keyboard: `a` accept · `r` reject · `s` skip · `j`/`k` next/prev. Edits save as
you type; closing the tab is safe (resume by reloading the same file).

### What to check per field
- **prompt** — rewrite so it reads like a *real operator question* and has
  **exactly one** correct tool. If two tools could plausibly answer, make it
  specific or reject. (Templated prompts are deliberately rough — fix them.)
- **gold_tool** — the single correct tool. Usually right by construction; fix if
  your reworded prompt now implies a different tool.
- **gold_args** — fill any `<EDIT>` required params with a concrete value
  (e.g. an application id, a time window). `{}` is fine if none are needed.
- **result_assertion** — *the key human contribution.* Replace the `<EDIT...>`
  placeholder with a **checkable predicate on the tool result** that is true iff
  the task was answered, robust to live-data drift. Examples:
  - `result['status'] == 'ok'`
  - `len(result.get('results', [])) >= 1`
  - `any(r.get('state') == 'open' for r in result.get('incidents', []))`
  Keep it a single Python expression over a variable named `result`.
- **family** — fix the auto-guess (`cost`/`apm`/`itsm`/`resilience`/`infra`).
- **difficulty** — `easy` | `medium` | `hard`.
- **hops** — tool calls a correct answer needs (1 for single-hop).
- **reasoning_type** — `lookup` | `filter` | `aggregate` | `compare` | `trace` | …
- **notes** — anything for adjudication.

**Accept** = include in the gold export. **Reject** = wrong/ambiguous/duplicate.
**Skip** = decide later.

### Balance (important)
Aim for a **per-family target** rather than accepting everything — the pool is
ITSM-heavy. For a 100-sample v1, accept ~20/family and **skip the surplus easy
ITSM** rows. The header shows running accept/reject/skip counts; watch the
`family` mix as you go.

## 3. Export
- **Export gold** → `dataset_A.gold.jsonl` — accepted rows only, with your edits,
  `verified=true`, `annotator`, `ts`. This is the dataset the harness consumes.
- **Export progress** → `dataset_A.progress.jsonl` — every row with its status,
  to resume on another machine or hand to a second annotator.

Save the exported file back into `candidates/` (the browser downloads to your
Downloads folder).

## 4. Quality control
- Have a second annotator do a **15–20% overlap** (load the same candidates,
  annotate independently) and compare gold_tool agreement (Cohen's κ) +
  adjudicate assertion disagreements.
- Solo? Do a delayed self-review pass over ~10% of accepted rows.

## Next (not yet built)
A gold scorer that replaces the `data_obtained` proxy in `run_faults.py` /
`run_rq3.py`: match the agent's chosen tool against `gold_tool` and evaluate
`result_assertion` against the tool output. Ask Claude to wire this once a first
`*.gold.jsonl` exists.
