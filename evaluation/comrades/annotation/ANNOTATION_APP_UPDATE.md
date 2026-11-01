# Note to the annotation-app session: update for the 151-tool surface + expanded tasks

**Context.** Two things changed under the annotation app (`annotate_app.py` Streamlit +
`gen_candidates.py` + `annotate.html`):

1. **mcp_dal grew 105 → 151 tools** and moved to a new dev host
   `http://internal-host:8002/mcp` (dev branch, gate
   `MCP_TOOL_PACKAGE=full`; stateless `tools/call` works — you can do live
   result-shape checks). New backends filled the thin families: infra 3→13 (incl.
   `execute_infra_iql`, graph schema/inventory/neighbor/path tools), resilience
   4→20 (demuxed Resilience: `list_resilience_*`, `get_resilience_*`), cost 4→7, apm 11→27.
2. **The task set grew 50 → 431 single-hop + multi-hop**, and — importantly — the new
   tasks are **already gold-tool-tagged by construction** (each row carries its
   `gold_tool`/`gold_tools`), so the human work shrinks to writing the
   `result_assertion` rather than re-templating prompts.

## Already provided for you (in `annotation/`)
- **`catalog_mcp_dal_v151.json`** — the live 151-tool catalog (name, title, description).
- **`catalog_mcp_dal_v151.enriched.json`** — 151 tools **with `params` + `backend`**
  (built from the new server's `tools/list` inputSchema). This is the drop-in
  replacement for `catalog_mcp_dal.enriched.json` that the app's tool-reference panel
  reads. **Backend tag is now reliable**: every tool's `title` starts with
  `[APM]`/`[Infra]`/`[Resilience]`/`[FinOps]`/`[ITSM]` — use that for
  family instead of the old name-keyword heuristics (which mis-tagged e.g.
  `get_application_topology` as infra).
- The gold-tagged task files (in `evaluation/comrades/`): `requests_read_v151.jsonl`
  (97, one/tool), `requests_large.jsonl` (431, paraphrase+slot), and the multi-hop sets
  `harness/tasks_multi.jsonl` (36) + `harness/tasks_realquery.jsonl` (15 real KG-log).

## Required app/generator changes

1. **Repoint the enriched catalog.** `annotate_app.py` loads
   `catalog_mcp_dal.enriched.json` (105 tools) for the read-only tool reference — for
   the 46 new tools `_ENRICHED.get(gold_tool)` returns `None` and the annotator sees no
   schema. Switch the path to `catalog_mcp_dal_v151.enriched.json` (or make it an env
   var). Same for `gen_candidates.py --catalog`.

2. **Family tagging from the title tag.** Replace `FAMILY_RULES` keyword inference with
   the `[Backend]` prefix in `title` (now present on every tool). Removes the
   hand-fix-family step for most rows.

3. **Ingest the already-gold-tagged tasks as candidates (new path).** Add a small
   converter that turns the new task files into annotation candidates so we don't
   re-template prompts we already wrote. Map each `{id, family, prompt, gold_tool}` →
   a candidate row with `gold_args: {}`, `result_assertion: "<EDIT...>"`,
   `difficulty/hops/reasoning_type` defaults, `source: "templated-large"`. The human
   then only verifies the prompt and **writes the assertion** — the highest-value field.
   (This replaces re-running `gen_candidates.py` for these; keep gen_candidates for
   net-new tools without a templated question.)

4. **Multi-hop support (schema change).** The app assumes a single `gold_tool`
   (`cand.get('gold_tool')`). The multi-product tasks carry `gold_tools` (a list, 2–4
   tools). Decide one of:
   - **Phase it:** annotate single-hop first (431 rows), defer multi-hop; or
   - **Extend the model:** accept a `gold_tools` list + either a per-tool assertion or a
     single final-answer assertion, and render all referenced tools in the reference
     panel. The `gold_level` enum would gain a `tools+assertion` (multi) variant.

5. **Result-assertion hints for the new tools** (the result shapes differ — add to the
   README/UI helper text):
   - `execute_infra_iql` → returns query **rows**: `len(result.get('results',[]))>=1`.
   - `list_infra_node_schemas` / `_edge_schemas` → a list of type defs.
   - `get_infra_one_hop_neighbors` → neighbor list keyed off a node id (fill
     `gold_args` with a real node id — use a live `tools/call` to discover one).
   - `list_resilience_applications` / `get_resilience_cve_details` / `list_resilience_compliance_risks`
     → Resilience record lists / a CVE record.
   - `list_finops_cost_measures` / `list_finops_vendors` → metadata lists.

6. **Update the Balance guidance.** The pool is no longer ITSM-dominated at the read
   level (read tools: itsm 30, apm 27, resilience 20, infra 13, cost 7), so the
   "ITSM-heavy, accept ~20/family and skip the surplus" note should be revised toward a
   per-family target that reflects the new balance.

## Suggested order
(1)+(2) are one-line repointings that unblock annotating the new tools immediately →
(3) converter to pull in the 431 gold-tagged single-hop tasks (biggest yield, assertions
only) → (5)/(6) helper-text + balance → (4) multi-hop last, since it needs a schema change.
