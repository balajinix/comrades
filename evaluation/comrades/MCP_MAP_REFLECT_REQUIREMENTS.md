# Requirements: `mcp_map_reflect` — an MCP server for schema mapping

For the schema-mapping team/session (repo: `~/src/schema-mapping`).
Requested by the COMRADES (ICSE) experiments. Goal: expose the **map_reflect** schema-
mapping capability as a standalone **MCP server** so (a) other agents can call it for
exploratory metadata / schema-mapping questions, and (b) COMRADES can inject faults at
its MCP tool boundary and score answers against the gold dataset.

## Why this is needed
- The schema-mapping agent does its mapping **in-process** (map_reflect → provider LLM). It
  exposes **no** mapping capability over MCP — its only MCP tools today are storage
  (`populate_graphdb`, `generate_wiki` in `agents/schema_mapping_agent/app/mcp_server.py`).
- COMRADES measures communication robustness by injecting faults at an **MCP tool
  boundary**. With no map-reflect MCP tool, schema mapping can't be put under fault
  injection, and other agents can't call it as a tool.
- map_reflect needs **only provider creds** (no VPN / no backend), so an MCP server around
  it produces **real** mappings that can be scored against gold — fixing the canned-data /
  brittle-success-metric problem we hit with mcp_dal.

## Workload + ground truth this must serve
`~/src/schema-mapping/data/itopsgraph_dataset.jsonl` — **521 rows, 100%
gold**, schema `{id, utterance, source_schema, target_schema, expected_target_type,
mapping, metadata{confidence}}`. It is a **many-to-many bipartite** field-mapping dataset
spanning COMRADES backends as source/target (apm, finops, infra, resilience,
optimizer, hcp/terraform, aws, ns1, packer, edge, base, …). Larger set available:
`itgraph_dataset_high_confidence.jsonl` (1,501). The server's output **must be directly
comparable to the `mapping` gold** so COMRADES can score precision/recall/F1 over field pairs.

## Tech / conventions (match mcp_dal so it slots into COMRADES)
- **FastMCP, `streamable-http` transport** on a fixed port (propose **8203**; mcp_dal=8201,
  dal_agent=8005). Endpoint `/mcp`, same as mcp_dal — so the COMRADES client + fault
  injector treat it identically.
- One MCP **tool call per mapping request** (clean, single fault-injection boundary).
- Deterministic: temperature=0 (map_reflect already does this); accept an optional `seed`.
- Reuse the existing map_reflect engine (`map_reflect/map_reflect_agent.py`) — do **not**
  write a new mapping algorithm. Drive `_find_mappings` directly for schema-pair requests
  (no utterance/retrieval needed when both schemas are given), as the vendored
  `agents/schema_mapping_agent/app/agent.py:find_mappings` already does.

## Tools to expose (MCP tool surface)

**Design principle (important for this paper).** COMRADES injects faults at the tool-call
boundary; the robustness signal scales with how many *distinct* tools an agent must
orchestrate per task. A single monolithic `map_schema` = one call = trivial. So decompose
map_reflect's real pipeline into **granular tools** that chain into a 5--7 step workflow.
This makes schema mapping the paper's **multi-hop, gold-scored tool-calling task** (the 5
DAL families are mostly single-hop). Map each tool to a real map_reflect stage.

**Tier 1 -- metadata exploration** (the "other agents call schema mapping for exploratory
questions to understand the metadata" use case):
1. `list_products() -> list[str]` — available schemas/products.
2. `list_types(product: str) -> list[str]` — types in a product.
3. `get_fields(product: str, type: str) -> list[{name, type, description}]` — fields of a type.
4. `search_properties(query: str, product: str | None = None, k: int = 10) -> list[{field_id, score}]`
   — semantic retrieval over properties (map_reflect's ChromaDB stage); for "find a property like X".

**Tier 2 -- mapping, coarse → fine:**
5. `find_type_mapping(source_product: str, source_type: str, target_product: str) -> list[{target_type, score}]`
   — API/type-level: candidate target **types** for a source type (the `expected_target_type`
   granularity in the gold dataset).
6. `find_field_mapping(source_product: str, source_type: str, target_product: str,
   target_type: str) -> dict`
   — field→field mappings for a chosen type pair (map_reflect's `_find_mappings` core).
   Returns `{ "<source_field_id>": [ {"mapProperty": "<target_field_id>", "confidence":
   float, "justification": str}, ... ], ... }`, ids `"{product}_{Type}.{field}"` (gold format).
   Bipartite many-to-many → values are lists.

**Tier 3 -- reflection** (map_reflect's signature stage; `reflect_and_prune`, currently
disabled internally — expose it as a tool):
7. `reflect_on_mapping(mappings: dict, source_type: str, target_type: str) -> dict`
   — critique + prune a proposed mapping; return validated pairs with revised confidence +
   reasons. Gives the agent a self-correction step (interesting under faults).

**Optional convenience (NOT for the robustness runs):**
8. `map_schema(source_product, target_product, source_type=None, utterance=None) -> dict`
   — runs the whole pipeline in one call, for the simple A2A path. The granular Tier 1--3
   tools are what the tool-calling experiments use.

### Why granular tools help the tool-calling experiments
- A task becomes `search/list → find_type_mapping → get_fields(both) → find_field_mapping →
  reflect_on_mapping` = 5--7 calls of *different* tools: real tool selection + chaining +
  mid-chain recovery when a call is faulted (unlike the single-tool DAL lookups).
- Likely **flips RQ1's story**: a multi-step task under faults can *amplify* communication
  (retry individual steps) where the single-step DAL agent *collapses* — a stronger contrast.
- `reflect_on_mapping` is itself a robustness probe: does self-correction recover a
  degraded/partially-faulted mapping?
- Sharpens the **redundancy classifier**: reflect/re-validate = necessary recovery; repeated
  identical `find_field_mapping` = redundant thrash.
- **Gold-backed success** (P/R/F1 vs the 521 gold) — a real metric, no prose proxy.

## Backing data / bootstrap
- Load, at startup, the property lookups for the supported products (the same lookups
  map_reflect uses, e.g. `data/itopsgraph_api_lookup.json`, or per-product `data/*.graphql`
  via `scripts/parse_product_graphql.py`). Build the combined `{product}_{Type}.{field}`
  lookup covering all source/target products in the 521 dataset.
- For schema-pair calls, drive `_find_mappings` directly (skip ChromaDB retrieval). Only the
  metadata-exploration / utterance-only paths need the index.

## COMRADES integration requirements (the must-haves)
- **No VPN / external network** — runs on provider creds only.
  Env: `PROVIDER_API_KEY` (or `PROVIDER_APIKEY`), `PROVIDER_PROJECT_ID`, `PROVIDER_URL`,
  optional `MAP_REFLECT_MODEL` (default `meta-llama/llama-3-3-70b-instruct`).
- **Errors surface as MCP tool errors** (so the COMRADES client-side fault injector can wrap
  calls and a faulted call is a clean failure the agent sees).
- **Stable/reproducible** output for a fixed (input, model, seed) so rate-0 baselines are
  meaningful.
- Deps: `chromadb`, `sentence-transformers`, `networkx`, `pyyaml`, `tqdm`, `litellm`,
  `provider_sdk` (already map_reflect's deps).

## Non-goals
- Not the storage tools (`populate_graphdb`/`generate_wiki`) — those already exist.
- Not a new mapping model — reuse map_reflect.
- Scoring (P/R/F1 vs gold) lives in the **COMRADES** harness, not the server; the server just
  needs to return the gold-comparable mapping shape.

## Acceptance criteria
1. `tools/list` over `http://localhost:8203/mcp` returns the Tier 1--3 tools (+ optional
   `map_schema`).
2. `find_field_mapping("finops","StringFilterInput","optimizer","FilterInput")`
   returns mappings in the gold shape; spot-check ≥5 gold rows of `itopsgraph_dataset.jsonl`
   (non-empty, field ids well-formed `{product}_{Type}.{field}`).
3. `find_type_mapping("finops","StringFilterInput","optimizer")` returns ranked
   candidate target types (incl. the gold `expected_target_type` near the top on spot-checks).
4. Boots and answers with only provider creds set (no VPN, no backend).
5. A plain MCP client can call every tool; wrapping a call to raise on demand (fault
   injection) yields a clean tool error.
6. temperature=0 (+ optional seed) → repeatable output on the same input.

## Interface contract to lock with COMRADES (so both sides align)
- Port **8203**, endpoint `/mcp`, transport `streamable-http`.
- Tool names + I/O shapes exactly as above (esp. the `find_field_mapping` return shape and
  the `{product}_{Type}.{field}` id convention), so an agent can chain Tier 1→2→3.
- COMRADES will add: a `schema_map` task family (the 521 questions), a runner that drives an
  agent through the granular tools under fault injection, and a gold P/R/F1 scorer.
</content>
