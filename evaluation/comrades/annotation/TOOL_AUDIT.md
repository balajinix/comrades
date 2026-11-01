# mcp_dal tool audit — expanding the thin backends

**Purpose.** The COMRADES data-retrieval benchmark draws one question per read tool, so
the question set inherits the tool surface's imbalance. ITSM exposes 29 discrete
read tools; the other four backends expose 3–11. This doc audits the current 105-tool
surface and proposes concrete **read** tools to add to the thin backends so the
benchmark can be balanced (target ≈ 12–15 read tools/backend). It is written for the
coding session that owns the mcp_dal server (to be deployed on a new dev machine,
`ssh anon@example.com`).

## Current surface (105 tools)

| Backend | Read | Write | Note |
|---|---|---|---|
| Cost / FinOps | 4 | 0 | thin |
| APM / APM | 11 | 0 | **multiplexed** (`manage_*` bundle many actions behind one tool) |
| ITSM / ITSM | 29 | 53 | discrete `list_*`/`get_*` — the model to mirror |
| Infra / Infra | 3 | 0 | thinnest; `nl2iql` *generates* IQL but cannot *execute* it |
| Resilience / Resilience | 4 | 1 | **multiplexed** (`query_resilience` has 5 query types in one tool) |

Two root causes of the imbalance, and the two corresponding fixes:

1. **Multiplexing.** APM and Resilience hide many read operations behind one tool with an
   `action`/`query_type` parameter. ITSM instead exposes each as its own tool.
   → **Demultiplex**: split each bundled read action into a discrete `get_*/list_*` tool.
   This adds tools and *reduces* per-tool parameter complexity, with no new backend
   capability required.
2. **Genuine coverage gaps.** A few common read endpoints are simply not wrapped yet.
   → **Add** the missing read endpoints (grounded list below).

---

## Infra / Infra — has 3, target ~12

**Biggest gap:** `query_infra_nl2iql` only *returns an IQL string* ("Does NOT
execute"). There is no tool to actually run it. Add:

- `execute_infra_iql(iql)` — run an IQL query and return rows. *(critical: closes the
  generate→execute loop; pairs with the existing nl2iql tool)*
- `list_infra_entity_types()` — enumerate node/relationship types in the graph (the
  schema). *Enables "what entity types exist", a real logged query.*
- `list_entities_by_type(type, filters)` — e.g. list all EC2 instances / VPCs / clusters.
- `get_entity(entity_id)` — fetch one node's properties.
- `get_entity_relationships(entity_id)` / `get_neighbors(entity_id)` — what connects to an
  entity. *Enables logged KG asks ("how does an application relate to a container?").*
- `find_path(from_entity, to_entity)` — shortest path between two nodes. *Enables "path
  between a User and a DatabaseInstance".*
- `get_infrastructure_summary()` — inventory counts by type.
- `search_entities(text)` — free-text node lookup.

## Cost / FinOps — has 4, target ~12

Currently: cost_report, rightsizing, anomalies, billing_accounts. Add the standard
FinOps reads (the prod logs asked several of these — "cost sliced by account and
country", "cost for each EC2 instance"):

- `get_cost_by_dimension(dimension)` — break cost down by service / account / region /
  tag / resource. *(or expose the cost_report's groupings as discrete tools)*
- `get_cost_forecast(window)` — projected spend.
- `get_cost_trend(window, granularity)` — time series of spend.
- `list_reservations()` + `get_reservation_utilization()` — RI / Savings Plan coverage.
- `get_unused_resources()` — idle / orphaned resources.
- `list_budgets()` + `get_budget_status()` — budgets vs actuals.
- `list_cost_dimensions()` — metadata: available dimensions/filters to query by.

## Resilience / Resilience — has 4 (1 multiplexed into 5), target ~12

`query_resilience` already multiplexes 5 query types (applications, workflows, …) and
`query_resilience_security` 3 types (cve_details, …). **Demultiplex** these into discrete
read tools, and add the SBOM/finding reads probed earlier:

- Split `query_resilience` → `list_resilience_applications`, `list_resilience_workflows`,
  `get_resilience_application(id)`, … (one per query type).
- Split `query_resilience_security` → `get_cve_details(cve_id)`, `list_security_findings(app)`,
  `get_vulnerability_summary(app)`.
- Split `query_resilience_resilience` → `list_resilience_assessments()`,
  `get_resilience_score(app)`.
- Add `get_sbom(app)` / `list_packages(app)` and `list_compliance_findings(app)` —
  the CVE/SBOM endpoints already probed against the Resilience API.

## APM / APM — has 11 (mostly multiplexed), target ~15

The 7 `manage_*` tools each bundle CRUD+report actions. **Demultiplex the read actions**
into discrete tools so the read surface matches ITSM's style:

- `manage_applications` → `list_applications`, `get_application_metrics(app)`,
  `get_application_alerts(app)`.
- `manage_slo` → `list_slos`, `get_slo_report(slo)`, `get_slo_status(slo)`.
- `manage_events` → `list_events`, `get_event(id)`.
- `manage_websites` / `manage_mobile_apps` → `list_websites` / `list_mobile_apps` (+ get).
- `manage_releases` → `list_releases`, `get_release_impact(release)`.
- Keep `get_application_topology`, `get_topology`, `get_related_hosts` as is.

---

## Suggested balanced target

| Backend | Now (read) | Target | How |
|---|---|---|---|
| Infra | 3 | ~12 | execute_iql + entity/relationship/path reads |
| Cost | 4 | ~12 | dimensioned cost + forecast + reservations + budgets |
| Resilience | 4 | ~12 | demultiplex query_resilience/security + SBOM/finding reads |
| APM | 11 | ~15 | demultiplex manage_* read actions |
| ITSM | 29 | 29 | already rich; no change |

This roughly doubles the read surface (51 → ~80) and lets the benchmark hit ~15
questions/family instead of the current 3–11. **Read-only is sufficient** for the
benchmark — no new write tools needed.

**Guardrails for the new tools** (so they fit the existing harness): keep names
discrete and self-describing (`list_*`/`get_*`), make required params minimal (the
benchmark issues these with no/default args), and return a consistent
`{status, results, ...}` shape so the fault proxy and the redundancy classifier keep
working unchanged.
