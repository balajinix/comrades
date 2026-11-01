"""Build typed `Rendering` objects from mcp_kg's leaf-tool responses.

Ported from mcp_kg's `_build_renderings` + `_extract_graph` family. The leaf
tools in mcp_kg today return *raw* shapes (`GenerateFinOpsQueryOutput`,
`GenerateGraphQLQueryOutput`, plain metric dicts); rendering construction
lives only in mcp_kg's `kg` router. Direct callers (us) have to do this
work themselves.

Long-term: the right place for this logic is inside the leaf tools so any
client gets renderings for free. Until that refactor, this module is the
mirror image of mcp_server.py:_build_renderings — keep it in sync if the
upstream shapes drift.
"""
from __future__ import annotations

import json
from typing import Any

from .models import Rendering


_GRAPHQL_SCHEMA_TOOLS = {
    "generate_kg_query",
    "generate_itom_query",
    "generate_registry_query",
}


# itsm-mcp service_desk package — these all return shapes like
# {success, message, incidents|incident|users|user|articles|article}.
# Handled by a single branch in build_renderings; per-tool labels come from
# the tool name itself.
_ITSM_MCP_TOOLS = {
    "create_incident",
    "update_incident",
    "add_comment",
    "resolve_incident",
    "list_incidents",
    "get_incident_by_number",
    "get_latest_incident",
    "get_user",
    "list_users",
    "get_article",
    "list_articles",
}

# Field name in the SN response → preferred render type. The first hit wins.
# Order matters: list-shape keys come first so e.g. list_incidents picks the
# "incidents" table over a stray "incident" key.
_SN_LIST_KEYS = ("incidents", "users", "articles")
_SN_SINGLE_KEYS = ("incident", "user", "article")


# mcp-apm smart-router tools. Each tool internally dispatches to many
# sub-operations, so response shapes vary widely (catalogs vs metrics vs
# single-entity detail). The renderer probes for common envelope shapes —
# see _build_mcp_apm_renderings for the priority order. Adding a new
# manage_*/analyze_* tool only requires adding it to this set.
_MCP_APM_TOOLS = {
    "manage_applications",
    "manage_websites",
    "manage_mobile_apps",
    "manage_custom_dashboards",
    "analyze_infrastructure",
    "manage_automation",
    "manage_events",
    "manage_slo",
    "manage_releases",
}

# Envelope keys that mcp-apm might wrap a list of dicts under. Order
# matters — domain-specific keys come before generic ones so a "data" key
# carrying metadata doesn't outrank an "applications" key carrying the
# real payload.
_APM_LIST_KEYS = (
    "applications", "services", "websites", "mobileApps",
    "dashboards", "events", "releases", "slos", "automations",
    "items", "results", "data",
)


# mcp_dal product tools. Each family returns a different envelope; the
# renderers below probe for a list of records → table, or a known single
# shape, and degrade to [] (markdown fallback) for anything unrecognised.
_FINOPS_TOOLS = {
    "query_finops_cost_report",
    "query_finops_rightsizing",
    "query_finops_anomalies",
    "query_finops_billing_accounts",
}
_RESILIENCE_TOOLS = {
    "query_resilience",
    "query_resilience_resilience",
    "query_resilience_security",
    "trigger_resilience_workflow",
    "query_resilience_workflow_status",
}
_INFRA_TOOLS = {"query_infra_nl2iql"}
# Topology tools vendored from the mcp-apm fork.
_TOPOLOGY_TOOLS = {"get_topology", "get_related_hosts", "get_application_topology"}

# List-bearing keys to probe per family (first hit wins).
_FINOPS_LIST_KEYS = ("results", "result", "recommendations", "anomalies",
                           "accounts", "rows", "items", "data")
_RESILIENCE_LIST_KEYS = ("applications", "workflows", "violations", "risks",
                      "assessments", "results", "items", "data")


_METRIC_LABELS: dict[str, str] = {
    "calculate_mttr": "MTTR",
    "calculate_mttr_sev1": "MTTR (Sev1)",
    "calculate_mttr_sev2": "MTTR (Sev2)",
    "calculate_mttr_sev1_by_change": "MTTR (Sev1, change-related)",
    "calculate_mttr_sev2_by_change": "MTTR (Sev2, change-related)",
    "calculate_mtbf": "MTBF",
    "calculate_mtbf_sev1": "MTBF (Sev1)",
    "calculate_mtbf_sev2": "MTBF (Sev2)",
    "calculate_availability_by_downtime": "Availability (by downtime)",
    "calculate_rpa_by_lbt": "RPA (by LBT)",
    "calculate_incidents_detected_by_monitoring": "Incidents detected by monitoring",
    "calculate_sev1_incidents_detected_by_monitoring": "Sev1 incidents detected by monitoring",
    "calculate_sev2_incidents_detected_by_monitoring": "Sev2 incidents detected by monitoring",
    "calculate_throughput_transactions": "Throughput (transactions)",
    "calculate_availability_by_error_rate": "Availability (by error rate)",
    "calculate_slo_latency_errorbudget": "SLO latency error budget",
    "calculate_slo_error_rate_errorbudget": "SLO error-rate error budget",
}


# ---------------------------------------------------------------------------
# Graph extraction: Apollo response → neutral {nodes, edges} shape.
#
# Each extractor handles one known query field. Unknown shapes fall through
# and the caller skips the graph rendering. Safe to extend.
# ---------------------------------------------------------------------------


def _node(id_: str, label: str, type_: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": id_, "label": label, "type": type_, "properties": properties or {}}


def _edge(source: str, target: str, type_: str, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"source": source, "target": target, "type": type_, "properties": properties or {}}


def _extract_edge_list(value: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """CEG `entityGraph` / `pathBetween`: [{source, relationship, target}]."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for edge in value:
        if not isinstance(edge, dict):
            continue
        src, tgt, rel = edge.get("source"), edge.get("target"), edge.get("relationship")
        if not (src and tgt and rel):
            continue
        for name in (src, tgt):
            nid = f"Entity:{name}"
            if nid not in nodes:
                nodes[nid] = _node(nid, name, "Entity")
        edges.append(_edge(f"Entity:{src}", f"Entity:{tgt}", rel))
    return list(nodes.values()), edges


def _extract_entities(value: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """CEG `entities` / `entitiesByProduct`: Entity with nested relationship lists."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for entity in value:
        if not isinstance(entity, dict):
            continue
        name = entity.get("name")
        if not name:
            continue
        nid = f"Entity:{name}"
        scalar_props = {k: v for k, v in entity.items() if not isinstance(v, (list, dict))}
        nodes[nid] = _node(nid, entity.get("label") or name, "Entity", scalar_props)
        for field, related in entity.items():
            if not isinstance(related, list):
                continue
            for r in related:
                if not isinstance(r, dict):
                    continue
                rname = r.get("name")
                if not rname:
                    continue
                rid = f"Entity:{rname}"
                if rid not in nodes:
                    nodes[rid] = _node(rid, r.get("label") or rname, "Entity")
                edges.append(_edge(nid, rid, field.upper()))
    return list(nodes.values()), edges


def _extract_reachable(value: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """CEG `reachableFrom`: [{name, hops, via}]. Endpoint nodes only."""
    nodes: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        nid = f"Entity:{name}"
        if nid not in nodes:
            nodes[nid] = _node(nid, name, "Entity",
                               {"hops": item.get("hops"), "via": item.get("via")})
    return list(nodes.values()), []


_REGISTRY_CHILD_EDGE_TYPES = {
    "services": "HAS_SERVICE",
    "containers": "HAS_CONTAINER",
    "images": "HAS_IMAGE",
    "incidents": "AFFECTS",
}
_REGISTRY_CHILD_NODE_TYPES = {
    "services": "Service",
    "containers": "Container",
    "images": "Image",
    "incidents": "Incident",
}


def _extract_applications(value: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Registry `applications`: Application + aliases + services + containers + …"""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for app in value:
        if not isinstance(app, dict):
            continue
        app_name = app.get("name")
        if not app_name:
            continue
        app_id = f"Application:{app_name}"
        nodes[app_id] = _node(
            app_id, app_name, "Application",
            {k: v for k, v in app.items() if not isinstance(v, (list, dict))},
        )

        for alias in app.get("aliases") or []:
            if not isinstance(alias, dict):
                continue
            aid = alias.get("alias_id")
            if not aid:
                continue
            alias_nid = f"Alias:{aid}"
            nodes[alias_nid] = _node(
                alias_nid, alias.get("product_entity_name") or aid, "Alias",
                {k: v for k, v in alias.items() if not isinstance(v, (list, dict))},
            )
            edges.append(_edge(alias_nid, app_id, "RESOLVES_TO"))

        for field, child_type in _REGISTRY_CHILD_NODE_TYPES.items():
            for child in app.get(field) or []:
                if not isinstance(child, dict):
                    continue
                cname = child.get("name") or child.get("number") or child.get("alias_id")
                if not cname:
                    continue
                cid = f"{child_type}:{cname}"
                if cid not in nodes:
                    nodes[cid] = _node(cid, cname, child_type)
                etype = _REGISTRY_CHILD_EDGE_TYPES[field]
                if field == "incidents":
                    edges.append(_edge(cid, app_id, etype))
                else:
                    edges.append(_edge(app_id, cid, etype))
    return list(nodes.values()), edges


def _extract_aliases(value: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Registry `aliases`: [{alias_id, product, application: {name}}]."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for alias in value:
        if not isinstance(alias, dict):
            continue
        aid = alias.get("alias_id")
        if not aid:
            continue
        alias_nid = f"Alias:{aid}"
        nodes[alias_nid] = _node(
            alias_nid, alias.get("product_entity_name") or aid, "Alias",
            {k: v for k, v in alias.items() if not isinstance(v, (list, dict))},
        )
        app = alias.get("application")
        if isinstance(app, dict) and app.get("name"):
            app_id = f"Application:{app['name']}"
            if app_id not in nodes:
                nodes[app_id] = _node(app_id, app["name"], "Application")
            edges.append(_edge(alias_nid, app_id, "RESOLVES_TO"))
    return list(nodes.values()), edges


def _extract_cross_product(value: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Registry `crossProductCorrelation`: per-product alias-for-app tuples."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        app_name = row.get("application")
        product = row.get("product")
        prod_entity = row.get("product_entity_name")
        if not (app_name and product and prod_entity):
            continue
        app_id = f"Application:{app_name}"
        if app_id not in nodes:
            nodes[app_id] = _node(app_id, app_name, "Application")
        alias_id = f"Alias:{product}:{prod_entity}"
        nodes[alias_id] = _node(
            alias_id, f"{product}/{prod_entity}", "Alias",
            {k: v for k, v in row.items() if k != "application"},
        )
        edges.append(_edge(alias_id, app_id, "RESOLVES_TO"))
    return list(nodes.values()), edges


_GRAPH_EXTRACTORS: dict[str, Any] = {
    "entityGraph": _extract_edge_list,
    "pathBetween": _extract_edge_list,
    "entities": _extract_entities,
    "entitiesByProduct": _extract_entities,
    "reachableFrom": _extract_reachable,
    "applications": _extract_applications,
    "aliases": _extract_aliases,
    "crossProductCorrelation": _extract_cross_product,
}


def extract_graph(raw: dict[str, Any]) -> Rendering | None:
    """Detect a known Apollo-response shape and return a graph rendering, or
    None if no extractor fires / the result is empty."""
    results = raw.get("results")
    if not isinstance(results, dict):
        return None
    data = results.get("data")
    if not isinstance(data, dict):
        return None
    for field, value in data.items():
        extractor = _GRAPH_EXTRACTORS.get(field)
        if extractor is None or not isinstance(value, list) or not value:
            continue
        nodes, edges = extractor(value)
        if nodes or edges:
            return Rendering(type="graph", title=None, nodes=nodes, edges=edges)
    return None


# ---------------------------------------------------------------------------
# Per-tool builders
# ---------------------------------------------------------------------------


def _per_application_table(per_app: list[dict[str, Any]]) -> Rendering | None:
    if not per_app or not isinstance(per_app, list):
        return None
    first = per_app[0]
    if not isinstance(first, dict):
        return None
    keys = list(first.keys())
    return Rendering(
        type="table",
        title="Per-application breakdown",
        columns=[{"key": k, "header": k.replace("_", " ").title()} for k in keys],
        rows=per_app,
    )


def _build_itsm_mcp_renderings(
    tool_name: str,
    raw: dict[str, Any],
) -> list[Rendering]:
    """Render service_desk tool responses from itsm-mcp.

    List-shape responses (`incidents`/`users`/`articles`) become a table.
    Single-record responses (`incident`/`user`/`article`) become a code block
    with the JSON record — readable, no schema guessing about which fields
    matter. Failures (`success: false`) become a markdown error.
    """
    if raw.get("success") is False:
        msg = raw.get("message") or "ITSM call failed"
        return [Rendering(type="markdown", body=f"**ITSM error:** {msg}")]

    out: list[Rendering] = []

    for key in _SN_LIST_KEYS:
        rows = raw.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            columns = list(rows[0].keys())
            out.append(Rendering(
                type="table",
                title=f"ITSM {key} ({tool_name})",
                columns=[{"key": k, "header": k.replace("_", " ").title()} for k in columns],
                rows=rows[:50],
            ))
            return out

    for key in _SN_SINGLE_KEYS:
        record = raw.get(key)
        if isinstance(record, dict) and record:
            out.append(Rendering(
                type="code",
                language="json",
                title=f"ITSM {key} ({tool_name})",
                body=json.dumps(record, indent=2, default=str),
            ))
            return out

    # Unrecognised shape — surface the message if any so the UI doesn't go blank.
    msg = raw.get("message")
    if msg:
        return [Rendering(type="markdown", body=str(msg))]
    return []


def _build_mcp_apm_renderings(
    tool_name: str,
    raw: dict[str, Any],
) -> list[Rendering]:
    """Render mcp-apm smart-router responses.

    Each `manage_*` / `analyze_*` tool dispatches internally to many
    sub-actions with different response shapes. Probe in priority:

      1. Top-level list of dicts → table.
      2. Error envelope (`success: false` or `error`) → markdown error.
      3. List under a recognised envelope key (`applications`, `events`,
         `data`, …) → table.
      4. Single record under `data` / `result` / `item` → JSON code block.
      5. Bare dict → JSON code dump.
      6. Nothing useful → empty (caller falls back to markdown).
    """
    # 1. Top-level list (some tools return raw arrays, no envelope).
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict):
            cols = list(raw[0].keys())
            return [Rendering(
                type="table",
                title=f"APM ({tool_name})",
                columns=[{"key": k, "header": k.replace("_", " ").title()} for k in cols],
                rows=raw[:50],
            )]
        return []

    if not isinstance(raw, dict):
        return []

    # 2. Error envelope.
    if raw.get("success") is False or raw.get("error"):
        msg = raw.get("error") or raw.get("message") or "APM call failed"
        return [Rendering(type="markdown", body=f"**APM error:** {msg}")]

    # 3. List under a recognised envelope key.
    for key in _APM_LIST_KEYS:
        value = raw.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            cols = list(value[0].keys())
            return [Rendering(
                type="table",
                title=f"APM {key} ({tool_name})",
                columns=[{"key": k, "header": k.replace("_", " ").title()} for k in cols],
                rows=value[:50],
            )]

    # 4. Single record under common envelope keys.
    for key in ("data", "result", "item"):
        value = raw.get(key)
        if isinstance(value, dict) and value:
            return [Rendering(
                type="code",
                language="json",
                title=f"APM {tool_name} → {key}",
                body=json.dumps(value, indent=2, default=str),
            )]

    # 5. Bare dict — dump as JSON so the UI sees the response verbatim.
    if raw:
        return [Rendering(
            type="code",
            language="json",
            title=f"APM ({tool_name})",
            body=json.dumps(raw, indent=2, default=str),
        )]
    return []


def _rows_to_table(rows: list[Any], title: str) -> Rendering | None:
    """Build a table Rendering from a non-empty list of dict rows, else None."""
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        cols = list(rows[0].keys())
        return Rendering(
            type="table",
            title=title,
            columns=[{"key": k, "header": k.replace("_", " ").title()} for k in cols],
            rows=rows[:50],
        )
    return None


def _build_list_table(
    tool_name: str,
    raw: dict[str, Any],
    list_keys: tuple[str, ...],
    label: str,
) -> list[Rendering]:
    """Generic renderer for product tools that return a list of records under
    one of `list_keys` (top-level or nested under `results`). Renders the first
    such list as a table. Error envelopes (`success: false` / `status: error`)
    become a markdown error. Returns [] (→ markdown fallback) for shapes we
    don't recognise — deliberately conservative, never a raw JSON dump."""
    if raw.get("success") is False or raw.get("status") == "error":
        msg = raw.get("message") or raw.get("error") or f"{label} call failed"
        return [Rendering(type="markdown", body=f"**{label} error:** {msg}")]

    # Candidate row-lists: a list directly under a known key, or under results.
    results = raw.get("results")
    containers = [raw]
    if isinstance(results, dict):
        containers.append(results)
    if isinstance(results, list):
        t = _rows_to_table(results, f"{label} results ({tool_name})")
        if t is not None:
            return [t]

    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in list_keys:
            t = _rows_to_table(container.get(key), f"{label} {key} ({tool_name})")
            if t is not None:
                return [t]
    return []


def _build_infra_renderings(raw: dict[str, Any]) -> list[Rendering]:
    """query_infra_nl2iql returns the generated IQL in `iql_generated`
    (it does NOT execute it). Surface that JSON as a code block — it's the
    answer. Error envelopes become markdown."""
    if not isinstance(raw, dict):
        return []
    iql = raw.get("iql_generated")
    if iql is not None:
        return [Rendering(
            type="code",
            language="json",
            title="Generated IQL (Infra)",
            body=json.dumps(iql, indent=2, default=str),
        )]
    if raw.get("status") == "error" or raw.get("error"):
        msg = raw.get("message") or raw.get("error") or "Infra call failed"
        return [Rendering(type="markdown", body=f"**Infra error:** {msg}")]
    return []


def _build_topology_renderings(tool_name: str, raw: dict[str, Any]) -> list[Rendering]:
    """APM topology tools. Full infra `get_topology` is too large to graph
    (tens of thousands of nodes) → surface the summary counts as scalars. A
    bounded `get_application_topology` (service call paths) → a graph.
    `get_related_hosts` → a table."""
    if not isinstance(raw, dict):
        return []
    out: list[Rendering] = []

    summary = raw.get("summary")
    if isinstance(summary, dict):
        for label, key in (("Total nodes", "totalNodes"), ("Total edges", "totalEdges")):
            val = summary.get(key)
            if val is not None:
                out.append(Rendering(type="scalar", label=label, value=val))

    nodes, edges = raw.get("nodes"), raw.get("edges")
    if isinstance(nodes, list) and isinstance(edges, list) and 0 < len(nodes) <= 300:
        gnodes = [
            {"id": str(n.get("id") or n.get("name") or i),
             "label": str(n.get("label") or n.get("name") or n.get("id") or i),
             "type": n.get("type") or n.get("plugin") or "Node"}
            for i, n in enumerate(nodes) if isinstance(n, dict)
        ]
        gedges = [
            {"source": str(e.get("source") or e.get("from") or ""),
             "target": str(e.get("target") or e.get("to") or ""),
             "type": e.get("type") or ""}
            for e in edges if isinstance(e, dict) and (e.get("source") or e.get("from"))
        ]
        if gnodes:
            out.append(Rendering(type="graph", title=f"Topology ({tool_name})",
                                 nodes=gnodes, edges=gedges))

    for key in ("relatedHosts", "hosts", "related"):
        table = _rows_to_table(raw.get(key), f"Related hosts ({tool_name})")
        if table is not None:
            out.append(table)
            break
    return out


def build_renderings(
    tool_name: str,
    raw: dict[str, Any],
) -> list[Rendering]:
    """Project a leaf-tool response into a list of typed renderings.

    Returns `[]` (rather than None) when the shape isn't recognised — caller
    can decide whether to fall back to a markdown rendering.
    """
    # Some MCP tools (e.g. parts of mcp-apm's smart-router surface)
    # return a top-level list rather than a {data: ...}-style envelope.
    # Route those to a tool-specific renderer that knows how to handle the
    # bare-list shape; the rest of this function assumes a dict.
    if not isinstance(raw, dict):
        if tool_name in _MCP_APM_TOOLS:
            return _build_mcp_apm_renderings(tool_name, raw)
        return []

    has_structure = any(
        raw.get(k) for k in ("graphql_query", "value", "endpoint", "table")
    )
    err = raw.get("error") or (raw.get("status") == "error" and raw.get("message"))
    if err and not has_structure:
        return [Rendering(type="markdown", body=f"**Error:** {err}")]

    # 1. GraphQL generators (CEG / Registry / ITOM): code + optional graph.
    if tool_name in _GRAPHQL_SCHEMA_TOOLS:
        gql = raw.get("graphql_query") or ""
        if not gql:
            return []
        out: list[Rendering] = []
        graph_rend = extract_graph(raw)
        if graph_rend is not None:
            out.append(graph_rend)
        out.append(Rendering(
            type="code",
            language="graphql",
            title=raw.get("title") or f"Generated {tool_name}",
            body=gql,
        ))
        return out

    # 2a. itsm-mcp service_desk tools (incidents, users, articles).
    # These return {success, message, <list-or-single-record>}; project to a
    # table (list shape) or a markdown summary (single-record shape).
    if tool_name in _ITSM_MCP_TOOLS:
        return _build_itsm_mcp_renderings(tool_name, raw)

    # 2b. mcp-apm smart-router tools (manage_*/analyze_*). Each tool
    # dispatches to many sub-actions; the renderer probes for common shapes
    # (list-under-envelope-key, single-record, error envelope) before
    # falling through to a JSON dump. See _build_mcp_apm_renderings.
    if tool_name in _MCP_APM_TOOLS:
        return _build_mcp_apm_renderings(tool_name, raw)

    # 2b-i. mcp_dal FinOps tools → table of cost/rightsizing/anomaly rows.
    if tool_name in _FINOPS_TOOLS:
        return _build_list_table(tool_name, raw, _FINOPS_LIST_KEYS, "FinOps")

    # 2b-ii. mcp_dal Resilience tools → table of applications / workflows / risks.
    if tool_name in _RESILIENCE_TOOLS:
        return _build_list_table(tool_name, raw, _RESILIENCE_LIST_KEYS, "Resilience")

    # 2b-iii. Infra NL→IQL → code block with the generated IQL (the answer).
    if tool_name in _INFRA_TOOLS:
        return _build_infra_renderings(raw)

    # 2b-iv. APM topology tools → summary scalars / bounded graph / host table.
    if tool_name in _TOPOLOGY_TOOLS:
        return _build_topology_renderings(tool_name, raw)

    # 2c. ITSM query template + execution (legacy mcp_kg path).
    if tool_name == "generate_itsm_query":
        if not raw.get("table"):
            return []
        payload = {
            "table": raw.get("table"),
            "params": raw.get("params") or {},
            "date_range": raw.get("date_range"),
            "application": raw.get("application"),
        }
        out = [Rendering(
            type="code",
            language="json",
            title="ITSM query template",
            body=json.dumps(payload, indent=2, default=str),
        )]
        if raw.get("status") == "error" and raw.get("message"):
            out.append(Rendering(type="markdown", body=f"**Execution error:** {raw['message']}"))
        return out

    # 3. APM query payload.
    if tool_name == "generate_apm_query":
        if not raw.get("endpoint"):
            return []
        payload = {
            "endpoint": raw.get("endpoint"),
            "payload": raw.get("payload"),
            "from_time_ms": raw.get("from_time_ms"),
            "to_time_ms": raw.get("to_time_ms"),
            "application": raw.get("application"),
        }
        out = [Rendering(
            type="code",
            language="json",
            title="APM query payload",
            body=json.dumps(payload, indent=2, default=str),
        )]
        if raw.get("status") == "error" and raw.get("message"):
            out.append(Rendering(type="markdown", body=f"**Execution error:** {raw['message']}"))
        return out

    # 4. FinOps cost report.
    if tool_name == "generate_finops_query":
        if not raw.get("endpoint"):
            return []
        meta = {
            "endpoint": raw.get("endpoint"),
            "payload": raw.get("payload"),
            "start_date": raw.get("start_date"),
            "end_date": raw.get("end_date"),
            "vendor": raw.get("vendor"),
        }
        out = [Rendering(
            type="code",
            language="json",
            title="FinOps request",
            body=json.dumps(meta, indent=2, default=str),
        )]
        results = raw.get("results") or {}
        rows = results.get("results") if isinstance(results, dict) else None
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            keys = list(rows[0].keys())
            out.append(Rendering(
                type="table",
                title="FinOps results",
                columns=[{"key": k, "header": k.replace("_", " ").title()} for k in keys],
                rows=rows[:50],
            ))
        if raw.get("status") == "error" and raw.get("message"):
            out.append(Rendering(type="markdown", body=f"**Execution error:** {raw['message']}"))
        return out

    # 5. Metric calculators → scalar (+ optional per-app table).
    if tool_name.startswith("calculate_"):
        value = raw.get("value")
        if value is None:
            return []
        out = [Rendering(
            type="scalar",
            label=_METRIC_LABELS.get(tool_name, tool_name),
            value=value,
            unit=raw.get("unit"),
        )]
        table = _per_application_table(raw.get("per_application") or [])
        if table is not None:
            out.append(table)
        return out

    return []
