"""Project a ReAct trace into the KgOutput shape the UI expects.

Per ui/UI_INTEGRATION.md §3 the UI consumes a fixed envelope; this module
is the boundary that adapts our internal trace to that envelope.

Strategy:
- `tool_invoked` = the LAST leaf tool the agent called (per brief §6,
  "whatever leaf made the final call"). For pure-LLM-final answers with
  no tool call, this is None.
- `routing_reason` = a chain summary like "agent chain: SN → Registry →
  FinOps" so the UI's italic routing tail is still informative.
- `renderings` = concatenation of each tool_result's renderings in the
  order the agent called them, plus a trailing `markdown` rendering with
  the final answer if the tool calls didn't already produce one.
- `source` = the source-label of the last leaf tool's data (CEG / Registry /
  ITOM / ITSM / APM / FinOps), so the answer panel header is
  meaningful.
- `raw_result` = a structured dump of the trace so the UI's "Raw Result"
  accordion gives full provenance.
"""
from __future__ import annotations

from typing import Any

from .models import Graph, Rendering, TraceStep, KgOutput
from .renderings import build_renderings


# Map a generate_*_query name to a UI-visible source label.
_SOURCE_FOR: dict[str, str] = {
    "generate_kg_query": "CEG",
    "generate_registry_query": "Registry",
    "generate_itom_query": "ITOM",
    "generate_itsm_query": "ITSM",
    "generate_apm_query": "APM",
    "generate_finops_query": "FinOps",
    # itsm-mcp on :8005, MCP_TOOL_PACKAGE=service_desk. Coexists with
    # mcp_kg's generate_itsm_query — both surface as "ITSM" in
    # the UI header. Add new entries here when widening the package.
    "create_incident": "ITSM",
    "update_incident": "ITSM",
    "add_comment": "ITSM",
    "resolve_incident": "ITSM",
    "list_incidents": "ITSM",
    "get_incident_by_number": "ITSM",
    "get_latest_incident": "ITSM",
    "get_user": "ITSM",
    "list_users": "ITSM",
    "get_article": "ITSM",
    "list_articles": "ITSM",
    # mcp-apm on :8004 — 9 smart-router tools that each dispatch
    # internally to many sub-operations. All surface as "APM" in the
    # UI header. mcp_var's calculate_*_throughput / *_availability_by_error_rate
    # / *_slo_* tools already map to "APM" via the platform field set by
    # the metric class itself; both paths coexist.
    "manage_applications": "APM",
    "manage_websites": "APM",
    "manage_mobile_apps": "APM",
    "manage_custom_dashboards": "APM",
    "analyze_infrastructure": "APM",
    "manage_automation": "APM",
    "manage_events": "APM",
    "manage_slo": "APM",
    "manage_releases": "APM",
    # mcp_dal topology tools (vendored from the mcp-apm fork).
    "get_topology": "APM",
    "get_related_hosts": "APM",
    "get_application_topology": "APM",
    # mcp_dal FinOps tools.
    "query_finops_cost_report": "FinOps",
    "query_finops_rightsizing": "FinOps",
    "query_finops_anomalies": "FinOps",
    "query_finops_billing_accounts": "FinOps",
    # mcp_dal Resilience tools.
    "query_resilience": "Resilience",
    "query_resilience_resilience": "Resilience",
    "query_resilience_security": "Resilience",
    "trigger_resilience_workflow": "Resilience",
    "query_resilience_workflow_status": "Resilience",
    # mcp_dal Infra.
    "query_infra_nl2iql": "Infra",
}


def synthesize(
    question: str,
    trace: list[TraceStep],
    elapsed_seconds: float,
) -> KgOutput:
    """Build the UI-facing KgOutput from a finished trace."""
    if not trace:
        return KgOutput(
            status="error",
            message="Agent produced no steps.",
            renderings=[Rendering(
                type="markdown",
                body="**Error:** agent produced no steps.",
            )],
            elapsed_seconds=elapsed_seconds,
        )

    actions = [s.action for s in trace if s.action]
    last_action = actions[-1] if actions else None
    final = trace[-1].final_answer

    chain = " → ".join(actions) if actions else "(no tool calls)"
    routing_reason = f"agent chain: {chain}"

    source = _resolve_source(last_action, trace)
    renderings = _collect_renderings(trace, final)
    raw_result = _build_raw_result(trace, final, elapsed_seconds)

    if final and not _is_error_final(final):
        return KgOutput(
            status="completed",
            message=f"Agent completed in {len(trace)} step(s).",
            tool_invoked=last_action,
            routing_reason=routing_reason,
            answer_markdown=final,
            renderings=renderings,
            raw_result=raw_result,
            graph=None,
            source=source,
            elapsed_seconds=elapsed_seconds,
        )

    # Error path — either the loop terminated with an error final_answer,
    # never reached one (max_steps), or the LLM/MCP/parse failed mid-loop.
    msg = (
        final
        or "Agent did not produce a final answer (max_steps reached or parse failure)."
    )
    return KgOutput(
        status="error",
        message=_strip_markdown_emphasis(msg)[:200],
        tool_invoked=last_action,
        routing_reason=routing_reason,
        answer_markdown=msg,
        renderings=renderings or [Rendering(type="markdown", body=msg)],
        raw_result=raw_result,
        graph=None,
        source=source,
        elapsed_seconds=elapsed_seconds,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_source(last_action: str | None, trace: list[TraceStep]) -> str:
    """Source label for the answer panel header. For generate_* tools it's a
    fixed map; for calculate_* we read the platform field from the result."""
    if not last_action:
        return "Agent"
    if last_action in _SOURCE_FOR:
        return _SOURCE_FOR[last_action]
    if last_action.startswith("calculate_"):
        for s in reversed(trace):
            if s.action == last_action and isinstance(s.tool_result, dict):
                platform = s.tool_result.get("platform")
                if platform:
                    return str(platform).title()
        return "ITSM"
    return "Agent"


def _collect_renderings(
    trace: list[TraceStep],
    final: str | None,
) -> list[Rendering]:
    """Build a flat list of renderings from each step's tool response.

    The leaf tools in mcp_kg return raw shapes today (no `renderings[]`
    field), so we project each step's `tool_result` ourselves via
    `build_renderings`. If a future mcp_kg version starts emitting
    renderings inline, we prefer those (richer than what we'd reconstruct).
    Append a trailing markdown rendering with the final answer.
    """
    out: list[Rendering] = []
    for step in trace:
        if not isinstance(step.tool_result, dict) or not step.action:
            continue
        # Prefer renderings the tool produced itself (forward-compat).
        inline = step.tool_result.get("renderings")
        if isinstance(inline, list) and inline:
            for r in inline:
                if isinstance(r, dict):
                    try:
                        out.append(Rendering(**r))
                    except Exception:
                        continue
            continue
        # Fall back to building them from the raw shape.
        out.extend(build_renderings(step.action, step.tool_result))

    if final:
        out.append(Rendering(type="markdown", body=final))
    return out


def _build_raw_result(
    trace: list[TraceStep],
    final: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Trace + final + meta. The UI shows this verbatim in the Raw Result
    accordion, so keep it informative but don't dump 50 KB of GraphQL rows
    here — those are already in the per-step renderings.

    Strip the bulky `tool_result.results.results[]` array from the trace
    dump (UI users can dig into per-step renderings if they want detail).
    """
    trace_out: list[dict[str, Any]] = []
    for s in trace:
        d = s.model_dump()
        # Slim the tool_result down; full version stays in memory only.
        tr = d.get("tool_result")
        if isinstance(tr, dict):
            slim = {k: v for k, v in tr.items() if k not in ("results", "raw_result")}
            slim["_results_summary"] = _summarise_results(tr.get("results"))
            d["tool_result"] = slim
        trace_out.append(d)

    return {
        "trace": trace_out,
        "final_answer": final,
        "steps_used": len(trace),
        "elapsed_seconds": elapsed_seconds,
    }


def _summarise_results(results: Any) -> dict[str, Any]:
    """One-line shape descriptor of a `results` payload for the raw_result
    dump. Avoids inlining full row arrays."""
    if results is None:
        return {"shape": "null"}
    if isinstance(results, list):
        return {"shape": "list", "length": len(results)}
    if isinstance(results, dict):
        out: dict[str, Any] = {"shape": "dict", "keys": list(results)[:10]}
        for k in ("results", "result"):
            v = results.get(k)
            if isinstance(v, list):
                out[f"{k}_length"] = len(v)
        return out
    return {"shape": type(results).__name__}


def _is_error_final(final: str) -> bool:
    """Detect when our own error formatting bubbled into final_answer."""
    return final.lstrip().lower().startswith("**error:**")


def _strip_markdown_emphasis(s: str) -> str:
    return s.replace("**", "").replace("`", "").replace("\n", " ").strip()
