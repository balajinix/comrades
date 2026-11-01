"""Generate balanced single-hop questions over the EXPANDED mcp_dal read surface (151 tools).

Supersedes gen_read_questions.py (which targeted the 105-tool surface). The dev server
(microservices2agents:8002) added 46 tools, filling the thin backends (infra 3->13,
resilience 4->20). This emits one curated natural-language question per READ tool, so the
question set is balanced by the now-balanced tool surface. Backend is read from each
tool's "[Backend]" title tag in catalog_mcp_dal_v151.json.

Outputs (in evaluation/comrades/):
  - requests_read_v151.jsonl       {id, family, prompt, gold_tool, source}
  - harness/tasks_read_v151.jsonl  {id, family, hops, products, prompt, gold_tools, source}

Usage:  python annotation/gen_read_questions_v151.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMRADES = HERE.parent
TAG2FAM = {"apm": "apm", "infra": "infra", "resilience": "resilience",
           "finops": "cost", "itsm": "itsm"}
WRITE_PREFIX = ("create_", "update_", "delete_", "add_", "remove_", "resolve_", "approve_",
                "reject_", "submit_", "commit_", "publish_", "trigger_", "activate_",
                "deactivate_", "reorder_", "move_", "set_")

# Curated NL prompts. Keys are tool names; value is the question. Any read tool not
# listed falls back to a title-derived prompt. (execute_infra_iql is included as a
# read-style retrieval despite the execute_ name -- it returns query results.)
CURATED: dict[str, str] = {
    # cost
    "query_finops_cost_report": "What are the top cloud costs this month?",
    "query_finops_rightsizing": "Show the AWS EC2 rightsizing recommendations.",
    "query_finops_anomalies": "Are there any cloud cost anomalies in the last 7 days?",
    "query_finops_billing_accounts": "List the cloud billing accounts.",
    "list_finops_cost_measures": "List the available FinOps cost measures.",
    "list_finops_vendors": "List the cloud vendors known to the account.",
    "get_finops_cost_estimate": "What is the projected cloud spend?",
    # apm / apm
    "get_application_topology": "Show the APM service topology for our applications.",
    "get_related_hosts": "Which hosts are related to this APM snapshot?",
    "get_topology": "Show the infrastructure topology from APM.",
    "list_applications": "List the APM applications.",
    "get_application_metrics": "Show the metrics for a specific APM application.",
    "get_application_alerts": "List the active alerts for an APM application.",
    "list_slos": "List the configured SLOs in APM.",
    "get_slo_report": "Show the SLO report for a specific SLO.",
    "get_slo_status": "What is the current status of a specific SLO?",
    "list_events": "List the recent APM events.",
    "get_event": "Show the details of a specific APM event.",
    "list_websites": "List the monitored APM websites.",
    "get_website": "Show the details of a specific APM website.",
    "list_mobile_apps": "List the monitored APM mobile apps.",
    "get_mobile_app": "Show the details of a specific APM mobile app.",
    "list_releases": "List the tracked APM releases.",
    "get_release_impact": "Show the impact analysis for a specific APM release.",
    "list_apm_application_service_endpoints": "List the endpoints for a service in an APM application.",
    # infra / infra
    "execute_infra_iql": "Run an IQL query against the infrastructure graph and return the results.",
    "query_infra_nl2iql": "Generate an IQL query to list all AWS EC2 instances.",
    "list_infra_node_schemas": "What node (entity) types exist in the infrastructure graph?",
    "list_infra_edge_schemas": "What relationship types exist in the infrastructure graph?",
    "get_infra_node": "Show the details of a specific infrastructure graph node.",
    "list_infra_node_inventory": "How many nodes of each type are in the infrastructure graph?",
    "list_infra_edge_inventory": "How many relationships of each type are in the infrastructure graph?",
    "list_infra_connections": "List the data connections feeding the infrastructure graph.",
    "get_infra_connection": "Show the details of a specific infrastructure graph data connection.",
    "get_infra_graph_sync_status": "What is the sync status of the infrastructure graph?",
    "list_infra_saved_queries": "List the saved infrastructure graph queries.",
    "get_infra_one_hop_neighbors": "What are the one-hop neighbors of a given infrastructure node?",
    "get_infra_node_by_resource_id": "Find the infrastructure node for a given business resource ID.",
    # resilience / resilience
    "query_resilience": "Show the Resilience resilience and compliance overview for our applications.",
    "query_resilience_resilience": "List the Resilience resilience assessments.",
    "query_resilience_security": "Show the Resilience security risk analysis.",
    "query_resilience_workflow_status": "What is the status of a Resilience workflow execution?",
    "list_resilience_applications": "List the applications tracked in Resilience.",
    "get_resilience_application": "Show the details of a specific Resilience application.",
    "list_resilience_workflows": "List the Resilience workflows.",
    "list_resilience_compliance_risks": "List the Resilience compliance risks.",
    "list_resilience_policy_violations": "List the Resilience policy violations.",
    "get_resilience_resilience_metrics": "Show the resilience metrics for a Resilience application.",
    "list_resilience_resilience_assessments": "List the Resilience resilience assessments.",
    "get_resilience_resilience_posture": "What is the resilience posture in Resilience?",
    "get_resilience_cve_details": "Show the details of a specific CVE in Resilience.",
    "get_resilience_risk_details": "Show the risk details for a Resilience finding.",
    "get_resilience_application_topology": "Show the application topology in Resilience.",
    "get_resilience_workflow": "Show the details of a specific Resilience workflow.",
    "list_resilience_workflow_executions": "List the Resilience workflow executions.",
    "get_resilience_compliance_status": "What is the compliance status in Resilience?",
    "get_resilience_slo_status": "Show the SLO status for a Resilience application.",
    "get_resilience_risk_status": "What is the risk status for a given context in Resilience?",
    # itsm reads keep their curated prompts from the prior set
    "list_incidents": "List the recent ITSM incidents.",
    "get_incident_by_number": "Show the details of ITSM incident INC0010001.",
    "get_latest_incident": "What is the most recent ITSM incident?",
    "list_catalog_items": "List the ITSM service catalog items.",
    "get_catalog_item": "Show the details of a specific service catalog item.",
    "list_catalog_categories": "List the service catalog categories.",
    "list_catalog_item_variables": "List the variables for a service catalog item.",
    "list_change_requests": "List the open ITSM change requests.",
    "get_change_request_details": "Show the details of change request CHG0030001.",
    "list_workflows": "List the ITSM workflows.",
    "get_workflow_details": "Show the details of a specific ITSM workflow.",
    "list_workflow_versions": "List the versions of a ITSM workflow.",
    "get_workflow_activities": "Show the activities for a ITSM workflow.",
    "list_changesets": "List the ITSM changesets.",
    "get_changeset_details": "Show the details of a specific ITSM changeset.",
    "list_script_includes": "List the ITSM script includes.",
    "get_script_include": "Show a specific ITSM script include.",
    "list_knowledge_bases": "List the ITSM knowledge bases.",
    "list_categories": "List the categories in a ITSM knowledge base.",
    "list_articles": "List the ITSM knowledge articles.",
    "get_article": "Show a specific ITSM knowledge article.",
    "get_user": "Show the details of a specific ITSM user.",
    "list_users": "List the ITSM users.",
    "list_groups": "List the ITSM groups.",
    "list_stories": "List the ITSM agile stories.",
    "list_story_dependencies": "List the dependencies between ITSM stories.",
    "list_epics": "List the ITSM epics.",
    "list_scrum_tasks": "List the ITSM scrum tasks.",
    "list_projects": "List the ITSM projects.",
}


def backend(t: dict) -> str:
    m = re.match(r"\[([^\]]+)\]", t.get("title", ""))
    return TAG2FAM.get(m.group(1).lower() if m else "", "itsm")


def title_prompt(t: dict) -> str:
    """Fallback: derive a question from the tool's title."""
    ttl = re.sub(r"^\[[^\]]+\]\s*", "", t.get("title", "")).strip()
    ttl = re.sub(r"\s*\([^)]*\)$", "", ttl)  # drop trailing parenthetical
    return f"Using the available data tools, {ttl[0].lower() + ttl[1:]}." if ttl else t["name"]


def is_read(name: str) -> bool:
    # execute_infra_iql is a retrieval despite the prefix
    return name == "execute_infra_iql" or not name.startswith(WRITE_PREFIX)


def main() -> None:
    cat = json.load(open(HERE / "catalog_mcp_dal_v151.json"))
    req_rows, task_rows = [], []
    for t in cat:
        name = t["name"]
        if not is_read(name):
            continue
        fam = backend(t)
        prompt = CURATED.get(name) or title_prompt(t)
        rid = f"v151-{name}"
        req_rows.append({"id": rid, "family": fam, "prompt": prompt,
                         "gold_tool": name, "source": "templated-read-v151"})
        task_rows.append({"id": rid, "family": fam, "hops": 1, "products": [fam],
                          "prompt": prompt, "gold_tools": [name],
                          "source": "templated-read-v151"})

    with open(COMRADES / "requests_read_v151.jsonl", "w") as f:
        for r in req_rows:
            f.write(json.dumps(r) + "\n")
    with open(COMRADES / "harness" / "tasks_read_v151.jsonl", "w") as f:
        for r in task_rows:
            f.write(json.dumps(r) + "\n")

    import collections
    fam = collections.Counter(r["family"] for r in req_rows)
    curated = sum(1 for t in cat if is_read(t["name"]) and t["name"] in CURATED)
    print(f"wrote {len(req_rows)} questions ({curated} curated, {len(req_rows)-curated} title-derived)")
    print("per-family:", dict(fam))


if __name__ == "__main__":
    main()
