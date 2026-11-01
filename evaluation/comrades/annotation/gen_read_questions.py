"""Generate single-hop data-retrieval questions from the mcp_dal READ tools.

Unlike gen_candidates.py (raw tool-description prompts, all 105 tools, unverified),
this emits *curated* natural-language prompts for the 51 read-only tools only, each
correctly family-tagged and with its gold tool known by construction. Two phrasings
per tool (a direct imperative + a tool-using frame) give natural variety without
hand-writing every variant.

Outputs (in evaluation/comrades/):
  - requests_read.jsonl   {id, family, prompt, gold_tool, source}   (DAL-agent workload)
  - harness/tasks_read.jsonl  {id, family, hops, products, prompt, gold_tools, source}

Usage:  python annotation/gen_read_questions.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMRADES = HERE.parent

# tool -> (family, canonical natural-language ask). One curated prompt per READ tool,
# grounded in the tool's purpose. Families are the 5 backend products.
READ_QUESTIONS: dict[str, tuple[str, str]] = {
    # ---- cost (FinOps) ----
    "query_finops_cost_report": ("cost", "What are the top cloud costs this month?"),
    "query_finops_rightsizing": ("cost", "Show the AWS EC2 rightsizing recommendations."),
    "query_finops_anomalies": ("cost", "Are there any cloud cost anomalies in the last 7 days?"),
    "query_finops_billing_accounts": ("cost", "List the cloud billing accounts."),
    # ---- apm (APM) ----
    "get_application_topology": ("apm", "Show the APM service topology for our applications."),
    "get_related_hosts": ("apm", "Which hosts are related to this APM snapshot?"),
    "get_topology": ("apm", "Show the infrastructure topology from APM."),
    "manage_applications": ("apm", "List the APM application perspectives."),
    "manage_automation": ("apm", "List the available APM automation actions."),
    "manage_custom_dashboards": ("apm", "List the APM custom dashboards."),
    "manage_events": ("apm", "Show the recent APM events."),
    "manage_mobile_apps": ("apm", "List the monitored APM mobile apps."),
    "manage_releases": ("apm", "List the tracked releases and their deployment status."),
    "manage_slo": ("apm", "Show the configured SLOs and their current status."),
    "manage_websites": ("apm", "List the monitored APM websites."),
    # ---- itsm (ITSM) ----
    "list_incidents": ("itsm", "List the recent ITSM incidents."),
    "get_incident_by_number": ("itsm", "Show the details of ITSM incident INC0010001."),
    "get_latest_incident": ("itsm", "What is the most recent ITSM incident?"),
    "list_catalog_items": ("itsm", "List the ITSM service catalog items."),
    "get_catalog_item": ("itsm", "Show the details of a specific service catalog item."),
    "list_catalog_categories": ("itsm", "List the service catalog categories."),
    "list_catalog_item_variables": ("itsm", "List the variables for a service catalog item."),
    "list_change_requests": ("itsm", "List the open ITSM change requests."),
    "get_change_request_details": ("itsm", "Show the details of change request CHG0030001."),
    "list_workflows": ("itsm", "List the ITSM workflows."),
    "get_workflow_details": ("itsm", "Show the details of a specific ITSM workflow."),
    "list_workflow_versions": ("itsm", "List the versions of a ITSM workflow."),
    "get_workflow_activities": ("itsm", "Show the activities for a ITSM workflow."),
    "list_changesets": ("itsm", "List the ITSM changesets."),
    "get_changeset_details": ("itsm", "Show the details of a specific ITSM changeset."),
    "list_script_includes": ("itsm", "List the ITSM script includes."),
    "get_script_include": ("itsm", "Show a specific ITSM script include."),
    "list_knowledge_bases": ("itsm", "List the ITSM knowledge bases."),
    "list_categories": ("itsm", "List the categories in a ITSM knowledge base."),
    "list_articles": ("itsm", "List the ITSM knowledge articles."),
    "get_article": ("itsm", "Show a specific ITSM knowledge article."),
    "get_user": ("itsm", "Show the details of a specific ITSM user."),
    "list_users": ("itsm", "List the ITSM users."),
    "list_groups": ("itsm", "List the ITSM groups."),
    "list_stories": ("itsm", "List the ITSM agile stories."),
    "list_story_dependencies": ("itsm", "List the dependencies between ITSM stories."),
    "list_epics": ("itsm", "List the ITSM epics."),
    "list_scrum_tasks": ("itsm", "List the ITSM scrum tasks."),
    "list_projects": ("itsm", "List the ITSM projects."),
    # ---- infra (Infra) ----
    "analyze_infrastructure": ("infra", "Analyze our infrastructure and summarize the key findings."),
    "get_optimization_recommendations": ("infra", "What optimization recommendations are there for the service catalog?"),
    "query_infra_nl2iql": ("infra", "Generate an IQL query to list all AWS EC2 instances."),
    # ---- resilience (Resilience) ----
    "query_resilience": ("resilience", "Show the Resilience resilience and compliance overview for our applications."),
    "query_resilience_resilience": ("resilience", "List the Resilience resilience assessments."),
    "query_resilience_security": ("resilience", "Show the Resilience security risk analysis."),
    "query_resilience_workflow_status": ("resilience", "What is the status of a Resilience workflow execution?"),
}


def reframe(canonical: str) -> str:
    """Second phrasing: a natural tool-using frame over the canonical ask."""
    s = canonical[0].lower() + canonical[1:]
    return f"Using the available data tools, {s}"


def main() -> None:
    catalog = {t["name"] for t in json.load(open(HERE / "catalog_mcp_dal.json"))}
    missing = [t for t in READ_QUESTIONS if t not in catalog]
    assert not missing, f"unknown tools: {missing}"

    req_rows, task_rows = [], []
    for tool, (family, canonical) in READ_QUESTIONS.items():
        for i, prompt in enumerate((canonical, reframe(canonical)), start=1):
            rid = f"tr-{tool}-{i}"
            req_rows.append({"id": rid, "family": family, "prompt": prompt,
                             "gold_tool": tool, "source": "templated-read"})
            task_rows.append({"id": rid, "family": family, "hops": 1,
                              "products": [family], "prompt": prompt,
                              "gold_tools": [tool], "source": "templated-read"})

    with open(COMRADES / "requests_read.jsonl", "w") as f:
        for r in req_rows:
            f.write(json.dumps(r) + "\n")
    with open(COMRADES / "harness" / "tasks_read.jsonl", "w") as f:
        for r in task_rows:
            f.write(json.dumps(r) + "\n")

    import collections
    fam = collections.Counter(r["family"] for r in req_rows)
    print(f"wrote {len(req_rows)} questions over {len(READ_QUESTIONS)} read tools")
    print("per-family:", dict(fam))


if __name__ == "__main__":
    main()
