"""Write workloads/itbench_questions.jsonl and check every gold query.

Questions are scoped to scenario 1 unless marked global, so their answers do not
change as more scenarios are loaded (the scale knob grows the haystack, not the
needle). Written before any LLM run; gold SQL is the answer key.
Usage: .venv/bin/python workloads/make_itbench_questions.py, then
workloads/add_questions_v2.py (this script rewrites the file with the first 24 only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

S = "scenario_id = 1"
CALLS = ("FROM span c JOIN span p ON c.parent_span_id = p.span_id "
         "AND c.trace_id = p.trace_id AND c.scenario_id = p.scenario_id "
         "WHERE p.service <> c.service AND c.scenario_id = 1")
Q = [
    # lookup
    ("IT-L1", "lookup", "const", False, "In scenario 1, on which node does pod checkout-5dccddf8bb-vh65b run?",
     f"SELECT node_name FROM k8s_object WHERE kind = 'Pod' AND name = 'checkout-5dccddf8bb-vh65b' AND {S}",
     "pod_placement", {"scenario": 1, "pod": "checkout-5dccddf8bb-vh65b"}),
    ("IT-L2", "lookup", "const", False, "In scenario 1, which Deployment owns pod cart-755465879b-j4pxv (through its ReplicaSet)?",
     f"SELECT d.name FROM k8s_object p JOIN k8s_object rs ON p.owner_uid = rs.uid JOIN k8s_object d ON rs.owner_uid = d.uid WHERE p.name = 'cart-755465879b-j4pxv' AND p.{S}",
     None, {}),
    ("IT-L3", "lookup", "const", False, "What fault was injected in scenario 1: which entity, of what kind, and in which fault category?",
     f"SELECT fault_entity, fault_kind, fault_category FROM scenario WHERE {S}", None, {}),
    # filter
    ("IT-F1", "filter", "small", False, "In scenario 1, which pods in namespace otel-demo run on node i-0386fc27c3538ac87?",
     f"SELECT name FROM k8s_object WHERE kind = 'Pod' AND namespace = 'otel-demo' AND node_name = 'i-0386fc27c3538ac87' AND {S}",
     "pods_on_node", {"scenario": 1, "node": "i-0386fc27c3538ac87"}),
    ("IT-F2", "filter", "small", False, "In scenario 1, list the Warning-type Kubernetes events with their reason, object kind and object name.",
     f"SELECT reason, object_kind, object_name FROM k8s_event WHERE type = 'Warning' AND {S}", None, {}),
    ("IT-F3", "filter", "small", False, "In scenario 1, which pods were deleted before the end of the observation window (no longer listed at the end)?",
     f"SELECT name FROM k8s_object WHERE kind = 'Pod' AND deleted AND {S}", None, {}),
    # aggregate
    ("IT-A1", "aggregate", "small", False, "In scenario 1, how many pods run on each node? Give node name and pod count.",
     f"SELECT node_name, count(*) FROM k8s_object WHERE kind = 'Pod' AND {S} GROUP BY 1", None, {}),
    ("IT-A2", "aggregate", "small", False, "In scenario 1, for each service with at least one error span, how many spans had status Error?",
     f"SELECT service, count(*) FROM span WHERE status = 'Error' AND {S} GROUP BY 1", None, {}),
    ("IT-A3", "aggregate", "small", False, "In scenario 1, what is the average duration in milliseconds of Server-kind spans for each service?",
     f"SELECT service, avg(duration_ms) FROM span WHERE kind = 'Server' AND {S} GROUP BY 1", None, {}),
    ("IT-A4", "aggregate", "small", False, "Across all scenarios in the data, how many pods does each scenario have? Give scenario id and pod count.",
     "SELECT scenario_id, count(*) FROM k8s_object WHERE kind = 'Pod' GROUP BY 1", None, {}),
    # join / multi-hop topology
    ("IT-J1", "join", "small", False, "In scenario 1, which services does the frontend service call directly, according to the traces?",
     f"SELECT DISTINCT c.service {CALLS} AND p.service = 'frontend'",
     "service_callees", {"scenario": 1, "service": "frontend"}),
    ("IT-J2", "join", "small", False, "In scenario 1, which services call the cart service directly, and on which node does each calling pod run? Give caller service and node name.",
     f"SELECT DISTINCT p.service, o.node_name {CALLS} AND c.service = 'cart' JOIN k8s_object o ON p.pod_uid = o.uid".replace("WHERE p.service <> c.service AND c.scenario_id = 1 AND c.service = 'cart' JOIN k8s_object o ON p.pod_uid = o.uid", "JOIN k8s_object o ON p.pod_uid = o.uid WHERE p.service <> c.service AND c.scenario_id = 1 AND c.service = 'cart'"),
     None, {}),
    ("IT-J3", "join", "small", False, "In scenario 1, for each alert raised on a specific pod, give the alert name, the pod name and the node that pod runs on.",
     f"SELECT a.alertname, a.pod, o.node_name FROM alert a JOIN k8s_object o ON a.pod_uid = o.uid WHERE a.{S}", None, {}),
    ("IT-J4", "join", "small", False, "In scenario 1, which Deployments in namespace otel-demo have at least one pod on node i-0386fc27c3538ac87?",
     f"SELECT DISTINCT d.name FROM k8s_object p JOIN k8s_object rs ON p.owner_uid = rs.uid JOIN k8s_object d ON rs.owner_uid = d.uid WHERE p.kind = 'Pod' AND p.namespace = 'otel-demo' AND p.node_name = 'i-0386fc27c3538ac87' AND d.kind = 'Deployment' AND p.{S}",
     None, {}),
    ("IT-J5", "join", "small", False, "In scenario 1, how many Kubernetes events were recorded about pods of the load-generator Deployment (through their ReplicaSets)?",
     f"SELECT count(*) FROM k8s_event e JOIN k8s_object p ON e.object_uid = p.uid JOIN k8s_object rs ON p.owner_uid = rs.uid JOIN k8s_object d ON rs.owner_uid = d.uid WHERE d.name = 'load-generator' AND e.{S}",
     None, {}),
    # top-k
    ("IT-T1", "topk", "small", True, "In scenario 1, which 5 (service, operation) pairs of Server-kind spans have the highest average duration in milliseconds? Give service, operation and average duration, highest first.",
     f"SELECT service, operation, avg(duration_ms) AS d FROM span WHERE kind = 'Server' AND {S} GROUP BY 1, 2 ORDER BY d DESC LIMIT 5", None, {}),
    ("IT-T2", "topk", "small", True, "In scenario 1, which 3 services produced the most log records? Give service and count, most first.",
     f"SELECT service, count(*) AS n FROM log WHERE {S} GROUP BY 1 ORDER BY n DESC LIMIT 3", None, {}),
    ("IT-T3", "topk", "small", True, "In scenario 1, which 3 pairs of different services (caller, callee) have the most direct calls in the traces? Give caller, callee and number of calls, most first.",
     f"SELECT p.service, c.service, count(*) AS n {CALLS} GROUP BY 1, 2 ORDER BY n DESC LIMIT 3", None, {}),
    # large result
    ("IT-G1", "large", "large", False, "In scenario 1, list every error span of the checkout service with its timestamp, operation and duration in milliseconds.",
     f"SELECT ts, operation, duration_ms FROM span WHERE service = 'checkout' AND status = 'Error' AND {S}",
     "service_error_spans", {"scenario": 1, "service": "checkout"}),
    ("IT-G2", "large", "large", False, "In scenario 1, list all Kubernetes events with timestamp, reason and the name of the object they are about.",
     f"SELECT ts, reason, object_name FROM k8s_event WHERE {S}", None, {}),
    ("IT-G3", "large", "large", False, "In scenario 1, list all log records of the payment service with their timestamp and body.",
     f"SELECT ts, body FROM log WHERE service = 'payment' AND {S}", None, {}),
    # time window
    ("IT-W1", "window", "const", False, "In scenario 1, how many error spans did frontend-proxy produce from 17:20:00 (inclusive) to 17:25:00 (exclusive) UTC on 2025-12-15?",
     f"SELECT count(*) FROM span WHERE service = 'frontend-proxy' AND status = 'Error' AND ts >= TIMESTAMP '2025-12-15 17:20:00' AND ts < TIMESTAMP '2025-12-15 17:25:00' AND {S}",
     None, {}),
    ("IT-W2", "window", "small", False, "In scenario 1, which alerts (alert name and pod, if any) were first seen at or after 17:35 UTC on 2025-12-15?",
     f"SELECT alertname, pod FROM alert WHERE first_seen >= TIMESTAMP '2025-12-15 17:35:00' AND {S}", None, {}),
    ("IT-W3", "window", "small", False, "In scenario 1, which Kubernetes event reasons occurred about the load-generator pods between 17:45 and 17:55 UTC on 2025-12-15, and how many times each?",
     f"SELECT reason, count(*) FROM k8s_event WHERE object_name LIKE 'load-generator-%' AND ts >= TIMESTAMP '2025-12-15 17:45:00' AND ts < TIMESTAMP '2025-12-15 17:55:00' AND {S} GROUP BY 1",
     None, {}),
]

WORKFLOWS = {
    "pod_placement": {"description": "Node a pod runs on, in a scenario.", "params": ["scenario", "pod"],
                      "sql": "SELECT node_name FROM k8s_object WHERE kind = 'Pod' AND name = $pod AND scenario_id = $scenario::INTEGER"},
    "pods_on_node": {"description": "Pods in namespace otel-demo on a node, in a scenario.", "params": ["scenario", "node"],
                     "sql": "SELECT name FROM k8s_object WHERE kind = 'Pod' AND namespace = 'otel-demo' AND node_name = $node AND scenario_id = $scenario::INTEGER"},
    "service_callees": {"description": "Services a service calls directly (from traces), in a scenario.", "params": ["scenario", "service"],
                        "sql": "SELECT DISTINCT c.service FROM span c JOIN span p ON c.parent_span_id = p.span_id AND c.trace_id = p.trace_id AND c.scenario_id = p.scenario_id WHERE p.service <> c.service AND c.scenario_id = $scenario::INTEGER AND p.service = $service"},
    "service_error_spans": {"description": "Error spans of a service (timestamp, operation, duration ms), in a scenario.", "params": ["scenario", "service"],
                            "sql": "SELECT ts, operation, duration_ms FROM span WHERE service = $service AND status = 'Error' AND scenario_id = $scenario::INTEGER"},
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    con = duckdb.connect(str(root / "data" / "itbench_s1.duckdb"), read_only=True)
    out = []
    for qid, fam, size, ordered, text, sql, wf, params in Q:
        rows = con.execute(sql).fetchall()
        if not rows:
            sys.exit(f"{qid}: empty answer")
        if ordered:  # no tie at the cut-off, or the gold answer is ambiguous
            inner = sql.rsplit(" LIMIT ", 1)
            k = int(inner[1])
            nxt = con.execute(inner[0] + f" LIMIT {k + 1}").fetchall()
            if len(nxt) > k and nxt[k][-1] == nxt[k - 1][-1]:
                sys.exit(f"{qid}: tie at the top-{k} boundary")
        if wf:
            wsql = WORKFLOWS[wf]["sql"]
            got = con.execute(wsql, params).fetchall()
            if sorted(map(str, got)) != sorted(map(str, rows)):
                sys.exit(f"{qid}: workflow {wf} disagrees with gold")
        print(f"{qid:6} {fam:9} rows={len(rows):5}  {str(rows[:2])[:90]}")
        out.append({"id": qid, "family": fam, "size_class": size, "ordered": ordered,
                    "question": text, "gold_sql": sql, "plan": [], "workflow": wf,
                    "workflow_params": params})
    (root / "workloads" / "itbench_questions.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in out))
    (root / "workloads" / "itbench_workflows.json").write_text(
        json.dumps(WORKFLOWS, indent=1) + "\n")
    print(f"wrote {len(out)} questions, {len(WORKFLOWS)} workflows")


if __name__ == "__main__":
    main()
