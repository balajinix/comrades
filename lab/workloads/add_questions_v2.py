"""Add the v2 questions (2026-10-21) to the TPC-H, TPC-DS and ITBench workloads.

Takes the workload to about 100 questions: TPC-H 24 -> 30, TPC-DS 24 -> 35,
ITBench 24 -> 35, weighted towards large results, multi-hop joins and time windows.
Every gold query must be non-empty at both scales, and top-k questions must have no
tie at the cut-off. New questions carry no scripted ``plan`` (LLM runs only).
Idempotent: questions already in a file are skipped.
Usage: .venv/bin/python workloads/add_questions_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dal_lab import itbench, tpcds, tpch  # noqa: E402

# (id, family, size_class, ordered, question, gold_sql)
TPCH = [
    ("L5", "lookup", "const", False, "What are the name and phone number of supplier 10?",
     "SELECT s_name, s_phone FROM supplier WHERE s_suppkey = 10"),
    ("A5", "aggregation", "const-out", False,
     "How many returned line items (return flag 'R') were shipped by each ship mode?",
     "SELECT l_shipmode, count(*) FROM lineitem WHERE l_returnflag = 'R' GROUP BY 1"),
    ("J5", "join", "const-out", False,
     "Which suppliers supply part 100, and in which region is each located? Give supplier "
     "name and region name.",
     "SELECT s_name, r_name FROM partsupp JOIN supplier ON ps_suppkey = s_suppkey "
     "JOIN nation ON s_nationkey = n_nationkey JOIN region ON n_regionkey = r_regionkey "
     "WHERE ps_partkey = 100"),
    ("T5", "topk", "const-out", True,
     "Which 5 suppliers hold the most stock value (supply cost times available quantity, "
     "summed over their parts)? Give supplier name and value, highest first.",
     "SELECT s_name, sum(ps_supplycost * ps_availqty) AS v FROM partsupp JOIN supplier "
     "ON ps_suppkey = s_suppkey GROUP BY s_suppkey, s_name ORDER BY v DESC LIMIT 5"),
    ("G5", "large_list", "linear", False,
     "List all orders with priority '1-URGENT' placed in March 1997: order key, customer "
     "key and total price.",
     "SELECT o_orderkey, o_custkey, o_totalprice FROM orders WHERE o_orderpriority = "
     "'1-URGENT' AND o_orderdate >= DATE '1997-03-01' AND o_orderdate < DATE '1997-04-01'"),
    ("G6", "large_list", "linear-small", False,
     "List the line items of all orders placed by customer 7: order key, line number, "
     "part key and quantity.",
     "SELECT l_orderkey, l_linenumber, l_partkey, l_quantity FROM lineitem JOIN orders "
     "ON l_orderkey = o_orderkey WHERE o_custkey = 7"),
]

TPCDS = [
    ("DS-L5", "lookup", "const", False,
     "What are the name, city and number of employees of store 1 (store_sk 1)?",
     "SELECT s_store_name, s_city, s_number_employees FROM store WHERE s_store_sk = 1"),
    ("DS-L6", "lookup", "const", False,
     "What are the name, class and manager of call center 1 (call_center_sk 1)?",
     "SELECT cc_name, cc_class, cc_manager FROM call_center WHERE cc_call_center_sk = 1"),
    ("DS-F5", "filtered_list", "const-out", False,
     "List the promotions advertised by direct mail: promotion id and promotion name.",
     "SELECT p_promo_id, p_promo_name FROM promotion WHERE p_channel_dmail = 'Y'"),
    ("DS-F6", "filtered_list", "const-out", False,
     "List the warehouses with their name, city and state.",
     "SELECT w_warehouse_name, w_city, w_state FROM warehouse"),
    ("DS-A5", "aggregation", "const-out", False,
     "How many store sale lines were sold in each month of 2001? Give month number and count.",
     "SELECT d_moy, count(*) FROM store_sales JOIN date_dim ON ss_sold_date_sk = d_date_sk "
     "WHERE d_year = 2001 GROUP BY 1"),
    ("DS-A6", "aggregation", "const-out", False,
     "In the inventory snapshot of 2002-12-26, how many units were on hand in each "
     "warehouse in total? Give warehouse name and units.",
     "SELECT w_warehouse_name, sum(inv_quantity_on_hand) FROM inventory JOIN warehouse "
     "ON inv_warehouse_sk = w_warehouse_sk JOIN date_dim ON inv_date_sk = d_date_sk "
     "WHERE d_date = DATE '2002-12-26' GROUP BY w_warehouse_sk, w_warehouse_name"),
    ("DS-T5", "topk", "const-out", True,
     "Which 3 ship mode codes (ship mode id) carried the most web sales revenue (net "
     "paid) in 2001, and how much each? Most first.",
     "SELECT sm_ship_mode_id, sum(ws_net_paid) AS n FROM web_sales JOIN ship_mode "
     "ON ws_ship_mode_sk = sm_ship_mode_sk JOIN date_dim ON ws_sold_date_sk = d_date_sk "
     "WHERE d_year = 2001 GROUP BY 1 ORDER BY n DESC LIMIT 3"),
    ("DS-T6", "topk", "const-out", True,
     "Which 3 items (item_sk) earned the highest store sales net profit in 2001, and "
     "what was each one's net profit? Highest first.",
     "SELECT ss_item_sk, sum(ss_net_profit) AS p FROM store_sales "
     "JOIN date_dim ON ss_sold_date_sk = d_date_sk "
     "WHERE d_year = 2001 GROUP BY 1 ORDER BY p DESC LIMIT 3"),
    ("DS-J5", "join", "const", False,
     "What was the store sales revenue (net paid) in 2002 from items in the 'Books' "
     "category sold in stores located in Tennessee (TN)?",
     "SELECT sum(ss_net_paid) FROM store_sales JOIN item ON ss_item_sk = i_item_sk "
     "JOIN store ON ss_store_sk = s_store_sk JOIN date_dim ON ss_sold_date_sk = d_date_sk "
     "WHERE i_category = 'Books' AND s_state = 'TN' AND d_year = 2002"),
    ("DS-J6", "join", "const-out", False,
     "How many web sale lines in 2000 went to customers whose current address is in "
     "California (CA), per web site? Give web site name and count.",
     "SELECT web_name, count(*) FROM web_sales JOIN customer ON ws_bill_customer_sk = "
     "c_customer_sk JOIN customer_address ON c_current_addr_sk = ca_address_sk "
     "JOIN web_site ON ws_web_site_sk = web_site_sk JOIN date_dim ON ws_sold_date_sk = "
     "d_date_sk WHERE ca_state = 'CA' AND d_year = 2000 GROUP BY 1"),
    ("DS-G5", "large_list", "linear", False,
     "List every web sale line shipped on 2001-11-15 (by ship date): order number, item "
     "and quantity.",
     "SELECT ws_order_number, ws_item_sk, ws_quantity FROM web_sales JOIN date_dim "
     "ON ws_ship_date_sk = d_date_sk WHERE d_date = DATE '2001-11-15'"),
]

S = "scenario_id = 1"
CALLS = ("FROM span c JOIN span p ON c.parent_span_id = p.span_id "
         "AND c.trace_id = p.trace_id AND c.scenario_id = p.scenario_id "
         "WHERE p.service <> c.service AND c.scenario_id = 1")
OWN = ("FROM k8s_object p JOIN k8s_object rs ON p.owner_uid = rs.uid "
       "JOIN k8s_object d ON rs.owner_uid = d.uid")
ITB = [
    ("IT-L4", "lookup", "const", False,
     "In scenario 1, when was pod frontend-675fd7b5c5-jlpzp created?",
     f"SELECT created_at FROM k8s_object WHERE kind = 'Pod' AND name = "
     f"'frontend-675fd7b5c5-jlpzp' AND {S}"),
    ("IT-L5", "lookup", "const", False,
     "In scenario 1, which ReplicaSet owns pod payment-55766d76b9-mgn4m?",
     f"SELECT rs.name {OWN.split(' JOIN k8s_object d')[0]} WHERE p.name = "
     f"'payment-55766d76b9-mgn4m' AND p.{S}"),
    ("IT-F4", "filter", "small", False,
     "In scenario 1, which Kubernetes Services exist in namespace otel-demo?",
     f"SELECT name FROM k8s_object WHERE kind = 'Service' AND namespace = 'otel-demo' "
     f"AND {S}"),
    ("IT-A5", "aggregate", "small", False,
     "In scenario 1, how many log records did each pod of the cart Deployment emit? Give "
     "pod name and count.",
     f"SELECT p.name, count(*) FROM log l JOIN k8s_object p ON l.pod_uid = p.uid "
     f"JOIN k8s_object rs ON p.owner_uid = rs.uid JOIN k8s_object d ON rs.owner_uid = d.uid "
     f"WHERE d.name = 'cart' AND d.kind = 'Deployment' AND l.{S} GROUP BY 1"),
    ("IT-A6", "aggregate", "const", False,
     "In scenario 1, what fraction of the frontend-proxy service's spans had status Error?",
     f"SELECT avg(CASE WHEN status = 'Error' THEN 1.0 ELSE 0.0 END) FROM span "
     f"WHERE service = 'frontend-proxy' AND {S}"),
    ("IT-J6", "join", "small", False,
     "In scenario 1, on which nodes do the pods of the services that checkout calls "
     "directly run? Give callee service and node name.",
     f"SELECT DISTINCT c.service, o.node_name FROM span c JOIN span p "
     f"ON c.parent_span_id = p.span_id AND c.trace_id = p.trace_id "
     f"AND c.scenario_id = p.scenario_id JOIN k8s_object o ON c.pod_uid = o.uid "
     f"WHERE p.service <> c.service AND c.scenario_id = 1 AND p.service = 'checkout'"),
    ("IT-J7", "join", "small", False,
     "In scenario 1, how many ReplicaSets does each Deployment in namespace otel-demo "
     "own? Give deployment name and count.",
     f"SELECT d.name, count(*) FROM k8s_object rs JOIN k8s_object d ON rs.owner_uid = d.uid "
     f"WHERE rs.kind = 'ReplicaSet' AND d.kind = 'Deployment' AND d.namespace = 'otel-demo' "
     f"AND rs.{S} GROUP BY 1"),
    ("IT-T4", "topk", "small", True,
     "In scenario 1, which 3 pods emitted the most error spans? Give pod name and count, "
     "most first.",
     f"SELECT o.name, count(*) AS n FROM span s JOIN k8s_object o ON s.pod_uid = o.uid "
     f"WHERE s.status = 'Error' AND s.{S} GROUP BY 1 ORDER BY n DESC LIMIT 3"),
    ("IT-G4", "large", "large", False,
     "In scenario 1, list every Server-kind span of the cart service with its timestamp, "
     "operation and duration in milliseconds.",
     f"SELECT ts, operation, duration_ms FROM span WHERE service = 'cart' AND kind = "
     f"'Server' AND {S}"),
    ("IT-G5", "large", "large", False,
     "In scenario 1, list every pod in every namespace with its namespace, node and phase.",
     f"SELECT name, namespace, node_name, phase FROM k8s_object WHERE kind = 'Pod' AND {S}"),
    ("IT-W4", "window", "small", False,
     "In scenario 1, how many spans did the checkout service emit in each minute from "
     "17:25 (inclusive) to 17:29 (exclusive) UTC on 2025-12-15? Give the minute start "
     "and count.",
     f"SELECT date_trunc('minute', ts), count(*) FROM span WHERE service = 'checkout' "
     f"AND ts >= TIMESTAMP '2025-12-15 17:25:00' AND ts < TIMESTAMP '2025-12-15 17:29:00' "
     f"AND {S} GROUP BY 1"),
]


def _check(stores: list[Path], qid: str, ordered: bool, sql: str) -> int:
    n = None
    for path in stores:
        con = duckdb.connect(str(path), read_only=True)
        rows = con.execute(sql).fetchall()
        if not rows or rows == [(None,)]:
            sys.exit(f"{qid}: empty answer on {path.name}")
        if ordered:
            head, k = sql.rsplit(" LIMIT ", 1)
            k = int(k)
            nxt = con.execute(f"{head} LIMIT {k + 1}").fetchall()
            if len(nxt) <= k:
                sys.exit(f"{qid}: fewer than k + 1 candidates on {path.name}")
            if nxt[k][-1] == nxt[k - 1][-1]:
                sys.exit(f"{qid}: tie at the top-{k} boundary on {path.name}")
        n = len(rows)
        con.close()
    return n


def _append(path: Path, stores: list[Path], items: list[tuple]) -> None:
    have = {json.loads(line)["id"] for line in path.open() if line.strip()}
    added = []
    for qid, fam, size, ordered, text, sql in items:
        if qid in have:
            continue
        n = _check(stores, qid, ordered, sql)
        print(f"{qid:7} {fam:13} rows(last scale)={n}")
        added.append({"id": qid, "family": fam, "size_class": size, "ordered": ordered,
                      "question": text, "gold_sql": sql, "plan": [], "workflow": None,
                      "workflow_params": {}})
    with path.open("a") as fh:
        for q in added:
            fh.write(json.dumps(q) + "\n")
    print(f"{path.name}: +{len(added)} -> {len(have) + len(added)}")


def main() -> None:
    w = ROOT / "workloads"
    _append(w / "tpch_questions.jsonl", [tpch.db_path(0.01), tpch.db_path(1)], TPCH)
    _append(w / "tpcds_questions.jsonl", [tpcds.db_path(0.01), tpcds.db_path(1)], TPCDS)
    _append(w / "itbench_questions.jsonl",
            [itbench.db_path("s1"), itbench.db_path("s35")], ITB)


if __name__ == "__main__":
    main()
