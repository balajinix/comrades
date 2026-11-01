"""Skills: schema and know-how delivered on demand (progressive disclosure).

An Agent Skill is a named bundle of instructions and resources. The agent sees only
an index (name + one line); it loads a skill's body when it decides the skill is
relevant. For data access this changes *how the schema reaches the agent*: instead
of the whole schema in every prompt (the ``generic`` facade), the agent pays for one
``load_skill`` call per domain it touches, and the schema text enters the context
once, at that point.

Skills here are generated from the catalog and grouped by business domain. Each body
lists the domain's tables with typed columns, the join paths, and two example
queries written for this module (never the benchmark's gold queries), plus a short
procedure. The query tool is the same generic SQL tool, but its description carries
no schema.
"""
from __future__ import annotations

import duckdb

from .facades import Catalog, Facade, GenericFacade
from .graphql_facade import foreign_keys
from .obfuscate import TABLE_MAP, rename_sql

# Domains over plain TPC-H names: (skill name, one-line summary, tables, examples).
DOMAINS = [
    ("customers-and-orders",
     "Customers, their orders, order status/dates/totals, market segments.",
     ["customer", "orders", "nation"],
     ["SELECT c_name, count(*) FROM customer JOIN orders ON o_custkey = c_custkey "
      "GROUP BY c_name ORDER BY 2 DESC LIMIT 3",
      "SELECT o_orderstatus, avg(o_totalprice) FROM orders GROUP BY 1"]),
    ("order-lines-and-revenue",
     "Line items of orders: quantities, prices, discounts, ship dates; revenue.",
     ["lineitem", "orders", "part"],
     ["SELECT l_orderkey, sum(l_quantity) FROM lineitem GROUP BY 1 LIMIT 5",
      "SELECT year(l_shipdate) AS y, sum(l_extendedprice * (1 - l_discount)) "
      "FROM lineitem GROUP BY 1 ORDER BY 1"]),
    ("parts-and-suppliers",
     "Parts (brand, type, size, price), suppliers, and who supplies what at what cost.",
     ["part", "partsupp", "supplier"],
     ["SELECT p_brand, count(*) FROM part GROUP BY 1 ORDER BY 1 LIMIT 5",
      "SELECT s_name, min(ps_supplycost) FROM partsupp JOIN supplier "
      "ON ps_suppkey = s_suppkey GROUP BY 1 LIMIT 5"]),
    ("geography",
     "Regions and nations; locating customers and suppliers by nation or region.",
     ["region", "nation", "customer", "supplier"],
     ["SELECT r_name, count(*) FROM nation JOIN region ON n_regionkey = r_regionkey "
      "GROUP BY 1",
      "SELECT n_name, count(*) FROM supplier JOIN nation ON s_nationkey = n_nationkey "
      "GROUP BY 1 ORDER BY 2 DESC LIMIT 3"]),
]

# TPC-DS domains: the broad-schema rung. Examples are generic, not benchmark answers.
DS_DOMAINS = [
    ("store-channel", "In-store sales and returns by store, item and date.",
     ["store_sales", "store_returns", "store", "item", "date_dim"],
     ["SELECT s_store_name, sum(ss_net_paid) FROM store_sales JOIN store "
      "ON ss_store_sk = s_store_sk GROUP BY 1 ORDER BY 2 DESC LIMIT 3",
      "SELECT d_year, count(*) FROM store_returns JOIN date_dim "
      "ON sr_returned_date_sk = d_date_sk GROUP BY 1"]),
    ("catalog-channel", "Catalog sales and returns, catalog pages, call centers.",
     ["catalog_sales", "catalog_returns", "catalog_page", "call_center", "item",
      "date_dim"],
     ["SELECT cc_name, count(*) FROM catalog_sales JOIN call_center "
      "ON cs_call_center_sk = cc_call_center_sk GROUP BY 1",
      "SELECT cp_department, sum(cs_quantity) FROM catalog_sales JOIN catalog_page "
      "ON cs_catalog_page_sk = cp_catalog_page_sk GROUP BY 1 LIMIT 5"]),
    ("web-channel", "Web sales and returns, web sites and pages.",
     ["web_sales", "web_returns", "web_site", "web_page", "item", "date_dim"],
     ["SELECT web_name, sum(ws_net_paid) FROM web_sales JOIN web_site "
      "ON ws_web_site_sk = web_site_sk GROUP BY 1",
      "SELECT wp_type, count(*) FROM web_sales JOIN web_page "
      "ON ws_web_page_sk = wp_web_page_sk GROUP BY 1"]),
    ("customers", "Customers, addresses, demographics, households, income bands.",
     ["customer", "customer_address", "customer_demographics",
      "household_demographics", "income_band"],
     ["SELECT ca_state, count(*) FROM customer JOIN customer_address "
      "ON c_current_addr_sk = ca_address_sk GROUP BY 1 ORDER BY 2 DESC LIMIT 3",
      "SELECT cd_education_status, count(*) FROM customer_demographics GROUP BY 1"]),
    ("items-and-promotions", "Items (brand, category, class, price) and promotions.",
     ["item", "promotion"],
     ["SELECT i_category, avg(i_current_price) FROM item GROUP BY 1",
      "SELECT p_channel_email, count(*) FROM promotion GROUP BY 1"]),
    ("inventory-and-warehouses", "Stock on hand per item, warehouse and week.",
     ["inventory", "warehouse", "item", "date_dim"],
     ["SELECT w_warehouse_name, sum(inv_quantity_on_hand) FROM inventory JOIN warehouse "
      "ON inv_warehouse_sk = w_warehouse_sk GROUP BY 1"]),
    ("shipping-time-reasons", "Ship modes, time of day, return reasons, calendar.",
     ["ship_mode", "time_dim", "reason", "date_dim"],
     ["SELECT sm_type, count(*) FROM ship_mode GROUP BY 1",
      "SELECT d_moy, count(*) FROM date_dim WHERE d_year = 2000 GROUP BY 1"]),
]

# ITBench domains (the observability store). Examples are generic.
ITB_DOMAINS = [
    ("kubernetes-topology", "Kubernetes objects: pods, nodes, workloads, owners, deletions.",
     ["k8s_object", "scenario"],
     ["SELECT node_name, count(*) FROM k8s_object WHERE kind = 'Pod' GROUP BY 1",
      "SELECT p.name, rs.name FROM k8s_object p JOIN k8s_object rs "
      "ON p.owner_uid = rs.uid WHERE p.kind = 'Pod' LIMIT 5"]),
    ("kubernetes-events", "Kubernetes events (reasons, types) about objects, over time.",
     ["k8s_event", "k8s_object"],
     ["SELECT reason, count(*) FROM k8s_event GROUP BY 1 ORDER BY 2 DESC LIMIT 5"]),
    ("traces", "OpenTelemetry spans: services, operations, durations, errors, call graph.",
     ["span", "k8s_object"],
     ["SELECT service, count(*) FROM span WHERE status = 'Error' GROUP BY 1",
      "SELECT DISTINCT p.service, c.service FROM span c JOIN span p "
      "ON c.parent_span_id = p.span_id AND c.trace_id = p.trace_id "
      "WHERE p.service <> c.service LIMIT 5"]),
    ("alerts-and-logs", "Prometheus alerts and application logs, linked to pods.",
     ["alert", "log", "k8s_object"],
     ["SELECT alertname, severity, first_seen FROM alert ORDER BY first_seen LIMIT 5",
      "SELECT service, count(*) FROM log GROUP BY 1 ORDER BY 2 DESC LIMIT 3"]),
    ("incidents", "Scenarios and the injected fault (ground truth) of each.",
     ["scenario"],
     ["SELECT scenario_id, fault_kind, fault_category FROM scenario"]),
]

PROCEDURE = ("Procedure: load every skill whose domain the question touches, then write "
             "ONE SQL query that computes the answer (filters, joins, aggregates, "
             "ORDER BY/LIMIT) and answer from its rows.")


class SkillsFacade(Facade):
    """``list_skills`` / ``load_skill`` plus a schema-less SQL tool."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog) -> None:
        super().__init__("skills")
        tpcds = "store_sales" in catalog.tables
        itb = "span" in catalog.tables and "k8s_object" in catalog.tables
        renamed = not tpcds and not itb and "lineitem" not in catalog.tables
        fks = foreign_keys(catalog)
        self.skills: dict[str, tuple[str, str]] = {}
        for name, summary, tables, examples in (DS_DOMAINS if tpcds else ITB_DOMAINS if itb else DOMAINS):
            if renamed:
                tables = [TABLE_MAP[t] for t in tables]
                examples = [rename_sql(e) for e in examples]
            self.skills[name] = (summary, self._body(catalog, fks, tables, examples))
        generic = GenericFacade(con, catalog)
        # As in Agent Skills, the index (name + summary) is always visible; bodies
        # are loaded on demand.
        index = "\n".join(f"- {n}: {s}" for n, (s, _) in self.skills.items())
        self._add("list_skills", self.list_skills,
                  "List the available data skills. Load a skill to get its tables, "
                  f"columns, join paths and examples. Skills:\n{index}", {}, [])
        self._add("load_skill", self.load_skill, "Load one skill's full text by name.",
                  {"name": {"type": "string"}}, ["name"])
        self._add("run_sql", generic.run_sql,
                  "Run a read-only SQL (DuckDB dialect) query and return up to "
                  f"{generic.max_rows} rows. The schema is in the skills.",
                  {"query": {"type": "string"}}, ["query"])

    @staticmethod
    def _body(catalog: Catalog, fks: dict, tables: list[str], examples: list[str]) -> str:
        lines = ["Tables:"]
        for t in tables:
            cols = ", ".join(f"{c} {ty}" for c, ty in catalog.tables[t])
            lines.append(f"- {t}({cols})")
            lines += [f"  - {c}: {catalog.note(t, c)}" for c, _ in catalog.tables[t]
                      if catalog.note(t, c)]
        joins = [f"{ct}.{cc} = {pt}.{pc}" for (ct, cc), (pt, pc) in fks.items()
                 if ct in tables and pt in tables]
        lines.append("Join paths: " + "; ".join(joins))
        lines.append("Examples:")
        lines += [f"- {e}" for e in examples]
        lines.append(PROCEDURE)
        return "\n".join(lines)

    def list_skills(self) -> dict:
        """Return the skill index."""
        return {"skills": [{"name": n, "summary": s} for n, (s, _) in self.skills.items()]}

    def load_skill(self, name: str) -> dict:
        """Return a skill's body.

        Raises:
            ValueError: For an unknown skill name.
        """
        if name not in self.skills:
            raise ValueError(f"unknown skill {name!r}; call list_skills")
        return {"name": name, "body": self.skills[name][1]}

