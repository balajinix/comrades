"""Adaptive data-access interfaces: the design rules turned into a mechanism.

- ``sql+auto``: one SQL tool whose *result representation* adapts to the result.
  Small results (at most ``inline_max`` rows) come back inline, as with plain SQL;
  larger ones become a handle with a preview, as with the handles interface. The
  model never chooses; the data layer does, from the actual cardinality.
- ``sql+schema``: SQL whose *schema delivery* is on demand. The tool description
  carries no schema; ``list_tables`` and ``describe_table`` return it when asked.
- ``router``: a rule-based adaptive data access layer. Coded workflows first, then
  ``sql+auto``; the schema goes inline when it is small and on demand when it is not.

Handles are registered in the lab's shared :class:`HandleFacade`, so an agent may
answer with ``answer_handle`` and grading reads the same table.
"""
from __future__ import annotations

import re

import duckdb

from .facades import _SELECT_RE, Catalog, Facade, HandleFacade, _rows
from .graphql_facade import foreign_keys
from .meter import count_tokens

INLINE_MAX = 50              # rows returned inline before switching to a handle
SCHEMA_INLINE_TOKENS = 2000  # router: schemas up to this size go in the description
_SHOW_TABLES_RE = re.compile(r"^\s*show\s+(all\s+)?tables\s*;?\s*$", re.IGNORECASE)
_DESCRIBE_RE = re.compile(r"^\s*(?:describe|desc|show\s+columns\s+from)\s+"
                          r"[\"`]?(\w+)[\"`]?\s*;?\s*$", re.IGNORECASE)


class SchemaTools:
    """``list_tables`` and ``describe_table`` over a catalog (schema on demand)."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog) -> None:
        self.con, self.catalog = con, catalog
        self.fks = foreign_keys(catalog)

    def list_tables(self) -> dict:
        """Table names with column counts."""
        return {"tables": [{"table": t, "columns": len(c)}
                           for t, c in self.catalog.tables.items()]}

    def describe_table(self, table: str) -> dict:
        """Columns (with types and notes) and foreign keys of one table.

        Raises:
            ValueError: For an unknown table.
        """
        if table not in self.catalog.tables:
            raise ValueError(f"unknown table {table!r}; call list_tables")
        cols = []
        for c, ty in self.catalog.tables[table]:
            col = {"column": c, "type": ty}
            if self.catalog.note(table, c):
                col["note"] = self.catalog.note(table, c)
            cols.append(col)
        fks = [f"{ct}.{cc} -> {pt}.{pc}" for (ct, cc), (pt, pc) in self.fks.items()
               if table in (ct, pt)]
        return {"table": table, "columns": cols, "foreign_keys": fks}

    def introspect(self, query: str) -> dict | None:
        """Answer schema statements sent to ``run_sql``; None for a real query.

        Some models probe an unseen schema with ``SHOW TABLES`` or ``DESCRIBE t``
        rather than the schema tools. Rejecting those made one model stop exploring
        and answer from nothing, so they are served from the catalog instead.

        Raises:
            ValueError: For any other statement that is not a query.
        """
        if _SHOW_TABLES_RE.match(query):
            return self.list_tables()
        m = _DESCRIBE_RE.match(query)
        if m:
            name = m.group(1)
            return self.describe_table(name if name in self.catalog.tables else name.lower())
        if not _SELECT_RE.match(query):
            raise ValueError("only SELECT/WITH queries are allowed; call list_tables "
                             "or describe_table to see the schema")
        return None

    def add_to(self, facade: Facade) -> None:
        """Register both tools on ``facade``."""
        facade._add("list_tables", self.list_tables,
                    "List the tables in the database (names and column counts).", {}, [])
        facade._add("describe_table", self.describe_table,
                    "Describe one table: columns, types, notes and foreign keys.",
                    {"table": {"type": "string"}}, ["table"])


class AutoHandleFacade(Facade):
    """SQL whose results come back inline when small and as a handle when large."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 handles: HandleFacade, inline_max: int = INLINE_MAX,
                 schema: str = "full", name: str = "sql+auto") -> None:
        super().__init__(name)
        self.con, self.handles, self.inline_max = con, handles, inline_max
        schema_text = (f"Schema:\n{catalog.describe()}" if schema == "full" else
                       "Call list_tables and describe_table to see the schema.")
        self._add("run_sql", self.run_sql,
                  "Run a read-only SQL (DuckDB) query. Results of at most "
                  f"{inline_max} rows come back in full under 'rows'. Larger results "
                  "come back as a handle with the row count and a preview; use "
                  "fetch_rows or aggregate on it, or answer with answer_handle. "
                  + schema_text, {"query": {"type": "string"}}, ["query"])
        for tool in ("fetch_rows", "aggregate"):
            desc = next(d for d in handles.descriptors if d["name"] == tool)
            self.tools[tool] = handles.tools[tool]
            self.descriptors.append(desc)
        self.schema_tools = None
        if schema != "full":
            self.schema_tools = SchemaTools(con, catalog)
            self.schema_tools.add_to(self)

    def run_sql(self, query: str) -> dict:
        """Run ``query``; inline rows if small, else a handle with a preview."""
        if self.schema_tools is not None:
            schema = self.schema_tools.introspect(query)
            if schema is not None:
                return schema
        out = self.handles.run_query(query)
        if out["row_count"] <= self.inline_max:
            cols, rows = _rows(self.con.execute(
                f"SELECT * FROM {self.handles.table_of(out['handle'])}"))
            return {"columns": cols, "rows": rows, "row_count": out["row_count"]}
        return out


class OnDemandSchemaFacade(Facade):
    """Plain SQL (inline rows) with the schema fetched by tool calls."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 generic: Facade) -> None:
        super().__init__("sql+schema")
        self.schema_tools = SchemaTools(con, catalog)
        self._sql = generic.tools["run_sql"]
        self.tools["run_sql"] = self.run_sql
        desc = dict(next(d for d in generic.descriptors if d["name"] == "run_sql"))
        desc["description"] = desc["description"].split("Schema:")[0] + (
            "Call list_tables and describe_table to see the schema.")
        self.descriptors.append(desc)
        self.schema_tools.add_to(self)

    def run_sql(self, query: str) -> dict:
        """Run ``query`` with plain SQL, serving schema statements from the catalog."""
        schema = self.schema_tools.introspect(query)
        return schema if schema is not None else self._sql(query)


def router_facade(con: duckdb.DuckDBPyConnection, catalog: Catalog,
                  handles: HandleFacade, workflows: Facade) -> Facade:
    """The rule-based adaptive DAL: workflows, then SQL with adaptive results.

    Schema delivery is chosen from the catalog's size: inline when its description
    fits in :data:`SCHEMA_INLINE_TOKENS`, on demand otherwise.
    """
    schema = "full" if count_tokens(catalog.describe()) <= SCHEMA_INLINE_TOKENS else "on_demand"
    sql = AutoHandleFacade(con, catalog, handles, schema=schema, name="router-sql")
    out = Facade("router")
    for fac in (workflows, sql):
        out.tools.update(fac.tools)
        out.descriptors.extend(fac.descriptors)
    return out
