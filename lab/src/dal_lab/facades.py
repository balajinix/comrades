"""One dataset, many data-access modalities.

Every facade is generated *from the catalog* of the same DuckDB store, with no
per-question tuning, so differences in communication cost come from the modality
and not from how much effort went into one interface:

- :class:`GenericFacade`: one ``run_sql`` tool (query shipping / pushdown).
- :class:`FineGrainedFacade`: ``get_<table>`` / ``list_<table>`` per table with
  equality and range filters and pagination but no joins or aggregation, like
  typical REST or vendor MCP tools (data shipping).
- :class:`HandleFacade`: ``run_query`` returns a result *handle*, a row count and
  a short preview; rows are fetched or aggregated by reference.
- :class:`WorkflowFacade`: one coded, parameterised tool per known intent.
- :func:`compact` / :class:`CompactedFacade`: observation compaction that cuts
  every list uniformly to fit a character budget, reproducing a common agent-framework
  ``observation.py`` behaviour.

Each facade exposes MCP-style descriptors (``name``, ``description``,
``inputSchema``) so the cost of delivering the interface to a model is measurable.
"""
from __future__ import annotations

import itertools
import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import duckdb

from .meter import Meter, to_json

MAX_ROWS = 1000
_OPS = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_SELECT_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def open_store(path: Path) -> duckdb.DuckDBPyConnection:
    """Open an in-memory session with the dataset attached read-only.

    Temp tables (for result handles) live in the in-memory session; the dataset
    itself cannot be modified.

    Args:
        path: DuckDB file to attach.

    Returns:
        A connection whose default schema is the attached dataset.
    """
    con = duckdb.connect()
    con.execute(f"ATTACH '{path}' AS store (READ_ONLY)")
    con.execute("USE store")
    return con


@dataclass
class Catalog:
    """Tables and typed columns of the attached store."""

    tables: dict[str, list[tuple[str, str]]]
    # Column comments from the store (its data dictionary): value sets and meaning.
    # Empty for the TPC stores, whose questions name literal values.
    notes: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def load(cls, con: duckdb.DuckDBPyConnection) -> "Catalog":
        """Read the catalog from ``information_schema``.

        Args:
            con: Connection from :func:`open_store`.

        Returns:
            The catalog.
        """
        rows = con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_catalog = 'store' ORDER BY table_name, ordinal_position").fetchall()
        tables: dict[str, list[tuple[str, str]]] = {}
        for table, col, typ in rows:
            tables.setdefault(table, []).append((col, typ))
        notes = {(t, c): n for t, c, n in con.execute(
            "SELECT table_name, column_name, comment FROM duckdb_columns() "
            "WHERE database_name = 'store' AND comment IS NOT NULL").fetchall()}
        return cls(tables, notes)

    def note(self, table: str, col: str) -> str | None:
        """The data-dictionary note for a column, if the store has one."""
        return self.notes.get((table, col))

    def columns(self, table: str) -> list[str]:
        """Column names of ``table``."""
        return [c for c, _ in self.tables[table]]

    def key(self, table: str) -> str:
        """Lookup key of ``table``: by convention its first column."""
        return self.tables[table][0][0]

    def describe(self) -> str:
        """Compact schema text a query-writing model needs (table(col type, ...))."""
        text = "\n".join(f"{t}({', '.join(f'{c} {ty}' for c, ty in cols)})"
                          for t, cols in self.tables.items())
        if self.notes:
            text += "\nColumn notes:\n" + "\n".join(
                f"- {t}.{c}: {n}" for (t, c), n in sorted(self.notes.items()))
        return text


def _rows(cur: duckdb.DuckDBPyConnection, limit: int | None = None) -> tuple[list[str], list[dict]]:
    cols = [d[0] for d in cur.description]
    data = cur.fetchall() if limit is None else cur.fetchmany(limit)
    return cols, [dict(zip(cols, r)) for r in data]


@dataclass
class Facade:
    """A named set of tools plus their MCP-style descriptors."""

    name: str
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)
    descriptors: list[dict] = field(default_factory=list)
    observe: Callable[[Any], Any] | None = None

    def bind(self, meter: Meter) -> dict[str, Callable[..., Any]]:
        """Return metered versions of every tool.

        Args:
            meter: The gateway that records each call.

        Returns:
            Tool name -> metered callable.
        """
        return {n: meter.wrap(n, fn, self.observe) for n, fn in self.tools.items()}

    def _add(self, name: str, fn: Callable[..., Any], description: str,
             properties: dict, required: list[str]) -> None:
        self.tools[name] = fn
        self.descriptors.append({
            "name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": required}})


class GenericFacade(Facade):
    """A single generic query tool: the model writes SQL, the store computes."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 max_rows: int = MAX_ROWS, truncation: str = "flag") -> None:
        """Args:
            truncation: How a capped result is reported, for the truncation ablation:
                ``flag`` (a boolean, the default), ``silent`` (no flag, and the tool
                description does not mention the cap) or ``labelled`` (the flag, the
                full row count and a note that the rows are incomplete).
        """
        if truncation not in ("flag", "silent", "labelled"):
            raise ValueError(f"unknown truncation mode {truncation!r}")
        super().__init__("generic" if truncation == "flag" else f"sql+{truncation}")
        self.con, self.max_rows, self.truncation = con, max_rows, truncation
        cap = "" if truncation == "silent" else f" up to {max_rows}"
        self._add("run_sql", self.run_sql,
                  "Run a read-only SQL (DuckDB dialect) query over the dataset and "
                  f"return{cap} rows. Schema:\n{catalog.describe()}",
                  {"query": {"type": "string", "description": "SELECT statement"}},
                  ["query"])

    def run_sql(self, query: str) -> dict:
        """Execute ``query`` and return at most ``max_rows`` rows.

        Args:
            query: A SELECT/WITH statement.

        Returns:
            ``{columns, rows, row_count, truncated}``; without ``truncated`` when
            silent, and with ``total_rows`` and ``note`` when labelled and capped.

        Raises:
            ValueError: If the statement is not a query.
        """
        if not _SELECT_RE.match(query):
            raise ValueError("only SELECT/WITH queries are allowed")
        cur = self.con.execute(query)
        cols, rows = _rows(cur, self.max_rows + 1)
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        out = {"columns": cols, "rows": rows, "row_count": len(rows)}
        if self.truncation == "silent":
            return out
        out["truncated"] = truncated
        if truncated and self.truncation == "labelled":
            total = self.con.execute(f"SELECT count(*) FROM ({query})").fetchone()[0]
            out["total_rows"] = total
            out["note"] = (f"INCOMPLETE: showing {len(rows)} of {total} rows; an answer "
                           "built from these rows alone is wrong")
        return out


class QueryAnswerFacade(GenericFacade):
    """Plain SQL that may also submit its final query as the answer.

    The control for answering by reference: ``run_sql`` is unchanged (rows inline,
    capped, flagged when truncated), and ``save_query`` stores a query's full result
    under a handle the model can answer with. Only how the answer is delivered
    differs from :class:`GenericFacade`; the model still sees no preview or count
    until it saves.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 max_rows: int = MAX_ROWS) -> None:
        super().__init__(con, catalog, max_rows)
        self.name = "sql+answer_query"
        self._ids = itertools.count(1)
        self._ns = uuid.uuid4().hex[:6]
        self.handles: dict[str, str] = {}
        self._add("save_query", self.save_query,
                  "Store the full result of a read-only SQL query as a candidate answer. "
                  "Returns a handle and the row count, not the rows; answer with that "
                  "handle when the result is too large to write out.",
                  {"query": {"type": "string", "description": "SELECT statement"}},
                  ["query"])

    def save_query(self, query: str) -> dict:
        """Materialise ``query`` under a new handle.

        Args:
            query: A SELECT/WITH statement.

        Returns:
            ``{handle, row_count, columns}``.

        Raises:
            ValueError: If the statement is not a query.
        """
        if not _SELECT_RE.match(query):
            raise ValueError("only SELECT/WITH queries are allowed")
        handle = f"q{self._ns}_{next(self._ids)}"
        self.con.execute(f"CREATE TEMP TABLE {handle} AS {query}")
        self.handles[handle] = f"temp.main.{handle}"
        count = self.con.execute(f"SELECT count(*) FROM {handle}").fetchone()[0]
        cols = [d[0] for d in self.con.execute(f"SELECT * FROM {handle} LIMIT 0").description]
        return {"handle": handle, "row_count": count, "columns": cols}

    def release(self) -> None:
        """Drop every saved result (between runs)."""
        for handle in self.handles:
            self.con.execute(f"DROP TABLE IF EXISTS {handle}")
        self.handles.clear()


class FineGrainedFacade(Facade):
    """Per-table lookup and list tools, auto-generated from the catalog."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 max_rows: int = MAX_ROWS) -> None:
        super().__init__("fine_grained")
        self.con, self.catalog, self.max_rows = con, catalog, max_rows
        for table in catalog.tables:
            cols = catalog.columns(table)
            key = catalog.key(table)
            self._add(f"get_{table}", self._getter(table),
                      f"Get {table} records by {key}.",
                      {"key": {"type": ["integer", "string"],
                               "description": f"value of {key}"}}, ["key"])
            filt = {c: {"description": "value, or {op: value} with op in "
                                        "eq, ne, lt, le, gt, ge, in"
                                        + (f". {catalog.note(table, c)}"
                                           if catalog.note(table, c) else "")}
                    for c in cols}
            self._add(f"list_{table}", self._lister(table),
                      f"List {table} records matching filters, paginated. "
                      f"Fields: {', '.join(cols)}.",
                      {"filters": {"type": "object", "properties": filt},
                       "limit": {"type": "integer", "maximum": max_rows},
                       "offset": {"type": "integer"}}, [])

    def _getter(self, table: str) -> Callable[..., dict]:
        def get(key: Any) -> dict:
            k = self.catalog.key(table)
            cur = self.con.execute(f'SELECT * FROM "{table}" WHERE "{k}" = ?', [key])
            return {"rows": _rows(cur)[1]}
        return get

    def _lister(self, table: str) -> Callable[..., dict]:
        def lst(filters: dict | None = None, limit: int = 100, offset: int = 0) -> dict:
            where, params = self._where(table, filters or {})
            limit = max(1, min(int(limit), self.max_rows))
            cur = self.con.execute(
                f'SELECT * FROM "{table}"{where} ORDER BY rowid LIMIT ? OFFSET ?',
                params + [limit + 1, int(offset)])
            rows = _rows(cur)[1]
            more = len(rows) > limit
            return {"rows": rows[:limit], "next_offset": offset + limit if more else None}
        return lst

    def _where(self, table: str, filters: dict) -> tuple[str, list]:
        cols = set(self.catalog.columns(table))
        clauses, params = [], []
        for col, spec in filters.items():
            if col not in cols:
                raise ValueError(f"unknown field {col} for {table}")
            items = spec.items() if isinstance(spec, dict) else [("eq", spec)]
            for op, val in items:
                if op == "in":
                    vals = list(val)
                    if not vals:
                        clauses.append("FALSE")
                        continue
                    clauses.append(f'"{col}" IN ({", ".join("?" * len(vals))})')
                    params.extend(vals)
                elif op in _OPS:
                    clauses.append(f'"{col}" {_OPS[op]} ?')
                    params.append(val)
                else:
                    raise ValueError(f"unsupported operator {op}")
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


class HandleFacade(Facade):
    """Query tool that returns results by reference (a handle plus a preview)."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 preview_rows: int = 5, max_rows: int = MAX_ROWS,
                 lazy: bool = False) -> None:
        # lazy=True keeps a handle as a view: nothing is materialised, and every
        # count, page or aggregate re-runs the query (cost moves to compute).
        super().__init__("handles+lazy" if lazy else "handles")
        self.lazy = lazy
        self.con, self.preview_rows, self.max_rows = con, preview_rows, max_rows
        self._ids = itertools.count(1)
        self._ns = uuid.uuid4().hex[:6]  # handles are unique per facade instance
        self.handles: dict[str, str] = {}
        self._add("run_query", self.run_query,
                  "Run a read-only SQL query; returns a result handle, the row count "
                  f"and a preview of at most {preview_rows} rows (not the full result). "
                  f"Schema:\n{catalog.describe()}",
                  {"query": {"type": "string"}}, ["query"])
        self._add("fetch_rows", self.fetch_rows, "Fetch rows of a result handle.",
                  {"handle": {"type": "string"}, "offset": {"type": "integer"},
                   "limit": {"type": "integer", "maximum": max_rows}}, ["handle"])
        self._add("aggregate", self.aggregate,
                  "Aggregate a result handle: op in count,sum,avg,min,max over column, "
                  "optionally grouped.",
                  {"handle": {"type": "string"}, "op": {"type": "string"},
                   "column": {"type": "string"}, "group_by": {"type": "string"}},
                  ["handle", "op"])

    def run_query(self, query: str) -> dict:
        """Materialise ``query`` as a handle and return count + preview.

        Args:
            query: A SELECT/WITH statement.

        Returns:
            ``{handle, row_count, columns, preview}`` plus a ``note`` when truncated.
        """
        if not _SELECT_RE.match(query):
            raise ValueError("only SELECT/WITH queries are allowed")
        handle = f"h{self._ns}_{next(self._ids)}"
        table = f"temp.main.{handle}"
        kind = "VIEW" if self.lazy else "TABLE"
        self.con.execute(f"CREATE TEMP {kind} {handle} AS {query}")
        self.handles[handle] = table
        count = self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        cols, preview = _rows(self.con.execute(
            f"SELECT * FROM {table} LIMIT {self.preview_rows}"))
        # The sample is called "preview" and carries an explicit note: with the key
        # "rows", models answered with the sample as if it were the whole result.
        shown = len(preview)
        out = {"handle": handle, "row_count": count, "columns": cols, "preview": preview}
        if count > shown:
            out["note"] = (f"preview shows {shown} of {count} rows; use fetch_rows, "
                           "aggregate, or answer with answer_handle")
        return out

    def fetch_rows(self, handle: str, offset: int = 0, limit: int = 100) -> dict:
        """Page through a handle's rows."""
        limit = max(1, min(int(limit), self.max_rows))
        cur = self.con.execute(f"SELECT * FROM {self.handles[handle]} LIMIT ? OFFSET ?",
                               [limit, int(offset)])
        return {"rows": _rows(cur)[1]}

    def aggregate(self, handle: str, op: str, column: str = "*",
                  group_by: str | None = None) -> dict:
        """Aggregate a handle without shipping its rows."""
        if op not in {"count", "sum", "avg", "min", "max"}:
            raise ValueError(f"unsupported op {op}")
        col = "*" if column == "*" else f'"{column}"'
        if group_by:
            sql = (f'SELECT "{group_by}", {op}({col}) AS value FROM {self.handles[handle]} '
                   f'GROUP BY 1 ORDER BY 1')
        else:
            sql = f"SELECT {op}({col}) AS value FROM {self.handles[handle]}"
        return {"rows": _rows(self.con.execute(sql))[1]}

    def release(self) -> None:
        """Drop every handle (between runs, so memory is not carried over)."""
        for handle in list(self.handles):
            if handle.startswith("h"):
                kind = "VIEW" if self.lazy else "TABLE"
                self.con.execute(f"DROP {kind} IF EXISTS {handle}")
            else:  # code-mode answers are always temp tables
                self.con.execute(f"DROP TABLE IF EXISTS {handle}")
        self.handles.clear()

    def table_of(self, handle: str) -> str:
        """Return the temp table holding a handle (for scoring, not for the model)."""
        return self.handles[handle]


class WorkflowFacade(Facade):
    """One coded tool per known intent; the developer did the planning."""

    def __init__(self, con: duckdb.DuckDBPyConnection,
                 workflows: dict[str, dict]) -> None:
        super().__init__("workflow")
        self.con = con
        for name, wf in workflows.items():
            props = {p: {"type": "string"} for p in wf["params"]}
            self._add(name, self._runner(wf["sql"]), wf["description"], props,
                      list(wf["params"]))

    def _runner(self, sql: str) -> Callable[..., dict]:
        def run(**params: Any) -> dict:
            cols, rows = _rows(self.con.execute(sql, params))
            return {"columns": cols, "rows": rows}
        return run


class HybridFacade(Facade):
    """Coded workflows first, with a general query tool as the fallback.

    The pilot showed workflows fail outright outside their coverage, and the model
    answers anyway instead of saying so. This arm keeps the cheap intents and adds
    an escape hatch, as a deployed DAL would.
    """

    def __init__(self, workflows: Facade, fallback: Facade, name: str) -> None:
        super().__init__(name)
        for fac in (workflows, fallback):
            self.tools.update(fac.tools)
            self.descriptors.extend(fac.descriptors)


def compact(obj: Any, limit: int = 4000, annotate: bool = False) -> Any:
    """Cut every list in ``obj`` to the same length k so the JSON fits ``limit`` chars.

    Mirrors a typical agent-side ``_compact``: the largest k whose serialisation fits is
    used, uniformly for all nested lists. With ``annotate=False`` (the common
    behaviour) nothing tells the model rows were dropped.

    Args:
        obj: Tool result.
        limit: Character budget.
        annotate: Append ``{"_truncated": n}`` markers to cut lists.

    Returns:
        The compacted result (``obj`` unchanged if it already fits).
    """
    if len(to_json(obj)) <= limit:
        return obj

    def cut(o: Any, k: int) -> Any:
        if isinstance(o, list):
            kept = [cut(x, k) for x in o[:k]]
            if annotate and len(o) > k:
                kept.append({"_truncated": len(o) - k})
            return kept
        if isinstance(o, dict):
            return {key: cut(v, k) for key, v in o.items()}
        return o

    lo, hi = 0, _max_list_len(obj)
    while lo < hi:  # largest k that fits
        mid = (lo + hi + 1) // 2
        if len(to_json(cut(obj, mid))) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return cut(obj, lo)


def _max_list_len(o: Any) -> int:
    if isinstance(o, list):
        return max([len(o)] + [_max_list_len(x) for x in o])
    if isinstance(o, dict):
        return max([0] + [_max_list_len(v) for v in o.values()])
    return 0


class CompactedFacade(Facade):
    """Any facade whose observations pass through :func:`compact`."""

    def __init__(self, base: Facade, limit: int = 4000, annotate: bool = False) -> None:
        super().__init__(f"{base.name}+compact", dict(base.tools), list(base.descriptors),
                         observe=lambda r: compact(r, limit, annotate))
        self.base = base
