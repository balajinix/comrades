"""Capability-matched fine-grained tools: per-table tools plus join and aggregate.

The plain fine-grained interface can select and filter one table at a time, so a
model must rebuild joins and aggregates by fetching rows. That confounds two things:
tool granularity and missing operations. This facade keeps structured tools (no
free-form SQL) but adds the two missing operations:

- ``join``: walk a path of tables along foreign keys, filter, project, page.
- ``aggregate``: the same path, plus group-by, count/sum/avg/min/max/count_distinct,
  ordering and a limit (so top-k runs on the server).

Both build SQL from validated identifiers only; values are bound parameters.
"""
from __future__ import annotations

import duckdb

from .facades import MAX_ROWS, Catalog, FineGrainedFacade, _rows
from .graphql_facade import foreign_keys

AGG_OPS = {"count": "count", "sum": "sum", "avg": "avg", "min": "min", "max": "max",
           "count_distinct": "count(DISTINCT {})"}


class FineGrainedOpsFacade(FineGrainedFacade):
    """Per-table tools plus structured ``join`` and ``aggregate`` tools."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 max_rows: int = MAX_ROWS) -> None:
        super().__init__(con, catalog, max_rows)
        self.name = "fine_grained+ops"
        self.fks = foreign_keys(catalog)
        fk_text = "; ".join(f"{ct}.{cc} -> {pt}.{pc}"
                            for (ct, cc), (pt, pc) in self.fks.items())
        path = {"type": "array", "items": {"type": "object", "properties": {
            "table": {"type": "string"},
            "on": {"type": "string", "description": "foreign-key column linking this "
                   "table to the previous one, when there is more than one"}}},
            "description": "tables to join in order, each linked to the previous by a "
                           "foreign key"}
        filters = {"type": "object", "description": "{'table.column': value or "
                   "{op: value}} with op in eq, ne, lt, le, gt, ge, in, like"}
        self._add("join", self.join,
                  "Join tables along foreign keys and return rows (paginated). "
                  f"Foreign keys: {fk_text}.",
                  {"path": path, "select": {"type": "array", "items": {"type": "string"},
                                            "description": "'table.column' to return"},
                   "filters": filters, "limit": {"type": "integer", "maximum": max_rows},
                   "offset": {"type": "integer"}}, ["path", "select"])
        self._add("aggregate", self.aggregate,
                  "Aggregate over one table or a foreign-key join path, on the server. "
                  "Measures: [{op, column}] with op in count, sum, avg, min, max, "
                  "count_distinct (column may be '*' for count). Optional group_by, "
                  "order_by (a group_by column or 'm0', 'm1', ... for measures), desc, "
                  "limit.",
                  {"path": path, "measures": {"type": "array", "items": {"type": "object"}},
                   "group_by": {"type": "array", "items": {"type": "string"}},
                   "filters": filters, "order_by": {"type": "string"},
                   "desc": {"type": "boolean"}, "limit": {"type": "integer"}},
                  ["path", "measures"])

    # -- query building -------------------------------------------------------------
    def _from(self, path: list[dict]) -> tuple[str, list[str]]:
        """FROM clause for a join path; returns it with the alias of each step."""
        if not path:
            raise ValueError("path must name at least one table")
        aliases, parts = [], []
        for i, step in enumerate(path):
            table = step["table"] if isinstance(step, dict) else str(step)
            if table not in self.catalog.tables:
                raise ValueError(f"unknown table {table}")
            alias = f"t{i}"
            if i == 0:
                parts.append(f'"{table}" {alias}')
            else:
                prev = path[i - 1]["table"] if isinstance(path[i - 1], dict) else path[i - 1]
                on = step.get("on") if isinstance(step, dict) else None
                cond = self._link(prev, aliases[-1], table, alias, on)
                parts.append(f'JOIN "{table}" {alias} ON {cond}')
            aliases.append(alias)
        return " ".join(parts), aliases

    def _link(self, a: str, aa: str, b: str, ba: str, on: str | None) -> str:
        cands = []
        for (ct, cc), (pt, pc) in self.fks.items():
            if ct == a and pt == b:
                cands.append((cc, f'{aa}."{cc}" = {ba}."{pc}"'))
            if ct == b and pt == a:
                cands.append((cc, f'{ba}."{cc}" = {aa}."{pc}"'))
        if on:
            cands = [c for c in cands if c[0] == on]
        if not cands:
            raise ValueError(f"no foreign key links {a} and {b}"
                             + (f" via {on}" if on else ""))
        if len(cands) > 1:
            raise ValueError(f"{a} and {b} are linked by several keys "
                             f"({', '.join(c[0] for c in cands)}); set 'on'")
        return cands[0][1]

    def _col(self, ref: str, path: list[dict], aliases: list[str]) -> str:
        table, _, col = ref.rpartition(".")
        tables = [s["table"] if isinstance(s, dict) else s for s in path]
        if not table:
            table = tables[0]
        if table not in tables:
            raise ValueError(f"{ref}: table not in path")
        if col not in self.catalog.columns(table):
            raise ValueError(f"unknown column {ref}")
        return f'{aliases[tables.index(table)]}."{col}"'

    def _filters(self, filters: dict | None, path, aliases) -> tuple[str, list]:
        clauses, params = [], []
        ops = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
               "like": "LIKE"}
        for ref, spec in (filters or {}).items():
            col = self._col(ref, path, aliases)
            for op, val in (spec.items() if isinstance(spec, dict) else [("eq", spec)]):
                if op == "in":
                    vals = list(val)
                    clauses.append(f"{col} IN ({', '.join('?' * len(vals))})"
                                   if vals else "FALSE")
                    params += vals
                elif op in ops:
                    clauses.append(f"{col} {ops[op]} ?")
                    params.append(val)
                else:
                    raise ValueError(f"unknown operator {op}")
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    # -- tools ----------------------------------------------------------------------
    def join(self, path: list[dict], select: list[str], filters: dict | None = None,
             limit: int = 100, offset: int = 0) -> dict:
        """Rows of a foreign-key join, filtered and projected, one page at a time."""
        frm, aliases = self._from(path)
        cols = [self._col(s, path, aliases) + f' AS "{s}"' for s in select]
        where, params = self._filters(filters, path, aliases)
        limit = max(1, min(int(limit), self.max_rows))
        cur = self.con.execute(f"SELECT {', '.join(cols)} FROM {frm}{where} "
                               f"LIMIT ? OFFSET ?", params + [limit + 1, int(offset)])
        rows = _rows(cur)[1]
        more = len(rows) > limit
        return {"rows": rows[:limit], "next_offset": offset + limit if more else None}

    def aggregate(self, path: list[dict], measures: list[dict],
                  group_by: list[str] | None = None, filters: dict | None = None,
                  order_by: str | None = None, desc: bool = False,
                  limit: int | None = None) -> dict:
        """Grouped aggregates computed on the server (top-k via order_by + limit)."""
        frm, aliases = self._from(path)
        groups = [self._col(g, path, aliases) for g in group_by or []]
        sel = [f'{g} AS "{name}"' for g, name in zip(groups, group_by or [])]
        for i, m in enumerate(measures):
            op = str(m.get("op", "count")).lower()
            if op not in AGG_OPS:
                raise ValueError(f"unknown aggregate {op}")
            colref = m.get("column", "*")
            arg = "*" if colref in ("*", None) else self._col(colref, path, aliases)
            expr = (AGG_OPS[op].format(arg) if op == "count_distinct"
                    else f"{AGG_OPS[op]}({arg})")
            sel.append(f'{expr} AS "m{i}"')
        where, params = self._filters(filters, path, aliases)
        sql = f"SELECT {', '.join(sel)} FROM {frm}{where}"
        if groups:
            sql += " GROUP BY " + ", ".join(groups)
        if order_by:
            names = list(group_by or []) + [f"m{i}" for i in range(len(measures))]
            if order_by not in names:
                raise ValueError(f"order_by must be one of {names}")
            sql += f' ORDER BY "{order_by}"' + (" DESC" if desc else "")
        cap = self.max_rows if limit is None else max(1, min(int(limit), self.max_rows))
        cur = self.con.execute(sql + " LIMIT ?", params + [cap + 1])
        cols, rows = _rows(cur)
        return {"columns": cols, "rows": rows[:cap], "truncated": len(rows) > cap}

