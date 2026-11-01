"""A GraphQL interface generated from the store, as GraphQL gateways expose databases.

One object type per table, scalar fields per column, and relationship fields from
foreign keys (``customer.orders``, ``orders.customer``). Root fields list a table
with a ``where`` filter and paging, or fetch one row by key. The model sends a query
string; nested selections fetch related rows in one call (projection and join
pushdown), but plain GraphQL has no aggregation, so counts, sums and top-k over
groups must be computed by the model from shipped rows. ``with_aggregates=True``
adds Hasura-style ``<table>_aggregate`` root fields, the common extension that
pushes aggregation down too.

The schema reaches the model as SDL in the tool description, which is what
introspection gives a GraphQL client.
"""
from __future__ import annotations

from typing import Any, Callable

import duckdb
from graphql import (GraphQLArgument, GraphQLBoolean, GraphQLField, GraphQLFloat,
                     GraphQLInputField, GraphQLInputObjectType, GraphQLInt, GraphQLList,
                     GraphQLNonNull, GraphQLObjectType, GraphQLScalarType, GraphQLSchema,
                     GraphQLString, graphql_sync, print_schema)

from .facades import _OPS, MAX_ROWS, Catalog, Facade
from .obfuscate import COLUMN_MAP, TABLE_MAP

# (child table, child column) -> (parent table, parent column), in plain TPC-H names.
TPCH_FOREIGN_KEYS = {
    ("nation", "n_regionkey"): ("region", "r_regionkey"),
    ("supplier", "s_nationkey"): ("nation", "n_nationkey"),
    ("customer", "c_nationkey"): ("nation", "n_nationkey"),
    ("partsupp", "ps_partkey"): ("part", "p_partkey"),
    ("partsupp", "ps_suppkey"): ("supplier", "s_suppkey"),
    ("orders", "o_custkey"): ("customer", "c_custkey"),
    ("lineitem", "l_orderkey"): ("orders", "o_orderkey"),
    ("lineitem", "l_partkey"): ("part", "p_partkey"),
    ("lineitem", "l_suppkey"): ("supplier", "s_suppkey"),
}

DEFAULT_LIMIT = 100


def foreign_keys(catalog: Catalog) -> dict[tuple[str, str], tuple[str, str]]:
    """Foreign keys for the attached store: TPC-DS, or plain or renamed TPC-H.

    Args:
        catalog: The store's catalog.

    Returns:
        (child table, child column) -> (parent table, parent column).
    """
    if "store_sales" in catalog.tables:
        from .tpcds import FOREIGN_KEYS  # noqa: PLC0415 - optional dataset

        return dict(FOREIGN_KEYS)
    if "span" in catalog.tables and "k8s_object" in catalog.tables:
        from .itbench import FOREIGN_KEYS as ITB  # noqa: PLC0415 - optional dataset

        return dict(ITB)
    if "lineitem" in catalog.tables:
        return dict(TPCH_FOREIGN_KEYS)
    return {(TABLE_MAP[ct], COLUMN_MAP[cc]): (TABLE_MAP[pt], COLUMN_MAP[pc])
            for (ct, cc), (pt, pc) in TPCH_FOREIGN_KEYS.items()}


def relation_names(fks: dict[tuple[str, str], tuple[str, str]], child: str,
                   child_col: str, parent: str) -> tuple[str, str]:
    """Field names for one foreign key: (object field on child, list field on parent).

    With a single key from ``child`` to ``parent`` the fields are named after the
    tables (``orders.customer``, ``customer.orders``). When a child references the
    same parent twice (a sale's sold date and ship date), the names come from the key
    column instead (``cs_ship_date``, ``catalog_sales_via_cs_ship_date``).
    """
    if sum(1 for (ct, _), (pt, _) in fks.items() if ct == child and pt == parent) == 1:
        return parent, child
    stem = child_col[:-3] if child_col.endswith("_sk") else child_col
    return stem, f"{child}_via_{stem}"


def _scalar(duck_type: str) -> GraphQLScalarType:
    t = duck_type.upper()
    if t in {"INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT"}:
        return GraphQLInt
    if t.startswith(("DECIMAL", "DOUBLE", "FLOAT", "REAL", "NUMERIC")):
        return GraphQLFloat
    if t == "BOOLEAN":
        return GraphQLBoolean
    return GraphQLString  # VARCHAR, DATE (ISO text), others


def _type_name(table: str) -> str:
    return "".join(p.capitalize() for p in table.split("_"))


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()[:10]
    return float(v)  # Decimal


class GraphQLFacade(Facade):
    """One ``graphql`` tool over a schema generated from the store."""

    def __init__(self, con: duckdb.DuckDBPyConnection, catalog: Catalog,
                 with_aggregates: bool = False, max_rows: int = MAX_ROWS) -> None:
        super().__init__("graphql+agg" if with_aggregates else "graphql")
        self.con, self.catalog, self.max_rows = con, catalog, max_rows
        self.fks = foreign_keys(catalog)
        self.schema = self._build(with_aggregates)
        self.sdl = print_schema(self.schema)
        note = ("Lists return at most `limit` rows (default 100, max 1000); page with "
                "`offset`. ")
        note += ("`<table>_aggregate` computes count/sum/avg/min/max, optionally "
                 "grouped. " if with_aggregates else
                 "There is no aggregation: fetch the rows you need and compute. ")
        self._add("graphql", self.execute,
                  "Run a GraphQL query against the data API. " + note +
                  "Schema (SDL):\n" + self.sdl,
                  {"query": {"type": "string"}}, ["query"])

    # -- schema -----------------------------------------------------------------
    def _build(self, with_aggregates: bool) -> GraphQLSchema:
        filters = {s.name: GraphQLInputObjectType(f"{s.name}Filter", {
            **{op: GraphQLInputField(s) for op in _OPS},
            "in": GraphQLInputField(GraphQLList(GraphQLNonNull(s))),
            **({"like": GraphQLInputField(GraphQLString)} if s is GraphQLString else {}),
        }) for s in (GraphQLInt, GraphQLFloat, GraphQLString, GraphQLBoolean)}
        where_types = {
            t: GraphQLInputObjectType(f"{_type_name(t)}Where", {
                c: GraphQLInputField(filters[_scalar(ty).name]) for c, ty in cols})
            for t, cols in self.catalog.tables.items()}
        objects: dict[str, GraphQLObjectType] = {}

        def list_args(table: str) -> dict[str, GraphQLArgument]:
            return {"where": GraphQLArgument(where_types[table]),
                    "limit": GraphQLArgument(GraphQLInt),
                    "offset": GraphQLArgument(GraphQLInt)}

        def fields_of(table: str) -> Callable[[], dict[str, GraphQLField]]:
            def thunk() -> dict[str, GraphQLField]:
                out = {c: GraphQLField(_scalar(ty),
                                       description=self.catalog.note(table, c))
                       for c, ty in self.catalog.tables[table]}
                for (ct, cc), (pt, pc) in self.fks.items():
                    obj_name, list_name = relation_names(self.fks, ct, cc, pt)
                    if ct == table:  # many-to-one: the parent object
                        out[obj_name] = GraphQLField(objects[pt],
                                                     resolve=self._parent(pt, pc, cc))
                    if pt == table:  # one-to-many: the child list
                        out[list_name] = GraphQLField(
                            GraphQLList(objects[ct]), args=list_args(ct),
                            resolve=self._children(ct, cc, pc))
                return out
            return thunk

        for t in self.catalog.tables:
            objects[t] = GraphQLObjectType(_type_name(t), fields_of(t))
        root: dict[str, GraphQLField] = {}
        for t in self.catalog.tables:
            key = self.catalog.key(t)
            key_type = _scalar(dict(self.catalog.tables[t])[key])
            root[t] = GraphQLField(GraphQLList(objects[t]), args=list_args(t),
                                   resolve=self._root_list(t))
            root[f"{t}_by_key"] = GraphQLField(
                objects[t], args={"key": GraphQLArgument(GraphQLNonNull(key_type))},
                resolve=self._by_key(t, key))
            if with_aggregates:
                root[f"{t}_aggregate"] = GraphQLField(
                    GraphQLList(GraphQLObjectType(f"{_type_name(t)}AggregateRow", {
                        "group": GraphQLField(GraphQLList(GraphQLString)),
                        "value": GraphQLField(GraphQLFloat)})),
                    args={"where": GraphQLArgument(where_types[t]),
                          "op": GraphQLArgument(GraphQLNonNull(GraphQLString)),
                          "column": GraphQLArgument(GraphQLString),
                          "group_by": GraphQLArgument(GraphQLList(GraphQLNonNull(
                              GraphQLString)))},
                    resolve=self._aggregate(t))
        return GraphQLSchema(GraphQLObjectType("Query", root))

    # -- resolvers --------------------------------------------------------------
    def _where(self, table: str, where: dict | None) -> tuple[str, list]:
        clauses, params = [], []
        for col, ops in (where or {}).items():
            for op, val in ops.items():
                if op == "in":
                    if not val:
                        clauses.append("FALSE")
                        continue
                    clauses.append(f'"{col}" IN ({", ".join("?" * len(val))})')
                    params.extend(val)
                elif op == "like":
                    clauses.append(f'"{col}" LIKE ?')
                    params.append(val)
                else:
                    clauses.append(f'"{col}" {_OPS[op]} ?')
                    params.append(val)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    def _select(self, table: str, where: str, params: list, limit: int | None,
                offset: int | None) -> list[dict]:
        n = max(1, min(int(limit or DEFAULT_LIMIT), self.max_rows))
        cur = self.con.execute(
            f'SELECT * FROM "{table}"{where} ORDER BY rowid LIMIT ? OFFSET ?',
            params + [n, int(offset or 0)])
        cols = [d[0] for d in cur.description]
        return [{c: _jsonable(v) for c, v in zip(cols, r)} for r in cur.fetchall()]

    def _root_list(self, table: str) -> Callable[..., list[dict]]:
        def resolve(_obj: Any, _info: Any, where: dict | None = None,
                    limit: int | None = None, offset: int | None = None) -> list[dict]:
            w, p = self._where(table, where)
            return self._select(table, w, p, limit, offset)
        return resolve

    def _by_key(self, table: str, key_col: str) -> Callable[..., dict | None]:
        def resolve(_obj: Any, _info: Any, key: Any) -> dict | None:
            rows = self._select(table, f' WHERE "{key_col}" = ?', [key], 1, 0)
            return rows[0] if rows else None
        return resolve

    def _parent(self, parent: str, parent_col: str,
                child_col: str) -> Callable[..., dict | None]:
        def resolve(obj: dict, _info: Any) -> dict | None:
            rows = self._select(parent, f' WHERE "{parent_col}" = ?', [obj[child_col]], 1, 0)
            return rows[0] if rows else None
        return resolve

    def _children(self, child: str, child_col: str,
                  parent_col: str) -> Callable[..., list[dict]]:
        def resolve(obj: dict, _info: Any, where: dict | None = None,
                    limit: int | None = None, offset: int | None = None) -> list[dict]:
            w, p = self._where(child, where)
            w = (w + " AND " if w else " WHERE ") + f'"{child_col}" = ?'
            return self._select(child, w, p + [obj[parent_col]], limit, offset)
        return resolve

    def _aggregate(self, table: str) -> Callable[..., list[dict]]:
        def resolve(_obj: Any, _info: Any, op: str, column: str | None = None,
                    group_by: list[str] | None = None,
                    where: dict | None = None) -> list[dict]:
            if op not in {"count", "sum", "avg", "min", "max"}:
                raise ValueError(f"unsupported op {op}")
            cols = set(self.catalog.columns(table))
            groups = list(group_by or [])
            if (column and column not in cols) or not set(groups) <= cols:
                raise ValueError("unknown column")
            arg = f'"{column}"' if column else "*"
            w, p = self._where(table, where)
            g = ", ".join(f'"{c}"' for c in groups)
            sql = (f"SELECT {g + ', ' if g else ''}{op}({arg}) FROM \"{table}\"{w}"
                   f"{' GROUP BY ' + g + ' ORDER BY ' + g if g else ''}")
            return [{"group": [str(_jsonable(v)) for v in r[:-1]],
                     "value": None if r[-1] is None else float(r[-1])}
                    for r in self.con.execute(sql, p).fetchall()]
        return resolve

    # -- the tool ---------------------------------------------------------------
    def execute(self, query: str) -> dict:
        """Execute a GraphQL query.

        Args:
            query: GraphQL query text.

        Returns:
            ``{"data": ...}`` and, on failure, ``{"errors": [...]}`` as a GraphQL
            server would return them.
        """
        res = graphql_sync(self.schema, query)
        out: dict[str, Any] = {"data": res.data}
        if res.errors:
            out["errors"] = [e.message for e in res.errors]
        return out
