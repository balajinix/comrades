"""Scripted (deterministic, no-LLM) agents, one per data-access modality.

Each policy answers a :class:`~dal_lab.workload.Question` the way a *competent*
agent would through one facade, and accounts for the context a ReAct-style model
would carry:

- turn 0 input = system prompt + tool descriptions + question;
- every tool call appends its request and its (delivered) observation;
- one model turn per call plus a final answer turn; cumulative input tokens is
  the sum of the context over all turns (no prompt caching; see ``costmodel``).

The policies isolate the *interface* cost. They assume the model always writes
the right query or plan (an upper bound on model skill), so measured differences
come from the modality alone. The LLM pilot measures the gap to real models.

- ``generic``: one SQL statement (the gold query), paging only if the result
  exceeds the row cap.
- ``fine_grained``: executes the question's pushdown ``plan`` with per-table list
  tools (semi-joins batched 100 keys per call, 1000-row pages), then joins and
  aggregates the fetched rows "in context", which is simulated by running the gold
  SQL over the fetched rows. Every fetched byte counts as context.
- ``handles``: the gold query as a handle; results of <= ``inline_rows`` rows are
  fetched into context, larger ones are answered by reference.
- ``workflow``: one call to the coded intent, or fails if none exists.
- ``workflow+sql``: the coded intent if one exists, otherwise the ``generic`` policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import duckdb

from .adaptive import AutoHandleFacade, OnDemandSchemaFacade, router_facade
from .capability import FineGrainedOpsFacade
from .code_mode import CodeFacade
from .facades import (Catalog, CompactedFacade, Facade, FineGrainedFacade, GenericFacade,
                      HandleFacade, HybridFacade, QueryAnswerFacade, WorkflowFacade)
from .graphql_facade import GraphQLFacade
from .meter import Meter, count_tokens, to_json, tool_description_tokens
from .skills import SkillsFacade
from .workload import WORKFLOWS, Question, gold_rows, same_answer

SYSTEM_PROMPT = (
    "You are a data agent. Answer the user's question using only the tools "
    "provided. Call tools as needed, then give the final answer.")
SYSTEM_TOKENS = count_tokens(SYSTEM_PROMPT)
DEFAULT_BUDGET = 128_000
IN_BATCH = 100
PAGE = 1000


@dataclass
class RunResult:
    """Outcome and communication cost of one (question, facade) run."""

    qid: str
    family: str
    facade: str
    status: str  # correct | wrong | overflow | no_workflow | error
    calls: int = 0
    turns: int = 0
    schema_tokens: int = 0
    base_tokens: int = 0
    context_peak_tokens: int = 0
    cumulative_input_tokens: int = 0
    output_tokens: int = 0
    response_bytes: int = 0
    rows_moved: int = 0
    db_latency_s: float = 0.0
    by_reference: bool = False
    detail: str = ""
    trace: list[dict] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        """Whether the answer matched gold."""
        return self.status == "correct"


class Overflow(Exception):
    """Raised when the context exceeds the budget."""


class _Ctx:
    """Context accounting for one simulated ReAct run."""

    def __init__(self, base: int, budget: int) -> None:
        self.base, self.budget = base, budget
        self.size = base
        self.cumulative = base  # input of turn 0
        self.output = 0

    def step(self, rec: Any) -> None:
        self.size += rec.request_tokens + rec.response_tokens
        self.output += rec.request_tokens
        if self.size > self.budget:
            raise Overflow
        self.cumulative += self.size  # input of the next turn


class Lab:
    """All facades over one store, plus the scripted policies."""

    def __init__(self, con: duckdb.DuckDBPyConnection, compact_limit: int = 4000,
                 workflows: dict[str, dict] | None = None) -> None:
        self.con = con
        workflows = WORKFLOWS if workflows is None else workflows
        self.catalog = Catalog.load(con)
        generic = GenericFacade(con, self.catalog)
        fine = FineGrainedFacade(con, self.catalog)
        self.handle_facade = HandleFacade(con, self.catalog)
        self.facades: dict[str, Facade] = {
            "generic": generic,
            "fine_grained": fine,
            "handles": self.handle_facade,
            "workflow": WorkflowFacade(con, workflows),
            "workflow+sql": HybridFacade(WorkflowFacade(con, workflows), generic,
                                         "workflow+sql"),
            "generic+compact": CompactedFacade(generic, compact_limit),
            "fine_grained+compact": CompactedFacade(fine, compact_limit),
            "graphql": GraphQLFacade(con, self.catalog),
            "graphql+agg": GraphQLFacade(con, self.catalog, with_aggregates=True),
            "skills": SkillsFacade(con, self.catalog),
            "code": CodeFacade(con, self.catalog, self.handle_facade),
            "code+prov": CodeFacade(con, self.catalog, self.handle_facade,
                                    provenance=True),
            "fine_grained+ops": FineGrainedOpsFacade(con, self.catalog),
            "sql+auto": AutoHandleFacade(con, self.catalog, self.handle_facade),
            "sql+answer_query": QueryAnswerFacade(con, self.catalog),
            "sql+silent": GenericFacade(con, self.catalog, truncation="silent"),
            "sql+labelled": GenericFacade(con, self.catalog, truncation="labelled"),
            "handles+lazy": HandleFacade(con, self.catalog, lazy=True),
            "sql+schema": OnDemandSchemaFacade(con, self.catalog, generic),
            "router": router_facade(con, self.catalog, self.handle_facade,
                                    WorkflowFacade(con, workflows)),
        }
        # Facades with a scripted policy (the analytic sweep); the rest are LLM-only.
        self.scripted = [f for f in self.facades
                         if not f.startswith(("graphql", "skills", "code", "fine_grained+ops",
                                              "sql+", "router", "handles+lazy"))]

    def run(self, q: Question, facade: str, budget: int = DEFAULT_BUDGET,
            inline_rows: int = 50) -> RunResult:
        """Answer ``q`` through ``facade`` and measure the cost.

        Args:
            q: The question.
            facade: One of ``self.facades``.
            budget: Context budget in tokens.
            inline_rows: Handles policy answers results up to this size in context.

        Returns:
            The run result.
        """
        fac = self.facades[facade]
        meter = Meter()
        tools = fac.bind(meter)
        schema = tool_description_tokens(fac.descriptors)
        base = SYSTEM_TOKENS + schema + count_tokens(q.question)
        ctx = _Ctx(base, budget)
        res = RunResult(q.id, q.family, facade, "error", schema_tokens=schema,
                        base_tokens=base)
        kind = facade.split("+")[0]
        answer: list[tuple] | None = None
        try:
            if kind == "generic":
                answer = self._generic(q, tools, meter, ctx)
            elif kind == "fine_grained":
                answer = self._fine(q, tools, meter, ctx)
            elif kind == "handles":
                answer = self._handles(q, tools, meter, ctx, inline_rows, res)
            elif kind == "workflow":
                answer = self._workflow(q, tools, meter, ctx)
                if answer is None and facade == "workflow+sql":
                    answer = self._generic(q, tools, meter, ctx)
            if answer is None and not res.by_reference:
                res.status = "no_workflow"
            else:
                gold = gold_rows(self.con, q)
                got = answer if answer is not None else gold  # by reference: checked below
                ok = same_answer(got, gold, q.ordered)
                res.status = "correct" if ok else "wrong"
                if not res.by_reference:
                    ctx.output += count_tokens(to_json([list(r) for r in got]))
                else:
                    ctx.output += 30  # "the result is handle hN (n rows)" style answer
        except Overflow:
            res.status = "overflow"
        except Exception as exc:  # noqa: BLE001 - record and continue the sweep
            res.status, res.detail = "error", f"{type(exc).__name__}: {exc}"
        t = meter.totals()
        res.calls, res.turns = int(t["calls"]), int(t["calls"]) + 1
        res.context_peak_tokens = ctx.size
        res.cumulative_input_tokens = ctx.cumulative
        res.output_tokens = ctx.output
        res.response_bytes, res.rows_moved = int(t["response_bytes"]), int(t["rows"])
        res.db_latency_s = t["latency_s"]
        res.trace = meter.as_dicts()
        return res

    # --- policies -----------------------------------------------------------------

    @staticmethod
    def _call(tools: dict, meter: Meter, ctx: _Ctx, name: str, **kw: Any) -> Any:
        out = tools[name](**kw)
        ctx.step(meter.records[-1])
        return out

    def _generic(self, q: Question, tools: dict, meter: Meter, ctx: _Ctx) -> list[tuple]:
        out = self._call(tools, meter, ctx, "run_sql", query=q.gold_sql)
        if not out.get("truncated"):
            return [tuple(r.values()) for r in out["rows"]]
        rows: list[tuple] = []
        offset = 0
        while True:  # re-issue as stable pages
            page = self._call(tools, meter, ctx, "run_sql",
                              query=f"SELECT * FROM ({q.gold_sql}) t ORDER BY ALL "
                                    f"LIMIT {PAGE} OFFSET {offset}")
            rows += [tuple(r.values()) for r in page["rows"]]
            if not page.get("truncated") and page["row_count"] < PAGE:
                return rows
            offset += PAGE

    def _fine(self, q: Question, tools: dict, meter: Meter, ctx: _Ctx) -> list[tuple]:
        local = duckdb.connect()
        for table, cols in self.catalog.tables.items():
            local.execute(f'CREATE TABLE "{table}" ('
                          + ", ".join(f'"{c}" {t}' for c, t in cols) + ")")
        for step in q.plan:
            table = step["table"]
            plain: dict = {}
            semi: tuple[str, list] | None = None
            for col, spec in step["filters"].items():
                if isinstance(spec, dict) and "in_from" in spec:
                    src_t, src_c = spec["in_from"].split(".")
                    vals = [r[0] for r in local.execute(
                        f'SELECT DISTINCT "{src_c}" FROM "{src_t}" ORDER BY 1').fetchall()]
                    semi = (col, vals)
                else:
                    plain[col] = spec
            batches = ([None] if semi is None else
                       [semi[1][i:i + IN_BATCH] for i in range(0, len(semi[1]), IN_BATCH)])
            cols = self.catalog.columns(table)
            for batch in batches:
                filters = dict(plain)
                if batch is not None:
                    filters[semi[0]] = {"in": batch}
                offset = 0
                while True:
                    page = self._call(tools, meter, ctx, f"list_{table}",
                                      filters=filters, limit=PAGE, offset=offset)
                    if page["rows"]:
                        local.executemany(
                            f'INSERT INTO "{table}" VALUES ({", ".join("?" * len(cols))})',
                            [[r.get(c) for c in cols] for r in page["rows"]])
                    if page.get("next_offset") is None:
                        break
                    offset = page["next_offset"]
        return local.execute(q.gold_sql).fetchall()

    def _handles(self, q: Question, tools: dict, meter: Meter, ctx: _Ctx,
                 inline_rows: int, res: RunResult) -> list[tuple] | None:
        out = self._call(tools, meter, ctx, "run_query", query=q.gold_sql)
        n = out["row_count"]
        if n <= len(out["preview"]):
            return [tuple(r.values()) for r in out["preview"]]
        if n <= inline_rows:
            page = self._call(tools, meter, ctx, "fetch_rows", handle=out["handle"],
                              offset=0, limit=n)
            return [tuple(r.values()) for r in page["rows"]]
        res.by_reference = True
        table = self.handle_facade.table_of(out["handle"])
        held = self.con.execute(f"SELECT * FROM {table}").fetchall()
        if not same_answer(held, gold_rows(self.con, q), q.ordered):
            raise ValueError("handle contents differ from gold")
        return None

    def _workflow(self, q: Question, tools: dict, meter: Meter,
                  ctx: _Ctx) -> list[tuple] | None:
        if not q.workflow:
            return None
        out = self._call(tools, meter, ctx, q.workflow, **q.workflow_params)
        return [tuple(r.values()) for r in out["rows"]]


def plan_rows(con: duckdb.DuckDBPyConnection, q: Question) -> dict[str, int]:
    """Rows the fine-grained plan must ship, per step, computed directly in SQL.

    Used to estimate the context a fine-grained run *would* need when it overflows.

    Args:
        con: Store connection.
        q: The question.

    Returns:
        ``{"<step index>:<table>": rows}``.
    """
    wheres: dict[str, str] = {}
    out: dict[str, int] = {}
    for i, step in enumerate(q.plan):
        table = step["table"]
        clauses = []
        for col, spec in step["filters"].items():
            items = spec.items() if isinstance(spec, dict) else [("eq", spec)]
            for op, val in items:
                if op == "in_from":
                    st, sc = val.split(".")
                    clauses.append(f'"{col}" IN (SELECT "{sc}" FROM "{st}"{wheres.get(st, "")})')
                else:
                    sym = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[op]
                    lit = f"'{val}'" if isinstance(val, str) else str(val)
                    clauses.append(f'"{col}" {sym} {lit}')
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        wheres[table] = where
        out[f"{i}:{table}"] = con.execute(f'SELECT count(*) FROM "{table}"{where}').fetchone()[0]
    return out


def tokens_per_row(con: duckdb.DuckDBPyConnection, table: str, sample: int = 200) -> float:
    """Average delivered tokens per row of ``table`` in the list-tool JSON format.

    Args:
        con: Store connection.
        table: Table name.
        sample: Rows to sample.

    Returns:
        Tokens per row.
    """
    cur = con.execute(f'SELECT * FROM "{table}" LIMIT {sample}')
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return count_tokens(to_json(rows)) / max(1, len(rows))
