"""Metering gateway: counts what every tool call costs in communication.

Every facade tool is wrapped by a :class:`Meter`, the single measurement boundary
of the lab (the in-process analogue of COMRADES' fault proxy). Per call it records
request and response bytes, rows returned, latency and tokens.

Tokens are counted with tiktoken's ``cl100k_base`` encoding. That is a *proxy* for
model-specific tokenizers (Llama, Granite, Claude and GPT tokenizers differ by
roughly 10-30% on JSON); relative comparisons between modalities are what matter,
and every modality is counted with the same encoder.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def to_json(obj: Any) -> str:
    """Serialise a tool payload the way an MCP/HTTP tool would put it on the wire.

    Args:
        obj: Any JSON-like value (dates and decimals are stringified / floated).

    Returns:
        Compact JSON text.
    """
    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        return str(o)

    return json.dumps(obj, default=_default, separators=(",", ":"))


def count_tokens(text: str) -> int:
    """Count cl100k_base tokens in ``text``.

    Args:
        text: Any string.

    Returns:
        Token count.
    """
    return len(_ENC.encode(text, disallowed_special=()))


def tool_description_tokens(tools: list[dict]) -> int:
    """Tokens needed to deliver tool descriptions (name, description, schema).

    This is the per-turn schema-delivery cost of an MCP-style interface: the tool
    list is part of every prompt.

    Args:
        tools: MCP-style tool descriptors.

    Returns:
        Token count of the serialised list.
    """
    return count_tokens(to_json(tools))


@dataclass
class CallRecord:
    """One metered tool call."""

    tool: str
    request_bytes: int
    response_bytes: int
    request_tokens: int
    response_tokens: int
    rows: int
    latency_s: float
    db_mem_bytes: int = 0         # DuckDB memory held after the call
    db_temp_table_bytes: int = 0  # of which in-memory temp tables (materialised handles)
    db_spill_bytes: int = 0       # temporary storage spilled to disk


def duckdb_probe(con: Any) -> Callable[[], dict[str, int]]:
    """A probe reading DuckDB's memory accounting after each tool call.

    DuckDB reports current usage per tag, not a peak, so this samples the state a
    call leaves behind: enough to see what materialised handles keep resident. The
    values are growth over the state when the probe was made (one probe per run),
    so earlier runs on the same connection are not counted.
    """
    def read() -> dict[str, int]:
        rows = con.execute("SELECT tag, memory_usage_bytes, temporary_storage_bytes "
                           "FROM duckdb_memory()").fetchall()
        return {"db_mem_bytes": sum(r[1] for r in rows),
                "db_temp_table_bytes": sum(r[1] for r in rows if r[0] == "IN_MEMORY_TABLE"),
                "db_spill_bytes": sum(r[2] for r in rows)}

    base = read()

    def probe() -> dict[str, int]:
        return {k: max(0, v - base[k]) for k, v in read().items()}
    return probe


def count_rows(delivered: Any) -> int:
    """Rows a tool result puts in front of the model.

    Tabular results carry ``rows`` (or a handle's ``preview``); a GraphQL result
    carries nested ``data``, where every object inside a list is one row shipped.

    Args:
        delivered: The observation as delivered.

    Returns:
        Number of rows.
    """
    if not isinstance(delivered, dict):
        return 0
    for key in ("rows", "preview"):
        if isinstance(delivered.get(key), list):
            return len(delivered[key])

    def walk(node: Any) -> int:
        if isinstance(node, list):
            return sum(1 + walk(x) if isinstance(x, dict) else walk(x) for x in node)
        if isinstance(node, dict):
            return sum(walk(v) for v in node.values())
        return 0

    return walk(delivered.get("data"))

@dataclass
class Meter:
    """Records every call made through wrapped tools."""

    records: list[CallRecord] = field(default_factory=list)
    db_probe: Callable[[], dict[str, int]] | None = None

    def wrap(self, name: str, fn: Callable[..., Any],
             observe: Callable[[Any], Any] | None = None) -> Callable[..., Any]:
        """Wrap a tool so each call is metered.

        Args:
            name: Tool name.
            fn: The tool implementation, called with keyword arguments.
            observe: Optional transform applied to the result before it is
                delivered (e.g. observation compaction). Metering counts what is
                delivered, since that is what enters the model's context.

        Returns:
            A callable with the same keyword interface that returns the delivered
            payload.
        """
        def metered(**kwargs: Any) -> Any:
            req = to_json({"tool": name, "arguments": kwargs})
            start = time.perf_counter()
            result = fn(**kwargs)
            latency = time.perf_counter() - start
            delivered = observe(result) if observe else result
            resp = to_json(delivered)
            rows = count_rows(delivered)
            db = self.db_probe() if self.db_probe else {}
            self.records.append(CallRecord(
                tool=name, request_bytes=len(req.encode()),
                response_bytes=len(resp.encode()),
                request_tokens=count_tokens(req), response_tokens=count_tokens(resp),
                rows=rows, latency_s=latency, **db))
            return delivered

        return metered

    def totals(self) -> dict[str, float]:
        """Aggregate the recorded calls.

        Returns:
            Dict with calls, bytes, tokens, rows and latency totals.
        """
        return {
            "calls": len(self.records),
            "request_bytes": sum(r.request_bytes for r in self.records),
            "response_bytes": sum(r.response_bytes for r in self.records),
            "request_tokens": sum(r.request_tokens for r in self.records),
            "response_tokens": sum(r.response_tokens for r in self.records),
            "rows": sum(r.rows for r in self.records),
            "latency_s": sum(r.latency_s for r in self.records),
            "db_mem_peak_bytes": max((r.db_mem_bytes for r in self.records), default=0),
            "db_temp_table_peak_bytes": max((r.db_temp_table_bytes for r in self.records),
                                            default=0),
            "db_spill_peak_bytes": max((r.db_spill_bytes for r in self.records),
                                       default=0),
        }

    def as_dicts(self) -> list[dict]:
        """Return the call log as plain dicts (for traces)."""
        return [asdict(r) for r in self.records]
