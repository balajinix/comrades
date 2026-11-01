"""Async wrapper around the MCP streamable-http client.

The agent federates THREE MCP servers (see config.MCP_SERVERS):
  - mcp_kg (:8001) — graph & metadata tools (CEG, Federated Registry, ITOM,
    temporal-graph). Mostly single-NL-string `generate_*` / graph tools.
  - mcp_dal (:8002) — product & operational data: FinOps, APM,
    ITSM, Resilience, Infra. 33 finer-grained tools that take
    STRUCTURED JSON arguments (resource_type/operation/params, filters=[...],
    typed ITSM schemas, …).
  - mcp_var (:8003) — resilience metrics (`calculate_*`), single optional NL arg.

This module:

- `list_tools()` — federates `tools/list` across all configured servers into a
  single flat catalog. Each tool entry carries its `server` label, the URL that
  serves it, the raw `input_schema`, a token-compact `compact_schema` rendering
  for the system prompt, and `single_string_key` (set only for tools whose sole
  argument is a natural-language string — the bare-text fallback target). Tool
  dispatch follows `url`. Per-server failures are logged and non-fatal; if EVERY
  server fails we re-raise. Tool-name collisions across servers raise loudly.
- `call_tool()` — invokes one tool against the URL the catalog says serves it,
  passing a caller-built structured `arguments` dict. JSON-fallback parsing for
  FastMCP tools that return a dict via `content[0].text` instead of
  `structuredContent`.

Each call opens a fresh streamable-http session. All three servers run with
`stateless_http=True`, so per-call sessions are the right pattern; pooling is a
future optimisation only if the LLM round-trip stops dominating.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .config import MCP_SERVERS


logger = logging.getLogger("kg_agent.mcp_client")

# Routing/agent tools we never call from inside the agent — the agent IS the
# router. Including them would cause indirection (`kg → leaf`) or recursion.
TOOL_BLACKLIST: set[str] = {"kg", "ask_kg"}

# tools/list cache. The federated catalog changes on the order of weeks, not
# minutes. Hitting the network on every agent run is just overhead.
_TOOLS_CACHE_TTL_SEC = 300
_tools_cache: tuple[float, list[dict[str, Any]]] | None = None  # (expires_at, tools)

# Per-tool char cap on the compact-schema rendering. Keeps the prompt bounded
# even when a tool (e.g. APM smart routers) has a sprawling input schema —
# nested shapes are taught via the few-shot examples, not the schema dump.
_COMPACT_SCHEMA_CAP = 600


class ToolNameCollisionError(RuntimeError):
    """Raised when two configured MCP servers advertise the same tool name.

    By design this is a hard failure rather than a silent drop. Same-named
    tools across servers means the LLM's behaviour depends on iteration
    order, which is a debugging nightmare. The fix is local — rename one
    tool or namespace one server (e.g. `mcp_kg.foo` / `mcp_dal.foo`).
    """


def invalidate_tools_cache() -> None:
    """Drop the tools cache. Call from tests or after a known catalog change."""
    global _tools_cache
    _tools_cache = None


async def list_tools(
    servers: list[tuple[str, str]] | None = None,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Return the whitelisted tool catalog aggregated across all configured
    MCP servers. Each entry is:
        {name, description, input_schema, compact_schema,
         single_string_key, server, url}

    `servers` overrides the configured list (`config.MCP_SERVERS`); it is a
    list of `(server_label, url)` pairs. Useful for tests that want to point
    at a specific subset.

    Tool-name collisions across servers raise `ToolNameCollisionError` —
    fail-loud is a deliberate choice; see the class docstring.

    Per-server transport failures are logged and skipped — the agent runs
    with whatever tools are reachable. If EVERY server fails, the first
    exception is re-raised so callers can see why.

    Cached for `_TOOLS_CACHE_TTL_SEC` per process when `servers` is None.
    Custom server lists (e.g. from tests) always hit the wire.
    """
    global _tools_cache
    now = time.monotonic()
    use_cache = servers is None and not force_refresh
    if use_cache and _tools_cache is not None:
        expires_at, cached = _tools_cache
        if now < expires_at:
            return cached

    targets = servers if servers is not None else MCP_SERVERS
    out: list[dict[str, Any]] = []
    seen: dict[str, str] = {}  # tool name → URL of the server that owns it
    failures: list[tuple[str, Exception]] = []

    for server, url in targets:
        try:
            tools_from_target = await _fetch_tools_from(url, server)
        except Exception as exc:
            logger.warning("list_tools failed for %s (%s): %s", server, url, exc)
            failures.append((url, exc))
            continue

        for tool in tools_from_target:
            name = tool["name"]
            if name in seen:
                raise ToolNameCollisionError(
                    f"tool {name!r} is advertised by both {seen[name]} and "
                    f"{url}; rename one of them or namespace one server"
                )
            seen[name] = url
            out.append(tool)

    if not out and failures:
        # Every configured server failed — re-raise the first exception so
        # the caller can see the actual root cause, not a generic empty list.
        raise failures[0][1]

    if use_cache:
        _tools_cache = (now + _TOOLS_CACHE_TTL_SEC, out)
    return out


async def _fetch_tools_from(url: str, server: str) -> list[dict[str, Any]]:
    """Open one streamable-http session, list its tools, return whitelisted
    entries with the server label + URL + schema rendering stamped on each."""
    out: list[dict[str, Any]] = []
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            for t in resp.tools:
                if t.name in TOOL_BLACKLIST:
                    continue
                schema = getattr(t, "inputSchema", None) or {}
                out.append({
                    "name": t.name,
                    "description": (t.description or "").strip(),
                    "input_schema": schema,
                    "compact_schema": _compact_schema(schema),
                    "single_string_key": _pick_primary_string_input_if_sole(schema),
                    "server": server,
                    "url": url,
                })
    return out


async def call_tool(
    name: str,
    arguments: dict[str, Any],
    url: str,
) -> dict[str, Any]:
    """Invoke one tool against the URL the catalog says serves it.

    `arguments` is the full structured argument object the caller built from
    the LLM's JSON Action Input (see prompts.parse_action_input). `url` is
    required — the caller looks it up from `list_tools()` output rather than
    relying on a default, so federated calls always land on the right server.

    Returns the structured content as a dict. Transport-level failures
    raise; tool-level errors come back as `{status: "error", ...}` for the
    caller's react loop to handle.
    """
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            if result.structuredContent is not None:
                return _unwrap_result(result.structuredContent)
            # FastMCP returns dict-shaped tool results as JSON-encoded text in
            # content[0].text rather than structuredContent. Parse it back into
            # a dict so the synthesizer/observation can read the fields. Fall
            # through to raw text wrapping for non-JSON / non-dict cases.
            text_parts = [c.text for c in result.content if hasattr(c, "text")]
            joined = "\n".join(text_parts)
            try:
                parsed = json.loads(joined)
                if isinstance(parsed, dict):
                    return _unwrap_result(parsed)
            except (ValueError, TypeError):
                pass
            return {"content": joined}


def _unwrap_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap FastMCP's `{"result": "<json string>"}` envelope.

    Tools that return a JSON *string* (notably itsm-mcp's service_desk
    tools) get wrapped by FastMCP as `{"result": "<json text>"}`. Parse that
    inner string back into its dict so the synthesizer's renderers and the
    observation formatter see the real shape (e.g. `{success, incidents:[...]}`)
    instead of an opaque string — otherwise ITSM results render as bare
    markdown and the agent can't cite incident numbers. Any other shape (a
    multi-key dict, a non-JSON `result` string, a non-dict inner value) is
    returned untouched.
    """
    if (isinstance(payload, dict) and list(payload) == ["result"]
            and isinstance(payload["result"], str)):
        try:
            inner = json.loads(payload["result"])
        except (ValueError, TypeError):
            return payload
        if isinstance(inner, dict):
            return inner
    return payload


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def _pick_primary_string_input_if_sole(schema: dict[str, Any]) -> str | None:
    """Return the single string argument's key IFF the tool's input is
    effectively one natural-language string — otherwise None.

    "Effectively one" means: exactly one property, OR exactly one required
    property (and it's string-typed). This is the dominant shape for mcp_var's
    `calculate_*` tools (single optional NL string), mcp_kg's `generate_*`
    tools (single NL query), and `query_infra_nl2iql`
    (`natural_language_query`). For these, the loop accepts bare text and wraps
    it as `{key: text}` (back-compat with the old single-string call model).

    Returns None for multi-arg / structured tools (most of mcp_dal) so the
    loop NEVER silently funnels a structured tool's input into one field — it
    requires real JSON and coaches the LLM on malformed input instead.
    """
    props: dict[str, Any] = schema.get("properties") or {}
    required = schema.get("required") or []

    def is_string_prop(spec: Any) -> bool:
        if not isinstance(spec, dict):
            return False
        if spec.get("type") == "string":
            return True
        # `Optional[str]` → anyOf [{type:string},{type:null}]
        any_of = spec.get("anyOf")
        if isinstance(any_of, list):
            return any(isinstance(b, dict) and b.get("type") == "string" for b in any_of)
        return False

    # Exactly one required string property → that's the NL arg.
    if len(required) == 1 and is_string_prop(props.get(required[0])):
        return required[0]

    # No required props but exactly one property total, and it's a string.
    if not required and len(props) == 1:
        only = next(iter(props))
        if is_string_prop(props[only]):
            return only

    return None


def _type_token(spec: Any) -> str:
    """Compact JSONSchema → short type token for the prompt rendering."""
    if not isinstance(spec, dict):
        return "any"

    enum = spec.get("enum")
    if isinstance(enum, list) and enum:
        vals = [str(v) for v in enum[:6]]
        suffix = "|…" if len(enum) > 6 else ""
        return f"enum[{'|'.join(vals)}{suffix}]"

    # Optional[T] / unions → render the first non-null branch with a trailing ?.
    any_of = spec.get("anyOf") or spec.get("oneOf")
    if isinstance(any_of, list):
        non_null = [b for b in any_of if isinstance(b, dict) and b.get("type") != "null"]
        has_null = any(isinstance(b, dict) and b.get("type") == "null" for b in any_of)
        if non_null:
            inner = _type_token(non_null[0])
            return f"{inner}?" if has_null else inner
        return "any"

    t = spec.get("type")
    if t == "array":
        items = spec.get("items")
        return f"{_type_token(items)}[]" if items else "list"
    if isinstance(t, list):
        return "|".join(x for x in t if x != "null")
    return t or "object"


def _compact_schema(schema: dict[str, Any]) -> str:
    """Render a tool's input schema compactly for the system prompt.

    One line per top-level property: `  name (type)[*]: <desc ≤80 chars>`,
    preceded by a `required:` header. Depth is capped at 1 — nested object
    properties (APM `params` / `tag_filter_expression`, Resilience `payload`)
    are NOT expanded; their shape is taught via the few-shot examples. The
    whole block is capped at `_COMPACT_SCHEMA_CAP` chars. Returns a no-args
    marker for argument-less tools.
    """
    props: dict[str, Any] = schema.get("properties") or {}
    if not props:
        return "  (no args — pass {})"

    required = list(schema.get("required") or [])
    lines: list[str] = []
    if required:
        lines.append(f"  required: {', '.join(required)}")

    # Required props first (in declaration order), then the rest.
    ordered = [n for n in props if n in required] + [n for n in props if n not in required]
    for name in ordered:
        spec = props[name]
        star = "*" if name in required else ""
        token = _type_token(spec)
        desc = ""
        if isinstance(spec, dict):
            raw_desc = (spec.get("description") or "").strip().replace("\n", " ")
            if raw_desc:
                desc = ": " + (raw_desc[:80] + "…" if len(raw_desc) > 80 else raw_desc)
        line = f"  {name} ({token}){star}{desc}"
        if token.startswith("object"):
            line += "  # shape varies — see example"
        lines.append(line)

    rendered = "\n".join(lines)
    if len(rendered) > _COMPACT_SCHEMA_CAP:
        kept = rendered[:_COMPACT_SCHEMA_CAP].rstrip()
        rendered = kept + "\n  …(more optional args omitted)"
    return rendered
