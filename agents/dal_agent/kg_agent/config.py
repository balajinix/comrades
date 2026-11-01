"""Centralised env-var loading. Read once at import; restart to pick up changes."""
from __future__ import annotations

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("kg_agent.config")

# LLM — provider is the default backend; the OPENAI_* pair routes through
# the litellm proxy when the caller asks for an `openai/` or `anthropic/`
# model id (Claude lives there). Both can be set; build_llm picks per-call.
PROVIDER_API_KEY = os.getenv("PROVIDER_API_KEY")
PROVIDER_URL = os.getenv("PROVIDER_URL")
PROVIDER_PROJECT_ID = os.getenv("PROVIDER_PROJECT_ID")
PROVIDER_MODEL_NAME = os.getenv("PROVIDER_MODEL_NAME", "mistralai/mistral-medium-2505")

# Read both names — OPENAI_BASE_URL is the openai-python convention,
# OPENAI_API_BASE is what litellm reads. Accept either to avoid a footgun.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")


def _renamed_env(new_name: str, old_name: str, default: str | None = None) -> str | None:
    """Read `new_name`, falling back to the deprecated `old_name`.

    The agent was renamed dal-agent → kg-agent; its env-var prefix moved
    DAL_AGENT_* → KG_AGENT_*. Deployed `.env` files may still carry the old
    names, so honour them with a one-time deprecation warning. Drop the
    fallback in a later cleanup once all hosts are migrated.
    """
    val = os.getenv(new_name)
    if val is not None:
        return val
    old = os.getenv(old_name)
    if old is not None:
        logger.warning(
            "env var %s is deprecated; rename it to %s (honouring %s for now)",
            old_name, new_name, old_name,
        )
        return old
    return default


# ---------------------------------------------------------------------------
# MCP server federation
# ---------------------------------------------------------------------------
# The agent federates THREE servers (see deployment.md port table):
#   :8001  mcp_kg  — graph & metadata (CEG, Federated Registry, ITOM, temporal)
#   :8002  mcp_dal  — product & operational data (FinOps, APM,
#                     ITSM, Resilience, Infra) — 33 tools
#   :8003  mcp_var  — resilience metrics (calculate_* family)
# Each tool is dispatched back to the server that advertised it; the `server`
# label is carried through the catalog and used to group the tool listing in
# the system prompt so the LLM can route metadata vs data vs resilience.
_DEFAULT_SERVERS: list[tuple[str, str]] = [
    ("mcp_kg", "http://localhost:8001/mcp"),
    ("mcp_dal", "http://localhost:8002/mcp"),
    ("mcp_var", "http://localhost:8003/mcp"),
]


def _resolve_mcp_servers() -> list[tuple[str, str]]:
    """Resolve the labeled list of MCP servers the agent talks to.

    Priority:
      1. `MCP_SERVERS` — preferred. Comma-separated `label=url` pairs, e.g.
         `mcp_kg=http://host:8001/mcp,mcp_dal=http://host:8002/mcp`. The
         label is used only for prompt grouping; dispatch always follows the
         URL the tool was advertised by.
      2. `MCP_URLS` — back-compat (unlabeled, comma-separated). Labels are
         inferred positionally (first→mcp_kg, second→mcp_dal, third→mcp_var,
         rest→server_N). Emits a deprecation warning recommending MCP_SERVERS.
      3. `MCP_KG_URL` — single-URL back-compat for v0.1 deploys.
      4. Default: the three production endpoints on localhost.

    Per-server failures are non-fatal at runtime (see mcp_client.list_tools),
    so defaulting to all three is safe — a missing env never silently drops
    a slice of the catalog.
    """
    labeled = os.getenv("MCP_SERVERS")
    if labeled:
        out: list[tuple[str, str]] = []
        for chunk in labeled.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                label, url = chunk.split("=", 1)
                out.append((label.strip(), url.strip()))
            else:
                # Tolerate a bare URL inside MCP_SERVERS — label it positionally.
                out.append((_positional_label(len(out)), chunk))
        if out:
            return out

    csv = os.getenv("MCP_URLS")
    if csv:
        logger.warning(
            "MCP_URLS is deprecated; set MCP_SERVERS with explicit "
            "`label=url` pairs (e.g. mcp_kg=...,mcp_dal=...,mcp_var=...)."
        )
        urls = [u.strip() for u in csv.split(",") if u.strip()]
        return [(_positional_label(i), u) for i, u in enumerate(urls)]

    single = os.getenv("MCP_KG_URL")
    if single:
        return [("mcp_kg", single.strip())]

    return list(_DEFAULT_SERVERS)


def _positional_label(index: int) -> str:
    """Infer a server label from position for the unlabeled back-compat path."""
    defaults = ["mcp_kg", "mcp_dal", "mcp_var"]
    return defaults[index] if index < len(defaults) else f"server_{index + 1}"


# Labeled federation list: [(server_label, url), ...]. This is the canonical
# config; mcp_client, react, and run_agent all thread it through.
MCP_SERVERS: list[tuple[str, str]] = _resolve_mcp_servers()

# Back-compat: a flat URL list. Some callers / tests still import this name.
MCP_URLS: list[str] = [url for _, url in MCP_SERVERS]

# Back-compat alias. Some external code (smoke tests, debug shells) imports
# this name. Points at the first configured URL.
MCP_KG_URL: str = MCP_URLS[0]

# Persistence. DB renamed dal_agent.db → kg_agent.db with the package rename.
KG_AGENT_DB_URL = _renamed_env(
    "KG_AGENT_DB_URL", "DAL_AGENT_DB_URL", "sqlite+aiosqlite:///./kg_agent.db"
)

# Loop limits. This bounds PRODUCTIVE tool calls; coaching/rejection turns
# (hallucination-reject, JSON-strike, loop-detection) don't count against it
# (see react.run_react), so 6 real tool calls of headroom is plenty for the
# multi-step chains we see while staying well short of runaway.
KG_AGENT_MAX_STEPS = int(_renamed_env("KG_AGENT_MAX_STEPS", "DAL_AGENT_MAX_STEPS", "6"))


def _bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Query cache (read-through against the runs table). See CACHE_DESIGN.md
# (v1) and CACHE_DESIGN_V2.md (current). v2 removes the TTL — cache lives
# forever; clear with `clear_cache()` / `POST /api/cache/clear`.
KG_AGENT_CACHE_ENABLED: bool = _bool_env(
    _renamed_env("KG_AGENT_CACHE_ENABLED", "DAL_AGENT_CACHE_ENABLED"), True
)

# Logging
KG_AGENT_LOG_LEVEL = (
    _renamed_env("KG_AGENT_LOG_LEVEL", "DAL_AGENT_LOG_LEVEL", "INFO") or "INFO"
).upper()
