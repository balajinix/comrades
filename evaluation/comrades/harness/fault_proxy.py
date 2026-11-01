"""COMRADES fault-injecting MCP proxy — the common measurement boundary.

A streamable-http MCP server that fronts ONE upstream MCP server (mcp_map_reflect
:8203 for the schema-map experiment; mcp_dal :8201 for the harness/RQ5 work). It
mirrors the upstream tool surface and, per call:

  - with probability ``rate`` → return an injected fault (a tool error carrying the
    fault type), drawn from a seeded RNG (policy.FAULT_CATALOG);
  - else → forward to the upstream server and return its result verbatim.

Every mechanism under study — coded workflow, NL-playbook skill, free ReAct,
and (later) real coding harnesses — points at THIS proxy, so faults are injected
at one boundary and every tool call is counted identically. The proxy writes a
JSONL log of every call (the ground-truth, mechanism-independent communication
count) keyed by ``run_id``.

Per-run control: a runner calls the unadvertised ``__comrades_control__`` tool
before each request to set (rate, seed, run_id); the RNG is reseeded to
``"{seed}:{run_id}"`` so a given (request, seed) reproduces its fault pattern
regardless of mechanism or call order. Runs are sequential (one shared RNG).

Usage:
    FAULT_PROXY_TARGET=http://localhost:8203/mcp FAULT_PROXY_PORT=8202 \
        python harness/fault_proxy.py
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

import anyio
import mcp.types as types
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy import FAULT_CATALOG, pick_fault  # noqa: E402

CONTROL_TOOL = "__comrades_control__"

UPSTREAM = os.getenv("FAULT_PROXY_TARGET", "http://localhost:8203/mcp")
HOST = os.getenv("FAULT_PROXY_HOST", "0.0.0.0")
PORT = int(os.getenv("FAULT_PROXY_PORT", "8202"))
# Replay mode: on a non-faulted call, return a synthetic success instead of
# forwarding to the upstream. Keeps the rate-0 baseline clean and VPN-free (the
# local mcp_dal has no live backends), so faults are ONLY the injected ones --
# the same canned-replay rationale as RQ1/RQ2. The DV is communication, not data.
REPLAY = os.getenv("FAULT_PROXY_REPLAY", "0") == "1"
# Tools to hide from the advertised surface (keeps the task multi-hop: the
# one-shot map_schema would collapse the workflow to a single call).
HIDE = {t.strip() for t in os.getenv("FAULT_PROXY_HIDE", "map_schema").split(",") if t.strip()}
LOG_PATH = Path(os.getenv("FAULT_PROXY_LOG",
                          str(Path(__file__).resolve().parent / "proxy_calls.jsonl")))

# --- proxy state (module-global; runs are sequential) -----------------------
_STATE = {
    "rate": float(os.getenv("COMRADES_FAULT_RATE", "0.0")),
    "seed": int(os.getenv("COMRADES_FAULT_SEED", "0")),
    "run_id": "boot",
    "types": None,            # None => all fault types
    "rng": random.Random(0),
    "seq": 0,                 # call index within the current run
}
_UPSTREAM_TOOLS: list[types.Tool] = []


def _reseed(rate: float, seed: int, run_id: str, fault_types):
    rng = random.Random()
    rng.seed(f"{seed}:{run_id}")  # str seed: stable across processes (sha512, v2)
    _STATE.update(rate=rate, seed=seed, run_id=run_id, rng=rng, seq=0,
                  types=fault_types or None)


def _log(record: dict) -> None:
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _args_fingerprint(arguments: dict) -> str:
    """Canonical, order-independent string of the args (for redundancy scoring)."""
    try:
        return json.dumps(arguments or {}, sort_keys=True, default=str)[:400]
    except (TypeError, ValueError):
        return str(arguments)[:400]


async def _forward(tool: str, arguments: dict) -> types.CallToolResult:
    """Forward one call to the upstream MCP server; return its result verbatim."""
    async with streamablehttp_client(UPSTREAM) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, arguments or {})


def _status_error(message: str, fault_type: str | None = None) -> types.CallToolResult:
    """Deliver a fault / upstream error the way mcp_dal and faults.py do: a
    ``{"status": "error", ...}`` payload carried as JSON text (NOT an MCP
    ``isError`` flag). mcp_client.call_tool parses content JSON into a dict, so
    the ReAct agent sees ``{status: error}`` — its error-handling and the
    redundancy classifier's error detection both fire — and the coded workflow
    reads the same ``status``/``fault_type`` to drive its retry policy."""
    payload = {"status": "error", "error": message, "fault_type": fault_type}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))], isError=False
    )


def _canned_success(name: str) -> types.CallToolResult:
    """A synthetic successful tool result for replay mode: a small, generic ``ok``
    payload (content-only JSON, matching how mcp_dal delivers results) so the harness
    sees a non-error response and proceeds. The DV is the harness's communication, not
    the data, so a uniform stand-in across tools keeps the rate-0 baseline clean."""
    payload = {
        "status": "ok", "source": "comrades-replay", "tool": name,
        "rows": [{"id": 1, "name": "sample-a", "value": 100},
                 {"id": 2, "name": "sample-b", "value": 75}],
        "note": "Synthetic data for COMRADES reproducibility (no live backend).",
    }
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))], isError=False)


async def _fetch_upstream_tools() -> list[types.Tool]:
    async with streamablehttp_client(UPSTREAM) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tl = await session.list_tools()
            # Strip outputSchema: an injected fault is content-only (a {status:error}
            # payload), which the mcp client would otherwise reject for a tool that
            # declares an output schema ("did not return structured content").
            # Successful forwards still carry their structured content through.
            return [t.model_copy(update={"outputSchema": None})
                    for t in tl.tools if t.name not in HIDE]


server = Server("comrades-fault-proxy")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return _UPSTREAM_TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    # Control channel: set the current run's (rate, seed, run_id). Never faulted,
    # never logged, never advertised in list_tools.
    if name == CONTROL_TOOL:
        a = arguments or {}
        _reseed(float(a.get("rate", 0.0)), int(a.get("seed", 0)),
                str(a.get("run_id", "run")), a.get("types"))
        return _error_result_ok(a)

    _STATE["seq"] += 1
    seq = _STATE["seq"]
    faulted = _STATE["rate"] > 0 and _STATE["rng"].random() < _STATE["rate"]
    fault_type = None
    if faulted:
        fault_type, template = pick_fault(_STATE["rng"], _STATE["types"])
        message = template.format(tool=name)
    _log({
        "ts": round(time.time(), 3), "run_id": _STATE["run_id"], "seq": seq,
        "tool": name, "args_fp": _args_fingerprint(arguments),
        "faulted": faulted, "fault_type": fault_type, "rate": _STATE["rate"],
    })
    if faulted:
        return _status_error(f"[{fault_type}] {message}", fault_type)
    if REPLAY:
        return _canned_success(name)
    try:
        return await _forward(name, arguments)
    except Exception as exc:  # upstream/transport failure (not an injected fault)
        return _status_error(
            f"upstream error calling {name}: {type(exc).__name__}: {exc}", "upstream")


def _error_result_ok(a: dict) -> types.CallToolResult:
    """Acknowledge a control call (isError=False so the runner sees success)."""
    return types.CallToolResult(
        content=[types.TextContent(
            type="text",
            text=f"control set rate={a.get('rate')} seed={a.get('seed')} run_id={a.get('run_id')}")],
        isError=False,
    )


session_manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=False)


async def _handle(scope, receive, send):
    await session_manager.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def _lifespan(app):
    global _UPSTREAM_TOOLS
    for attempt in range(30):  # upstream may still be booting (embedder load)
        try:
            _UPSTREAM_TOOLS = await _fetch_upstream_tools()
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 29:
                raise
            print(f"[fault_proxy] upstream not ready ({exc}); retrying...", flush=True)
            await anyio.sleep(2)
    print(f"[fault_proxy] mirroring {len(_UPSTREAM_TOOLS)} tools from {UPSTREAM} "
          f"(hidden: {sorted(HIDE)})", flush=True)
    async with session_manager.run():
        yield


app = Starlette(routes=[Mount("/mcp", app=_handle)], lifespan=_lifespan)


def main() -> None:
    print(f"[fault_proxy] serving http://{HOST}:{PORT}/mcp -> upstream {UPSTREAM}; "
          f"log={LOG_PATH}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
