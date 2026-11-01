"""MCP-only mechanism: a single agent federates the mcp_dal tool catalog over
the Model Context Protocol and chains calls itself, with no inter-agent
messaging (the "model-as-router" design point).

Wraps the vendored kg_agent ReAct loop directly (the same engine behind the
running dal_agent A2A service), so this measures the federation path without
A2A/HTTP transport overhead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the vendored kg_agent importable: the package lives in agents/dal_agent/.
# (comrades/ itself is put on sys.path by the driver, so runner/metrics import bare.)
_DAL_AGENT = Path(__file__).resolve().parents[3] / "agents" / "dal_agent"
if str(_DAL_AGENT) not in sys.path:
    sys.path.insert(0, str(_DAL_AGENT))

from kg_agent import run_agent  # noqa: E402  (after sys.path bootstrap)

from runner import RunResult, WorkflowRunner  # noqa: E402
from metrics import capture_tokens  # noqa: E402


def _had_transient_failure(raw: dict) -> bool:
    """True if any ReAct step saw a tool/observation error (used for recovery)."""
    for step in raw.get("trace", []) or []:
        tr = step.get("tool_result")
        if isinstance(tr, dict) and str(tr.get("status", "")).lower() == "error":
            return True
        obs = (step.get("observation") or "")
        if any(m in obs for m in ("status=error", "ERROR:", "Failed to",
                                  "NameResolution", "Unauthorized", "Connection")):
            return True
    return False


class MCPOnlyRunner(WorkflowRunner):
    mechanism = "mcp_only"

    def __init__(self, model: str | None = None,
                 mcp_url: str | None = None, max_steps: int = 6):
        super().__init__(model=model)
        self.mcp_url = (mcp_url or os.getenv("MCP_DAL_URL") or os.getenv("DAL_MCP_URL")
                        or "http://localhost:8201/mcp")
        self.max_steps = max_steps
        self.mcp_servers = [("mcp_dal", self.mcp_url)]

    async def arun(self, request_id: str, prompt: str) -> RunResult:
        res = RunResult(request_id=request_id, mechanism=self.mechanism, model=self.model)
        try:
            with capture_tokens() as tok:
                out = await run_agent(
                    prompt,
                    max_steps=self.max_steps,
                    mcp_servers=self.mcp_servers,
                    model=self.model,
                )
        except Exception as e:  # never raise out of a runner
            res.error = f"{type(e).__name__}: {e}"
            return res

        raw = out.raw_result or {}
        res.success = out.status == "completed"
        res.artifact = out.answer_markdown or out.message
        res.error = None if res.success else (out.message or "run errored")
        res.latency_s = float(out.elapsed_seconds or 0.0)
        res.messages = int(raw.get("steps_used") or 0)
        res.recovered = res.success and _had_transient_failure(raw)
        res.tool_invoked = out.tool_invoked
        res.model = out.model or self.model
        res.tokens_total = tok.get("total") or None
        res.tokens_prompt = tok.get("prompt") or None
        res.tokens_completion = tok.get("completion") or None
        res.raw = {"source": out.source, "routing_reason": out.routing_reason,
                   "trace": raw.get("trace")}  # for the redundancy classifier (RQ3)
        return res
