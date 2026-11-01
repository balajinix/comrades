"""Direct (in-process) mechanism: the orchestrator embeds the data agent as a
library and calls it via a plain Python call --- no serialization, no HTTP, no
protocol. This is the lower bound on communication overhead.

It wraps the SAME ``DALAgent`` engine that the A2A service (dal_agent:8005)
exposes. So ``direct`` vs ``a2a`` runs the identical agent on both sides and the
difference is purely the A2A transport --- the controlled comparison the
benchmark wants.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DAL_AGENT = Path(__file__).resolve().parents[3] / "agents" / "dal_agent"
if str(_DAL_AGENT) not in sys.path:
    sys.path.insert(0, str(_DAL_AGENT))

from app.agent import DALAgent  # noqa: E402  (the production agent object)

from runner import RunResult, WorkflowRunner  # noqa: E402
from metrics import capture_tokens  # noqa: E402


class DirectRunner(WorkflowRunner):
    mechanism = "direct"

    def __init__(self, model: str | None = None,
                 mcp_url: str | None = None, max_steps: int = 6):
        super().__init__(model=model)
        # DALAgent reads DAL_MCP_URL / DAL_AGENT_MODEL / DAL_AGENT_MAX_STEPS from
        # env; pass explicit values so the harness controls them. Remote-friendly:
        # MCP_DAL_URL (e.g. the kg box) takes precedence, then DAL_MCP_URL.
        mcp_url = mcp_url or os.getenv("MCP_DAL_URL") or os.getenv("DAL_MCP_URL")
        if mcp_url:
            os.environ["DAL_MCP_URL"] = mcp_url
        self.agent = DALAgent(mcp_url=mcp_url, model=model, max_steps=max_steps)

    async def arun(self, request_id: str, prompt: str) -> RunResult:
        res = RunResult(request_id=request_id, mechanism=self.mechanism, model=self.model)
        try:
            with capture_tokens() as tok:
                out = await self.agent.arun_agent(prompt)
        except Exception as e:  # never raise out of a runner
            res.error = f"{type(e).__name__}: {e}"
            return res

        if out.get("error"):
            res.error = out["error"]
            res.model = out.get("model") or self.model
            res.messages = int(out.get("messages") or 0)
            return res

        res.success = True
        res.artifact = out.get("answer")
        res.tool_invoked = out.get("tool_invoked")
        res.model = out.get("model") or self.model
        res.latency_s = float(out.get("elapsed_seconds") or 0.0)
        res.messages = int(out.get("messages") or 0)
        res.recovered = bool(out.get("recovered"))
        res.tokens_total = tok.get("total") or None
        res.tokens_prompt = tok.get("prompt") or None
        res.tokens_completion = tok.get("completion") or None
        res.raw = {"source": out.get("source"), "routing_reason": out.get("routing_reason")}
        return res
