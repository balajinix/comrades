import logging
import os

from kg_agent import run_agent, run_agent_sync

logger = logging.getLogger(__name__)


class DALAgent:
    """Data Access Layer agent.

    Wraps the federated ReAct agent (vendored ``kg_agent``) and points it at the
    ``mcp_dal`` MCP server, which exposes natural-language access to heterogeneous
    observability, cloud-cost, ITSM, resilience, and infrastructure backends.

    Exposes the same ``run_agent(user_input) -> dict`` surface the other
    comrades A2A agents use, so the orchestrator can route to it unchanged.
    """

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, mcp_url: str | None = None, model: str | None = None,
                 max_steps: int | None = None):
        self.mcp_url = mcp_url or os.getenv("DAL_MCP_URL", "http://localhost:8201/mcp")
        # Optional explicit model route (e.g. "provider/...", "openai/...").
        # When None, kg_agent resolves it from PROVIDER_*/OPENAI_* env vars.
        self.model = model or os.getenv("DAL_AGENT_MODEL")
        self.max_steps = max_steps or int(os.getenv("DAL_AGENT_MAX_STEPS", "6"))
        self.mcp_servers = [("mcp_dal", self.mcp_url)]
        logger.info("DALAgent configured: mcp=%s model=%s max_steps=%s",
                    self.mcp_url, self.model or "<env-resolved>", self.max_steps)

    async def arun_agent(self, user_input: str) -> dict:
        """Async entry point. Awaits the federated ReAct loop directly so it
        shares the caller's event loop (required when invoked from the A2A
        server, which is itself async)."""
        try:
            out = await run_agent(
                user_input,
                max_steps=self.max_steps,
                mcp_servers=self.mcp_servers,
                model=self.model,
            )
        except Exception as e:  # noqa: BLE001 - surface any failure to the executor
            logger.error("DAL agent run failed", exc_info=True)
            return {"error": str(e)}
        return self._normalize(out)

    def run_agent(self, user_input: str) -> dict:
        """Sync entry point for non-async callers (spins up its own loop)."""
        try:
            out = run_agent_sync(
                user_input,
                max_steps=self.max_steps,
                mcp_servers=self.mcp_servers,
                model=self.model,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("DAL agent run failed", exc_info=True)
            return {"error": str(e)}
        return self._normalize(out)

    @staticmethod
    def _had_transient_failure(raw: dict) -> bool:
        """True if any ReAct step saw a tool/observation error (recovery signal)."""
        for step in (raw or {}).get("trace", []) or []:
            tr = step.get("tool_result")
            if isinstance(tr, dict) and str(tr.get("status", "")).lower() == "error":
                return True
            obs = step.get("observation") or ""
            if any(m in obs for m in ("status=error", "ERROR:", "Failed to",
                                      "NameResolution", "Unauthorized", "Connection")):
                return True
        return False

    @staticmethod
    def _normalize(out) -> dict:
        raw = out.raw_result or {}
        messages = int(raw.get("steps_used") or 0)
        if out.status == "error":
            return {"error": out.message or "DAL agent returned an error",
                    "model": out.model, "messages": messages, "recovered": False}
        return {
            "error": None,
            "answer": out.answer_markdown or out.message or "",
            "tool_invoked": out.tool_invoked,
            "routing_reason": out.routing_reason,
            "source": out.source,
            "model": out.model,
            "elapsed_seconds": out.elapsed_seconds,
            "messages": messages,
            "recovered": DALAgent._had_transient_failure(raw),
            "renderings": [r.model_dump() for r in (out.renderings or [])],
        }
