"""A2A mechanism: drive the workflow over the agent-to-agent (A2A) protocol by
sending the request to a running A2A agent service (default: the dal_agent on
:8005, which runs the same ReAct engine as MCPOnlyRunner).

Because the engine is identical, MCPOnlyRunner (in-process) vs A2ARunner (over
A2A/HTTP) isolates the *transport* cost of the A2A protocol — exactly the
controlled comparison the benchmark wants.

Limitation: token usage is computed server-side and is not returned over A2A, so
tokens stay None for this mechanism unless the agent is extended to report usage.
Latency is measured client-side and therefore includes transport, which is the
point of this comparison.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

_DAL_AGENT = Path(__file__).resolve().parents[3] / "agents" / "dal_agent"
if str(_DAL_AGENT) not in sys.path:
    sys.path.insert(0, str(_DAL_AGENT))

import httpx  # noqa: E402
from a2a.client import A2ACardResolver, A2AClient  # noqa: E402
from a2a.types import (  # noqa: E402
    Message, MessageSendParams, Part, Role, SendMessageRequest, TextPart,
)

from runner import RunResult, WorkflowRunner  # noqa: E402


class A2ARunner(WorkflowRunner):
    mechanism = "a2a"

    def __init__(self, model: str | None = None, agent_url: str | None = None,
                 timeout: float = 180.0):
        super().__init__(model=model)
        self.agent_url = agent_url or os.getenv("DAL_AGENT_URL", "http://localhost:8005")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._a2a: A2AClient | None = None

    async def _ensure_client(self) -> None:
        if self._a2a is not None:
            return
        self._client = httpx.AsyncClient(timeout=self.timeout)
        card = await A2ACardResolver(self._client, self.agent_url).get_agent_card()
        self._a2a = A2AClient(self._client, card, url=self.agent_url)

    async def arun(self, request_id: str, prompt: str) -> RunResult:
        res = RunResult(request_id=request_id, mechanism=self.mechanism, model=self.model)
        t0 = time.monotonic()
        try:
            await self._ensure_client()
            msg = Message(role=Role.user, messageId=uuid.uuid4().hex,
                          parts=[Part(root=TextPart(text=prompt))])
            req = SendMessageRequest(id=uuid.uuid4().hex,
                                     params=MessageSendParams(message=msg))
            resp = await self._a2a.send_message(req)
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            res.latency_s = round(time.monotonic() - t0, 3)
            return res

        res.latency_s = round(time.monotonic() - t0, 3)
        d = resp.model_dump()
        result = d.get("result", {}) or {}
        status = result.get("status", {}) or {}
        state = str(status.get("state", "")).lower()
        res.success = "completed" in state

        # final answer text
        parts = (status.get("message") or {}).get("parts", []) or []
        text = " ".join(p.get("text", "") for p in parts if p.get("text"))
        res.artifact = text or None
        res.error = None if res.success else (text or f"task state: {state}")

        # messages exchanged over A2A (proxy for coordination volume for this
        # single-agent service: user + working + completed)
        res.messages = len(result.get("history", []) or []) + 1
        res.raw = {"state": state}
        return res

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._a2a = None
