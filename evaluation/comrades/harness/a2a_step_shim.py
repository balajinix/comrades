"""A2A step-shim — the transport off-diagonal for the control-discipline study.

A thin A2A agent that accepts a single ``{"tool": ..., "args": {...}}`` request,
calls that tool on the fault-proxy over MCP, and returns the result as JSON text.
It adds NO control logic of its own — that is the point: the SAME coded workflow
(run_control.py, --transport a2a) dispatches each granular map_reflect step over
A2A through this shim instead of calling the proxy directly over MCP. Control
discipline (workflow) and the retry policy are held fixed; only the transport
changes, so the wf-mcp vs wf-a2a contrast isolates transport.

Faults still occur at the proxy (the shim just forwards), so wf-mcp and wf-a2a
cross the identical fault boundary and the proxy log counts both the same way.

Run with the dal_agent venv (proxy :8202 up):
    A2A_SHIM_PORT=8210 FAULT_PROXY_URL=http://localhost:8202/mcp \
        python harness/a2a_step_shim.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (BasePushNotificationSender, InMemoryPushNotificationConfigStore,
                              InMemoryTaskStore, TaskUpdater)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2a.types import UnsupportedOperationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "agents" / "dal_agent"))
from kg_agent.mcp_client import call_tool  # noqa: E402

PROXY_URL = os.getenv("FAULT_PROXY_URL", "http://localhost:8202/mcp")
HOST = os.getenv("A2A_SHIM_HOST", "localhost")
PORT = int(os.getenv("A2A_SHIM_PORT", "8210"))


class StepShimExecutor(AgentExecutor):
    """Forward one {tool, args} request to the fault-proxy and return the result."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)
        try:
            req = json.loads(user_input)
            result = await call_tool(req["tool"], req.get("args") or {}, PROXY_URL)
            payload = json.dumps(result, default=str)
            state = "completed"
        except Exception as exc:  # noqa: BLE001
            payload = json.dumps({"status": "error", "error": f"shim: {exc}"})
            state = "failed"
        await updater.update_status(
            state, new_agent_text_message(payload, task.contextId, task.id), final=True)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


def main() -> None:
    skill = AgentSkill(
        id="call_tool", name="MCP tool relay",
        description="Relays one {tool, args} call to the fault-proxy and returns its result.",
        tags=["mcp", "relay", "comrades"], examples=['{"tool": "find_type_mapping", "args": {}}'])
    card = AgentCard(
        name="COMRADES_A2A_Step_Shim",
        description="Thin A2A relay to the fault-proxy for the wf-a2a transport cell.",
        url=f"http://{HOST}:{PORT}/", version="1.0.0",
        defaultInputModes=["text/plain"], defaultOutputModes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False, pushNotifications=True),
        skills=[skill])
    httpx_client = httpx.AsyncClient(timeout=120.0)
    push_store = InMemoryPushNotificationConfigStore()
    handler = DefaultRequestHandler(
        agent_executor=StepShimExecutor(), task_store=InMemoryTaskStore(),
        push_config_store=push_store,
        push_sender=BasePushNotificationSender(httpx_client, push_store))
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    print(f"[a2a_step_shim] serving http://{HOST}:{PORT}/ -> proxy {PROXY_URL}", flush=True)
    uvicorn.run(app.build(), host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
