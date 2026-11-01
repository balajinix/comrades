import json
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Part, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from app.agent import DALAgent

logger = logging.getLogger(__name__)


class DALAgentExecutor(AgentExecutor):
    """A2A executor for the Data Access Layer agent."""

    def __init__(self):
        self.agent = DALAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("DAL agent execution started")

        user_input = context.get_user_input()
        logger.info(f"Processing DAL request: {user_input}")

        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            await updater.update_status(
                "working",
                new_agent_text_message(
                    f"Querying the data access layer: {user_input}",
                    task.contextId, task.id),
                final=False,
            )

            # Await the async ReAct loop so it shares this server's event loop.
            result = await self.agent.arun_agent(user_input)

            if result.get("error"):
                error_msg = f"❌ DAL query failed: {result['error']}"
                logger.error(error_msg)
                await updater.update_status(
                    "failed",
                    new_agent_text_message(error_msg, task.contextId, task.id),
                    final=True,
                )
                return

            # Attach the raw renderings (timeseries/table/graph/etc.) as an artifact.
            renderings = result.get("renderings") or []
            if renderings:
                await updater.add_artifact(
                    [Part(root=TextPart(text=json.dumps(renderings, indent=2)))],
                    name="dal_renderings",
                )

            tool = result.get("tool_invoked") or "n/a"
            source = result.get("source") or "DAL"
            completion_msg = (
                f"{result.get('answer', '')}\n\n"
                f"_Source: {source} · tool: {tool} · model: {result.get('model', 'n/a')} "
                f"· {result.get('elapsed_seconds', 0):.1f}s_"
            )
            await updater.update_status(
                "completed",
                new_agent_text_message(completion_msg, task.contextId, task.id),
                final=True,
            )
            logger.info("DAL agent execution completed (tool=%s)", tool)

        except Exception as e:
            error_msg = f"Unexpected error during DAL execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            try:
                await updater.update_status(
                    "failed",
                    new_agent_text_message(error_msg, task.contextId, task.id),
                    final=True,
                )
            except Exception as update_error:
                logger.error(f"Failed to send error status: {update_error}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.info("Cancellation requested - not supported")
        raise ServerError(error=UnsupportedOperationError())
