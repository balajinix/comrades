import logging
from typing import Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError # For raising proper A2A errors

from app.agent import TracePruningAgent 

logger = logging.getLogger(__name__)

class TracePruningAgentExecutor(AgentExecutor):
    """TracePruning Agent Executor for A2A SDK integration."""

    def __init__(self):
        self.agent = TracePruningAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Validate incoming request
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError(message=str(error)))

        query = context.get_user_input()
        if not query: 
            raise ServerError(error=InvalidParamsError(message="User input query is missing."))

        task = context.current_task
        if not task:
            if context.message: 
                task = new_task(context.message) 
                await event_queue.enqueue_event(task) 
            else: 
                logger.error("TracePruningAgentExecutor.execute called without a current_task or an incoming message.")
                raise ServerError(error=InternalError(message="Task context is critically missing."))
        
        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("TracePruningAgent is processing your request...", task.contextId, task.id),
            )

            final_response_text = "No final response captured by TracePruningAgent."
            async for step_result in self.agent.process_request(query, task.contextId):
                
                intermediate_message_text = f"[{step_result.get('type', 'update').upper()}] {step_result.get('content', '')}"
                logger.debug(f"TracePruningAgentExecutor: Streaming update: {intermediate_message_text}")

                await updater.update_status(
                    TaskState.working, 
                    new_agent_text_message(intermediate_message_text, task.contextId, task.id)
                )
                
                if step_result.get('type') == 'ai_message':
                    final_response_text = str(step_result.get('content', final_response_text))
                elif step_result.get('type') == 'error': # If the agent itself yields an error object
                    final_response_text = f"Agent error: {step_result.get('content', 'Unknown agent error')}"
                    logger.warning(f"Task {task.id} encountered an agent-yielded error: {final_response_text}")
            logger.info(f"Task {task.id} completion. Final response text: {final_response_text}")
            await updater.add_artifact(
                [Part(root=TextPart(text=final_response_text))],
                name='trace_pruning_agent_response',
            )

            await updater.complete() 
            logger.info(f"Task {task.id} marked as completed successfully.")

        except Exception as e:
            error_message_for_client = f"An unexpected error occurred in TracePruningAgent: {str(e)}"
            logger.error(f'Critical error in TracePruningAgentExecutor for task {task.id}: {e}', exc_info=True)
            
            try:
                # Try to update task status to failed using update_status as .fail() is missing
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(error_message_for_client, task.contextId, task.id)
                )
            except Exception as update_err:
                logger.error(f"Failed to send task failure update for task {task.id} after critical error: {update_err}", exc_info=True)
            
            # Re-raise as a ServerError for the A2A framework
            if isinstance(e, ServerError):
                raise  # If it's already a ServerError, re-raise it
            else:
                # Wrap other exceptions in InternalError
                raise ServerError(error=InternalError(message=error_message_for_client)) from e

    def _validate_request(self, context: RequestContext) -> Optional[str]:
        """Basic validation. Returns an error message string if invalid, else None."""
        if not context.get_user_input():
            return "User input (query) is missing."
        return None

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        task = request.current_task
        if task:
            logger.info(f"Cancelling task {task.id} for TracePruningAgent.")
            updater = TaskUpdater(event_queue, task.id, task.contextId)
            await updater.update_status(TaskState.cancelled, new_agent_text_message("Task cancelled by user.", task.contextId, task.id))
            return await updater.get_task() 
        
        logger.warning("TracePruningAgentExecutor.cancel called without a current task in context.")
        return None

    async def cleanup(self):
        """Called on server shutdown to clean up agent resources."""
        logger.info("TracePruningAgentExecutor initiating cleanup for TracePruningAgent...")
        await self.agent.cleanup()
        logger.info("TracePruningAgentExecutor cleanup for TracePruningAgent complete.")
