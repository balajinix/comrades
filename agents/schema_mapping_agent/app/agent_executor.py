from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2a.types import InternalError, UnsupportedOperationError
import json
import logging

from app.agent import GraphQLSchemaMappingAgent

logger = logging.getLogger(__name__)

class GraphQLSchemaMappingAgentExecutor(AgentExecutor):
    """A2A Executor for GraphQLSchemaMappingAgent"""

    def __init__(self):
        self.agent = GraphQLSchemaMappingAgent(base_schema_dir="./data/graphql")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Make sure the agent is initialized before execution
        await self.agent.async_init()

        logger.info("Schema mapping agent execution started")

        # Validate incoming request
        error = self._validate_request(context)
        if error:
            logger.error(f"Request validation failed: {error}")
            raise ServerError(error=error)

        # Get user input and ensure a task exists
        user_input = context.get_user_input()
        logger.info(f"Processing user input: {user_input}")
        
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
            logger.info(f"Created new task: {task.id}")

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            # Send initial status
            logger.info("Sending initial processing status")
            await updater.update_status(
                "working",
                new_agent_text_message(f"Processing schema mapping request: {user_input}", task.contextId, task.id),
                final=False,
            )
            
            # Run the agent directly (no streaming)
            logger.info("Running agent.run_agent()")
            result = self.agent.run_agent(user_input)
            
            # Check for errors
            if result.get("error"):
                error_msg = f"❌ Schema mapping failed: {result['error']}"
                logger.error(error_msg)
                
                await updater.update_status(
                    "failed",
                    new_agent_text_message(error_msg, task.contextId, task.id),
                    final=True,
                )
                return
            
            # Send progress update
            progress_msg = f"Successfully mapped {result.get('source_schema', 'Unknown')} to {result.get('target_schema', 'Unknown')}"
            logger.info("Sending progress update")
            await updater.update_status(
                "working",
                new_agent_text_message(progress_msg, task.contextId, task.id),
                final=False,
            )
            
            # Add artifact if mapping result exists
            if result.get("mapping_result"):
                logger.info("Adding mapping result artifact")
                artifact_text = json.dumps(result["mapping_result"], indent=2)
                await updater.add_artifact(
                    [Part(root=TextPart(text=artifact_text))],
                    name="schema_mapping_result",
                )
            
            # Send final completion status
            completion_msg = f"""🎉 Schema mapping completed successfully!

**Source Schema:** {result.get('source_schema', 'Unknown')}
**Target Schema:** {result.get('target_schema', 'Unknown')}

**Selection Reasoning:**
{result.get('schema_selection_reasoning', 'No reasoning provided')}

**Mapping Summary:** Generated {len(result.get('mapping_result', {}).get('properties', {}))} property mappings."""

            logger.info("Sending completion status")
            await updater.update_status(
                "completed",
                new_agent_text_message(completion_msg, task.contextId, task.id),
                final=True,
            )
            
            logger.info("Schema mapping execution completed successfully")
                
        except Exception as e:
            error_msg = f"Unexpected error during execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Send error status
            try:
                await updater.update_status(
                    "failed",
                    new_agent_text_message(error_msg, task.contextId, task.id),
                    final=True,
                )
            except Exception as update_error:
                logger.error(f"Failed to send error status: {update_error}")
            
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext):
        """Validate the incoming request - simplified"""
        # For now, just return None to accept all requests
        # We can add validation back later once basic functionality works
        return None

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.info("Cancellation requested - not supported")
        raise ServerError(error=UnsupportedOperationError())
