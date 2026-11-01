import logging
import time
import asyncio
from typing import Optional, Dict, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils.errors import ServerError
from app.agent import GraphQLQueryAgent

logger = logging.getLogger(__name__)
a2a_logger = logging.getLogger("a2a_integration")
a2a_logger.setLevel(logging.INFO)

class GraphQLQueryAgentExecutor(AgentExecutor):
    """
    Wraps GraphQLQueryAgent so it can run inside the A2A runtime.
    """

    def __init__(self) -> None:
        a2a_logger.info("🏗️ Initializing GraphQLQueryAgentExecutor...")
        start_time = time.time()
        
        try:
            self.agent = GraphQLQueryAgent()
            elapsed = time.time() - start_time
            a2a_logger.info(f"✅ Agent initialized in {elapsed:.3f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            a2a_logger.error(f"❌ Agent initialization failed after {elapsed:.3f}s: {e}")
            raise

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute with detailed A2A integration logging.
        """
        execution_start = time.time()
        a2a_logger.info("🚀 A2A execute() called")
        a2a_logger.info(f"📋 Task ID: {context.task_id}")
        a2a_logger.info(f"🆔 Context ID: {context.context_id}")
        
        # Validate context
        if not (context.task_id and context.context_id and context.message):
            a2a_logger.error("❌ RequestContext validation failed - missing mandatory fields")
            raise ServerError(error=InternalError(message="RequestContext is missing mandatory fields"))

        a2a_logger.info("✅ RequestContext validation passed")

        # Initialize task updater
        updater_start = time.time()
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        updater_time = time.time() - updater_start
        a2a_logger.info(f"🔄 TaskUpdater created in {updater_time:.3f}s")

        # Submit and start work
        try:
            if not context.current_task:
                submit_start = time.time()
                await updater.submit()
                submit_time = time.time() - submit_start
                a2a_logger.info(f"📤 Task submitted in {submit_time:.3f}s")
            
            work_start = time.time()
            await updater.start_work()
            work_time = time.time() - work_start
            a2a_logger.info(f"🏃 Work started in {work_time:.3f}s")
            
        except Exception as e:
            a2a_logger.error(f"❌ Task submission/start failed: {e}")
            raise

        user_query = context.get_user_input()
        if not user_query:
            a2a_logger.error("❌ User input is empty")
            raise ServerError(error=InternalError(message="User input was empty"))

        a2a_logger.info(f"📝 User query: {user_query}")

        # Execute agent with detailed monitoring
        agent_start = time.time()
        step_count = 0
        
        try:
            a2a_logger.info("🤖 Starting agent.stream()...")
            
            async for step in self.agent.stream(user_query, context.context_id):
                step_count += 1
                step_elapsed = time.time() - agent_start
                
                a2a_logger.info(f"📥 Agent step #{step_count} at {step_elapsed:.3f}s")
                a2a_logger.debug(f"   Step data: {list(step.keys())}")
                
                # Log step content details
                content = step.get("content", "")
                is_complete = step.get("is_task_complete", False)
                a2a_logger.info(f"   Content length: {len(content)} chars")
                a2a_logger.info(f"   Is complete: {is_complete}")
                a2a_logger.debug(f"   Content preview: {content[:100]}...")
                
                # Handle stream step with timing
                handle_start = time.time()
                await self._handle_stream_step(step, updater)
                handle_time = time.time() - handle_start
                a2a_logger.debug(f"   Step handled in {handle_time:.3f}s")
                
                # Check for completion
                if is_complete:
                    total_agent_time = time.time() - agent_start
                    a2a_logger.info(f"🏁 Agent completed after {step_count} steps in {total_agent_time:.3f}s")
                    
                    # CRITICAL FIX: Give A2A framework time to process artifacts
                    a2a_logger.info("⏳ Allowing time for A2A framework to process artifacts...")
                    await asyncio.sleep(0.5)  # 500ms delay
                    a2a_logger.info("✅ A2A processing delay completed")
                    break
                    
                # Log if taking too long
                if step_elapsed > 10:
                    a2a_logger.warning(f"⚠️ Agent has been running for {step_elapsed:.1f}s")
                if step_elapsed > 30:
                    a2a_logger.error(f"🚨 Agent running for {step_elapsed:.1f}s - potential timeout risk")
                    
        except Exception as exc:
            agent_elapsed = time.time() - agent_start
            a2a_logger.error(f"❌ Agent execution failed after {agent_elapsed:.3f}s and {step_count} steps")
            a2a_logger.error(f"   Error: {exc}")
            a2a_logger.error(f"   Error type: {type(exc).__name__}")
            
            # Send failure status
            await updater.update_status(
                "failed",
                updater.new_agent_message([Part(root=TextPart(text=f"❌ Agent failed: {str(exc)}"))]),
            )
            raise ServerError(error=InternalError()) from exc

        # Log total execution time
        total_execution = time.time() - execution_start
        a2a_logger.info(f"🎉 A2A execution completed in {total_execution:.3f}s")

    async def _handle_stream_step(self, step: Dict[str, Any], updater: TaskUpdater) -> None:
        """
        Convert agent-emitted dicts → A2A task updates.
        """
        handle_start = time.time()
        
        try:
            state = TaskState.working
            if step.get("is_task_complete"):
                state = "completed"

            content = step.get("content", "")
            if not content:
                content = "Processing..."

            # Create message
            message_start = time.time()
            message = updater.new_agent_message([Part(root=TextPart(text=content))])
            message_time = time.time() - message_start
            a2a_logger.debug(f"📨 Message created in {message_time:.3f}s")

            # Update status
            status_start = time.time()
            await updater.update_status(state, message=message)
            status_time = time.time() - status_start
            a2a_logger.debug(f"📊 Status updated to '{state}' in {status_time:.3f}s")

            # Handle artifacts on completion
            if state == "completed":
                a2a_logger.info("🔄 Processing completion - attaching artifacts...")
                artifact_start = time.time()
                await self._attach_artifacts(step, updater)
                artifact_time = time.time() - artifact_start
                a2a_logger.info(f"📎 Artifacts attached in {artifact_time:.3f}s")
                
                # Additional small delay after artifact attachment
                await asyncio.sleep(0.1)
                a2a_logger.debug("✅ Post-artifact delay completed")

        except Exception as exc:
            handle_time = time.time() - handle_start
            a2a_logger.error(f"❌ Stream step handling failed after {handle_time:.3f}s: {exc}")
            
            # Try to send error status
            try:
                await updater.update_status(
                    "working",
                    updater.new_agent_message([
                        Part(root=TextPart(text="⚠️ Processing step encountered an issue, continuing..."))
                    ]),
                )
            except Exception as status_error:
                a2a_logger.error(f"❌ Failed to send error status: {status_error}")

        finally:
            total_handle_time = time.time() - handle_start
            a2a_logger.debug(f"⏱️ Total step handling time: {total_handle_time:.3f}s")

    async def _attach_artifacts(self, step: Dict[str, Any], updater: TaskUpdater) -> None:
        """
        Attach artifacts with detailed logging.
        """
        artifacts_to_attach = []
        
        # Check what artifacts are available
        if step.get("graphql_query"):
            artifacts_to_attach.append(("graphql_query", step["graphql_query"]))
            a2a_logger.info(f"📋 GraphQL query artifact: {len(step['graphql_query'])} chars")
            
        if step.get("title"):
            artifacts_to_attach.append(("query_title", step["title"]))
            a2a_logger.info(f"🏷️ Title artifact: '{step['title']}'")
            
        if step.get("report"):
            artifacts_to_attach.append(("query_report", step["report"]))
            a2a_logger.info(f"📄 Report artifact: {len(step['report'])} chars")

        if not artifacts_to_attach:
            a2a_logger.warning("⚠️ No artifacts to attach")
            return

        a2a_logger.info(f"📎 Attaching {len(artifacts_to_attach)} artifacts...")

        # Attach artifacts with individual timing and delays
        for i, (name, content) in enumerate(artifacts_to_attach):
            try:
                attach_start = time.time()
                await updater.add_artifact(
                    [Part(root=TextPart(text=content))],
                    name=name,
                )
                attach_time = time.time() - attach_start
                a2a_logger.debug(f"   {name}: attached in {attach_time:.3f}s")
                
                # Small delay between artifacts to prevent queue issues
                if i < len(artifacts_to_attach) - 1:  # Don't delay after last artifact
                    await asyncio.sleep(0.05)  # 50ms between artifacts
                
            except Exception as exc:
                a2a_logger.error(f"❌ Failed to attach artifact '{name}': {exc}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """
        A2A host asked to cancel the running task.
        This agent doesn't implement granular cancellation yet,
        so we simply mark the task as cancelled and return.
        """
        cancel_start = time.time()
        a2a_logger.info(f"🛑 Cancel requested for task {context.task_id}")

        try:
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await updater.update_status(
                "cancelled",
                updater.new_agent_message([
                    Part(root=TextPart(text="⏹️ Task cancelled by user"))
                ]),
            )
            
            cancel_time = time.time() - cancel_start
            a2a_logger.info(f"✅ Cancellation completed in {cancel_time:.3f}s")
            
        except Exception as exc:
            cancel_time = time.time() - cancel_start
            a2a_logger.error(f"❌ Cancellation failed after {cancel_time:.3f}s: {exc}")
            
        return None
