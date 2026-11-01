import logging
import json
import os
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from app.agent import DashboardVisualizationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardVisualizationAgentExecutor(AgentExecutor):
    """Dashboard Visualization Agent Executor for A2A protocol."""

    def __init__(self):
        self.agent = DashboardVisualizationAgent(
            results_dir="./data/results"
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        query = context.get_user_input()
        
        try:
            async for item in self.agent.stream(query, context.context_id):
                is_task_complete = item.get("is_task_complete", False)
                require_user_input = item.get("require_user_input", False)
                content = item.get("content", "")
                
                # Limit content size to prevent token overflow
                if len(content) > 100000:  # 100KB limit
                    logger.warning(f"Large content detected ({len(content)} chars), truncating for A2A response")
                    
                    # Keep only the important summary parts
                    lines = content.split('\n')
                    summary_lines = []
                    for line in lines:
                        if any(keyword in line for keyword in [
                            'Results Summary', 'Output Directory', 'Generated Chart Files',
                            'Chart Types Created', 'Next Steps', 'completed successfully'
                        ]):
                            summary_lines.append(line)
                    
                    if summary_lines:
                        content = '\n'.join(summary_lines[:50])  # First 50 summary lines
                    else:
                        content = f"Dashboard visualization completed successfully with {item.get('charts_generated', 0)} charts. Full details available in output directory."
                
                # Create parts for the content
                parts = [Part(root=TextPart(text=content))]

                if not is_task_complete and not require_user_input:
                    await updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(parts),
                    )
                elif require_user_input:
                    await updater.update_status(
                        TaskState.input_required,
                        message=updater.new_agent_message(parts),
                    )
                    break
                else:
                    # Task complete - add artifacts if available
                    charts_generated = item.get("charts_generated", 0)
                    
                    try:
                        # Add the main result first
                        await updater.add_artifact(
                            parts,
                            name="visualization_result",
                        )
                        
                        # Add summary artifacts only (not full data)
                        if "output_dir" in item and item["output_dir"]:
                            output_dir = item["output_dir"]
                            
                            # Create a concise summary instead of full data
                            summary_data = {
                                "status": "completed",
                                "charts_generated": charts_generated,
                                "output_directory": output_dir,
                                "data_source": item.get("data", {}).get("selected_file", "unknown"),
                                "chart_files": [os.path.basename(f) for f in item.get("data", {}).get("chart_files", [])],
                                "message": f"Successfully generated {charts_generated} visualizations"
                            }
                            
                            summary_parts = [Part(root=TextPart(text=json.dumps(summary_data, indent=2)))]
                            await updater.add_artifact(
                                summary_parts,
                                name="generation_summary",
                            )
                            
                            # Add charts count
                            count_parts = [Part(root=TextPart(text=str(charts_generated)))]
                            await updater.add_artifact(
                                count_parts,
                                name="charts_count",
                            )
                        
                        await updater.complete()
                        break
                        
                    except Exception as artifact_error:
                        logger.warning(f"Error adding artifacts: {artifact_error}")
                        await updater.complete()
                        break

        except Exception as e:
            logger.error(f"Error during dashboard visualization generation: {e}")
            error_message = f"An error occurred while generating visualizations: {str(e)}"
            error_parts = [Part(root=TextPart(text=error_message))]
            
            try:
                await updater.update_status(
                    TaskState.failed,
                    message=updater.new_agent_message(error_parts),
                )
            except Exception as update_error:
                logger.error(f"Failed to update task status: {update_error}")
            

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Log the cancellation attempt
        logger.info(f"Cancel requested for task {context.task_id}")
        raise ServerError(error=UnsupportedOperationError())
