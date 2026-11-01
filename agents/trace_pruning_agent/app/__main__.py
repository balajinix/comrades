import logging
import os
import sys
import click
import httpx
import uvicorn
import asyncio # For cleanup

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="trace_pruning_agent_server"
    )
except ImportError:
    pass  # agent_analytics not available

from app.agent import TracePruningAgent
from app.agent_executor import TracePruningAgentExecutor
from dotenv import load_dotenv

load_dotenv()


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


AGENT_NAME = "Trace Pruning Agent"
AGENT_DESCRIPTION = (
    "An intelligent assistant for managing cloud costs (K8sCost) and "
    "trace data sampling. It uses RITS models and MCP tools."
)
AGENT_VERSION = "1.0.0"

agent_skills = [
    AgentSkill(
        id="get_allocation_data",
        name="Get Namespace Allocation Costs",
        description="Retrieves CPU and RAM cost data for Kubernetes namespaces. Can also identify cost alerts.",
        tags=["k8scost", "kubernetes", "cost", "allocation", "namespace", "cpu", "ram", "alerts"],
        examples=[
            "Show me the allocation costs for the 'prod' namespace.",
            "Are there any cost alerts in my cluster?",
            "Get CPU/RAM costs."
        ]
    ),
    AgentSkill(
        id="get_cloud_cost_data",
        name="Get Cloud Service Costs",
        description="Fetches overall cloud service costs from K8sCost.",
        tags=["k8scost", "cloud", "cost", "aws", "gcp", "azure", "spending"],
        examples=[
            "What were my cloud costs last week?",
            "Show me the cloud cost data for the finance department."
        ]
    ),
    AgentSkill(
        id="evaluate_sampling_methods",
        name="Evaluate Trace Sampling Methods",
        description="Compares different trace sampling methods for a given trace file, sampling rate, and window.",
        tags=["trace", "sampling", "otel", "opentelemetry", "telemetry", "evaluate", "compare", "performance", "cost-saving"],
        examples=[
            "Evaluate sampling methods for 'traces.json' with rate 0.3 and window 3600.",
            "Compare sampling techniques for my trace data."
        ]
    ),
    AgentSkill(
        id="execute_sampling_method",
        name="Execute Trace Sampling Method",
        description="Applies a specific trace sampling method to a trace file with given parameters.",
        tags=["trace", "sampling", "otel", "opentelemetry", "telemetry", "execute", "apply", "reduce data"],
        examples=[
            "Execute random sampling on 'traces.json' with rate 0.3, window 3600.",
            "Run head sampling for my trace file."
        ]
    ),
    AgentSkill(
        id="data_reduction_workflow",
        name="Cost-Driven Data Reduction Workflow",
        description="Identifies high-cost namespaces and offers to reduce associated trace data via sampling.",
        tags=["workflow", "cost optimization", "data reduction", "automation", "sampling"],
        examples=[
            "Check for high costs and help me reduce data.",
            "Optimize my trace data based on allocation costs."
        ]
    )
]

@click.command()
@click.option('--host', 'host', default=os.getenv('AGENT_HOST', 'localhost'))
@click.option('--port', 'port', default=int(os.getenv('AGENT_PORT', 8004)))  # Changed port
def main(host: str, port: int):
    """Starts the DLM Agent server using A2A protocol."""
    logger.info(f"Starting {AGENT_NAME} server v{AGENT_VERSION} on http://{host}:{port}")

    agent_capabilities = AgentCapabilities(
        streaming=True,  # Set to True because process_request is an async generator
        pushNotifications=True, # Standard to enable
        stateTransitionHistory=True # Useful to see task history
    )

    agent_card = AgentCard(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        url=f'http://{host}:{port}/',
        version=AGENT_VERSION,
        defaultInputModes=TracePruningAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=TracePruningAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=agent_capabilities,
        skills=agent_skills,
    )

    agent_executor = TracePruningAgentExecutor()

    httpx_client = httpx.AsyncClient()
    push_config_store = InMemoryPushNotificationConfigStore()

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
        push_config_store=push_config_store,
        push_sender=BasePushNotificationSender(httpx_client, push_config_store),
    )
    
    server = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )

    async def on_shutdown():
        logger.info("Server is shutting down. Cleaning up agent resources...")
        await agent_executor.cleanup() 
        await httpx_client.aclose()   
        logger.info("Agent resources cleaned up.")

    app = server.build()
    app.add_event_handler("shutdown", on_shutdown)

    logger.info(f"Agent Card: {agent_card.model_dump_json(indent=2, exclude_none=True)}")
    logger.info(f"{AGENT_NAME} ready to serve requests!")
    logger.info("Available capabilities via MCP and Langchain:")
    
    # Create agent instance to get capabilities
    agent_instance = TracePruningAgent()
    if hasattr(agent_instance, 'get_capabilities_description'):
        for cap_desc in agent_instance.get_capabilities_description(): 
            logger.info(f"{cap_desc}")
    else:
        logger.info("K8sCost integration for cost monitoring")
        logger.info("Trace sampling and evaluation")
        logger.info("MCP protocol support")

    uvicorn.run(app, host=host, port=port, log_config=None) 

if __name__ == "__main__":
    main()
