import logging
import os
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="graphql_query_agent_server"
    )
except ImportError:
    pass  # agent_analytics not available

from app.agent import GraphQLQueryAgent
from app.agent_executor import GraphQLQueryAgentExecutor
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger("graphql_query_agent")

# --------------------------------------------------------------------------- #
HOST = os.getenv("AGENT_HOST", "localhost")
PORT = int(os.getenv("AGENT_PORT", "8001"))


def build_agent_card() -> AgentCard:
    """Metadata the A2A host will read."""
    capabilities = AgentCapabilities(streaming=False, pushNotifications=True)

    skill = AgentSkill(
        id="generate_graphql_query",
        name="GraphQL Query Generation",
        description="Translates natural-language requests into executable GraphQL queries "
                    "and returns a short report.",
        tags=["graphql", "query", "provider"],
        examples=[
            "List applications created in May 2025",
            "Show error metrics for Quote of the Day",
        ],
    )

    return AgentCard(
        name="GraphQL Query Agent",
        description="LLM-powered assistant that crafts and executes GraphQL queries.",
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        defaultInputModes=GraphQLQueryAgent.SUPPORTED_CONTENT_TYPES,
        defaultOutputModes=GraphQLQueryAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=capabilities,
        skills=[skill],
    )


def main() -> None:
    logger.info("Starting GraphQL Query Agent …")

    httpx_client = httpx.AsyncClient(timeout=120.0)
    push_config_store = InMemoryPushNotificationConfigStore()

    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=DefaultRequestHandler(
            agent_executor=GraphQLQueryAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=BasePushNotificationSender(httpx_client, push_config_store),
        ),
    ).build()

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
