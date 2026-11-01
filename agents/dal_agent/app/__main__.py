import logging
import os
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from app.agent import DALAgent
from app.agent_executor import DALAgentExecutor
from app import faults as comrades_faults

load_dotenv()

# COMRADES experiments only: if COMRADES_FAULT_RATE is set, inject backend faults so
# failure propagation through the networked multi-agent chain can be measured.
comrades_faults.install_from_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts the Data Access Layer (DAL) A2A agent server."""
    host = os.getenv("DAL_AGENT_HOST", "localhost")
    port = int(os.getenv("DAL_AGENT_PORT", "8005"))
    try:
        # The agent reaches an LLM provider through kg_agent. Provider is the
        # default route; warn (don't hard-fail) if creds are absent so the
        # server can still be discovered and report errors per request.
        if not (os.getenv("PROVIDER_API_KEY") or os.getenv("PROVIDER_APIKEY")
                or os.getenv("OPENAI_API_KEY")):
            logger.warning("No LLM credentials found (PROVIDER_API_KEY / OPENAI_API_KEY). "
                           "The DAL agent will start but cannot answer until creds are set.")

        capabilities = AgentCapabilities(streaming=False, pushNotifications=True)

        skill = AgentSkill(
            id="query_data_access_layer",
            name="Heterogeneous Data Access",
            description=(
                "Natural-language access to heterogeneous observability, cloud-cost, "
                "ITSM, resilience/compliance, and cloud-infrastructure backends via the "
                "mcp_dal MCP server. Federates and chains tools across systems."
            ),
            tags=["dal", "observability", "finops", "itsm", "mcp", "data-access"],
            examples=[
                "list up to 3 ITSM incidents",
                "what are my top cloud costs this month",
                "show the resilience posture for application X",
                "get the application topology from the observability platform",
            ],
        )

        agent_card = AgentCard(
            name="DAL_Agent",
            description=(
                "Federated Data Access Layer agent: natural-language queries over "
                "heterogeneous enterprise and hyperscaler data systems via mcp_dal."
            ),
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=DALAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DALAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        httpx_client = httpx.AsyncClient(timeout=120.0)
        push_config_store = InMemoryPushNotificationConfigStore()

        request_handler = DefaultRequestHandler(
            agent_executor=DALAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=BasePushNotificationSender(httpx_client, push_config_store),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        logger.info(f"Starting DAL Agent on {host}:{port} "
                    f"(mcp_dal at {os.getenv('DAL_MCP_URL', 'http://localhost:8201/mcp')})")
        print(f"Starting DAL Agent on {host}:{port}")

        uvicorn.run(server.build(), host=host, port=port, log_level="info")

    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
