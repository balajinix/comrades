import logging
import os
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotificationConfigStore, BasePushNotificationSender
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# Agent Analytics imports
try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="dashboard_agent_server"
    )
except ImportError:
    pass  # agent_analytics not available

from app.agent import DashboardVisualizationAgent
from app.agent_executor import DashboardVisualizationAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts Dashboard Visualization Agent server."""
    host = "localhost"
    port = 8003
    
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id="generate_data_visualizations",
            name="Data Visualization & Dashboard Generation",
            description="Processes CSV data and generates interactive visualizations and dashboards",
            tags=["visualization", "dashboard", "data-analysis", "vega-lite", "charts"],
            examples=[
                "Create visualizations from sales_data.csv showing revenue trends",
                "Generate a dashboard overview of customer demographics",
                "Visualize performance metrics from system logs",
                "Create charts showing correlation between variables in dataset"
            ],
        )
        agent_card = AgentCard(
            name="Dashboard_Visualization_Agent",
            description="Processes CSV data and generates interactive visualizations, charts, and dashboards using Vega-Lite",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=DashboardVisualizationAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DashboardVisualizationAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        request_handler = DefaultRequestHandler(
            agent_executor=DashboardVisualizationAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=BasePushNotificationSender(httpx_client, push_config_store),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        print(f"Starting Dashboard Visualization Agent on {host}:{port}")
        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
