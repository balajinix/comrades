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
from app.agent import GraphQLSchemaMappingAgent
from app.agent_executor import GraphQLSchemaMappingAgentExecutor
from dotenv import load_dotenv

load_dotenv()

try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="schema_mapping_agent"
    )
except ImportError:
    pass  # agent_analytics not available

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts GraphQL Schema Mapping Agent server."""
    host = "localhost"
    port = 8002
    try:
        # Validate environment variables
        required_keys = ["PROVIDER_APIKEY", "PROVIDER_PROJECT_ID"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            raise MissingAPIKeyError(f"Missing environment variables: {', '.join(missing_keys)}")

        # Create agent capabilities - DISABLE STREAMING
        capabilities = AgentCapabilities(streaming=False, pushNotifications=True)
        
        skill = AgentSkill(
            id="map_graphql_schema",
            name="GraphQL Schema Mapping",
            description="Maps and transforms GraphQL schemas between different systems and formats",
            tags=["graphql", "schema", "mapping", "transformation"],
            examples=[
                "map resilience vulnerabilities to optimizer market actions",
                "map resilience vulnerabilities to optimizer entities"
            ],
        )
        
        agent_card = AgentCard(
            name="GraphQL_Schema_Mapping_Agent",
            description="Maps and transforms GraphQL schemas across different systems and formats",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=GraphQLSchemaMappingAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=GraphQLSchemaMappingAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # Create HTTP client
        httpx_client = httpx.AsyncClient(timeout=30.0)
        push_config_store = InMemoryPushNotificationConfigStore()

        # Create request handler
        request_handler = DefaultRequestHandler(
            agent_executor=GraphQLSchemaMappingAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=BasePushNotificationSender(httpx_client, push_config_store),
        )
        
        # Create server
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )

        logger.info(f"Starting GraphQL Schema Mapping Agent on {host}:{port}")
        logger.info(f"Streaming disabled, capabilities: {capabilities}")
        
        print(f"Starting GraphQL Schema Mapping Agent on {host}:{port}")
        print("Streaming: DISABLED (to avoid queue issues)")
        
        # Start server
        uvicorn.run(
            server.build(), 
            host=host, 
            port=port,
            log_level="info"
        )

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
