import asyncio
import json
import uuid
import os
import logging
from datetime import datetime
from typing import Any, AsyncIterable, List, Dict

import httpx
import nest_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug file for tracking initialization
DEBUG_FILE = "/tmp/orchestrator_debug.log"
def debug_log(msg):
    """Write debug message to file and logger."""
    with open(DEBUG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")
    logger.info(msg)
    print(msg)  # Also print

# Log module import at module level
debug_log("=== orchestrator.agent module being imported ===")
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Provider model configuration - uses LiteLLM for non-Gemini models
PROVIDER_MODEL = os.getenv("PROVIDER_MODEL", "provider/ibm/granite-3-8b-instruct")
USE_PROVIDER = os.getenv("USE_PROVIDER", "true").lower() == "true"

try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="comrades_dal_orchestrator"
    )
except ImportError:
    pass  # agent_analytics not available

from .remote_agent_connection import RemoteAgentConnections
from .dal_tools import (
    coordinate_data_access,
    get_access_status,
    generate_dal_report,
)

load_dotenv()
nest_asyncio.apply()


def get_agent_urls() -> List[str]:
    """Get agent URLs from environment variables or use defaults."""
    # Check for environment variable with comma-separated URLs
    env_urls = os.getenv("AGENT_URLS")
    if env_urls:
        return [url.strip() for url in env_urls.split(",") if url.strip()]
    
    # Default URLs - try common ports
    default_urls = [
        "http://localhost:8001",  # GraphQL Query Agent
        "http://localhost:8002",  # Schema Mapping Agent
        "http://localhost:8003",  # Data Dashboard Agent
        "http://localhost:8004",  # Trace Pruning Agent
        "http://localhost:8005",  # DAL Agent (mcp_dal via federated kg_agent)
    ]
    
    return default_urls


class DALOrchestrator:
    def __init__(self):
        self.remote_agent_connections: Dict[str, RemoteAgentConnections] = {}
        self.cards: Dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent()
        self._user_id = "comrades_dal_orchestrator"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        """Initialize connections to available agents."""
        successful_connections = 0

        async with httpx.AsyncClient(timeout=10) as client:  # Shorter timeout
            for address in remote_agent_addresses:
                try:
                    print(f"Attempting to connect to agent at {address}...")

                    # Try to get agent card from multiple possible endpoints
                    card = await self._resolve_agent_card(client, address)

                    if card:
                        card = card.model_copy(update={"memory": {"type": "ephemeral"}})
                        remote_connection = RemoteAgentConnections(
                            agent_card=card, agent_url=address
                        )
                        self.remote_agent_connections[card.name] = remote_connection
                        self.cards[card.name] = card
                        successful_connections += 1
                        print(f"Successfully connected to {card.name} at {address}")
                    else:
                        print(f"Could not resolve agent card at {address}")

                except httpx.ConnectError as e:
                    print(f"Failed to connect to {address}: Connection refused")
                except httpx.TimeoutException:
                    print(f"Failed to connect to {address}: Timeout")
                except Exception as e:
                    print(f"Error initializing {address}: {e}")

        # Set the agents string based on connection results
        if successful_connections == 0:
            print("Warning: No agents are currently available")
            self.agents = "No agents currently available. Please start agents and restart orchestrator."
        else:
            print(f"Successfully connected to {successful_connections} agent(s)")
            agent_info = [
                json.dumps({
                    "name": card.name,
                    "description": card.description,
                })
                for card in self.cards.values()
            ]
            self.agents = "\n".join(agent_info)

    async def _resolve_agent_card(self, client: httpx.AsyncClient, address: str) -> AgentCard:
        """Try multiple endpoints to resolve agent card."""
        # Endpoints to try in order of preference
        endpoints = [
            "/.well-known/agent.json",      # A2A 0.2.x protocol
            "/.well-known/agent-card.json", # A2A 0.3.x protocol
        ]

        for endpoint in endpoints:
            try:
                url = f"{address.rstrip('/')}{endpoint}"
                response = await client.get(url)
                if response.status_code == 200:
                    card_data = response.json()
                    # Parse the agent card
                    card = AgentCard.model_validate(card_data)
                    print(f"  Found agent card at {endpoint}")
                    return card
            except Exception as e:
                # Try next endpoint
                continue

        # Fallback: try A2ACardResolver
        try:
            card_resolver = A2ACardResolver(client, address)
            return await card_resolver.get_agent_card()
        except Exception:
            return None

    @classmethod
    async def create(cls, remote_agent_addresses: List[str] = None):
        """Create orchestrator instance with flexible agent discovery."""
        if remote_agent_addresses is None:
            remote_agent_addresses = get_agent_urls()
            
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def determine_best_agent(self, user_message: str) -> str:
        """
        Enhanced smart routing logic to determine the best agent for a user message.
        Returns the agent name that should handle the request.
        """
        user_message_lower = user_message.lower()
        
        # Define keyword patterns for each agent with more specific matching
        
        # Trace Pruning Agent keywords (highest priority for cost/trace queries)
        trace_pruning_keywords = [
            'namespace allocation costs', 'allocation costs', 'namespace costs', 
            'cost alerts', 'cloud costs', 'k8scost', 'kubernetes costs',
            'trace sampling', 'sampling methods', 'trace data', 'trace optimization',
            'reduce trace data', 'trace pruning', 'cost optimization',
            'namespace allocation', 'cpu costs', 'ram costs', 'memory costs',
            'cost reduction', 'data reduction', 'telemetry costs'
        ]
        
        # Dashboard Agent keywords (visualization focus)
        dashboard_keywords = [
            'dashboard', 'visualize', 'visualization', 'chart', 'charts', 'graph', 'graphs',
            'plot', 'plotting', 'create visualization', 'generate charts',
            'performance metrics', 'trends', 'overview', 'comprehensive dashboard',
            'interactive charts', 'data visualization', 'visual', 'analytics',
            'show me data', 'display data', 'create charts', 'generate dashboard',
            'endpoint visualizations', 'application performance', 'service metrics',
            'endpoint metrics', 'performance trends', 'metrics overview'
        ]
        
        graphql_keywords = [
            'graphql', 'query', 'generate query', 'fetch data', 'database query',
            'apm_applications', 'graphql query', 'data query', 'sql query',
            'query generation', 'database', 'fetch', 'retrieve data'
        ]
        
        schema_keywords = [
            'map', 'mapping', 'transform', 'schema mapping', 'convert schema',
            'optimizer', 'resilience', 'vulnerabilities to', 'schema transformation',
            'data mapping', 'convert', 'transform data'
        ]
        
        # Score each agent based on keyword matches
        scores = {}
        
        # Trace Pruning Agent scoring (highest priority for cost-related queries)
        trace_score = 0
        for keyword in trace_pruning_keywords:
            if keyword in user_message_lower:
                trace_score += 5  # High weight for exact matches
        
        # Special handling for "allocation costs" - should always go to trace pruning
        if any(phrase in user_message_lower for phrase in ['allocation costs', 'namespace costs', 'cost alerts']):
            trace_score += 10
        
        if trace_score > 0:
            scores['Trace Pruning Agent'] = trace_score
        
        # Dashboard Agent scoring (only if not a cost query)
        if trace_score == 0:  # Only consider dashboard if not a cost query
            dashboard_score = 0
            for keyword in dashboard_keywords:
                if keyword in user_message_lower:
                    dashboard_score += 3
            
            # Check for visualization indicators
            viz_indicators = ['show me', 'display', 'visualize', 'chart', 'dashboard']
            for indicator in viz_indicators:
                if indicator in user_message_lower:
                    dashboard_score += 2
            
            if dashboard_score > 0:
                scores['Dashboard_Visualization_Agent'] = dashboard_score
        
        # GraphQL Agent scoring
        graphql_score = sum(2 if keyword in user_message_lower else 0 for keyword in graphql_keywords)
        if graphql_score > 0:
            scores['GraphQL Query Agent'] = graphql_score
        
        # Schema Mapping Agent scoring
        schema_score = sum(2 if keyword in user_message_lower else 0 for keyword in schema_keywords)
        if schema_score > 0:
            scores['GraphQL_Schema_Mapping_Agent'] = schema_score
        
        # Return the agent with the highest score
        if scores:
            best_agent = max(scores, key=scores.get)
            print(f"🎯 Enhanced routing: '{user_message[:50]}...' -> {best_agent} (score: {scores[best_agent]})")
            return best_agent
        
        # Fallback: if no clear match, check for general data terms
        data_terms = ['data', 'metrics', 'performance', 'analysis', 'report']
        if any(term in user_message_lower for term in data_terms):
            # If it contains cost-related terms, go to trace pruning
            if any(cost_term in user_message_lower for cost_term in ['cost', 'allocation', 'namespace']):
                print(f"🎯 Fallback to Trace Pruning for cost-related query: '{user_message[:50]}...'")
                return 'Trace Pruning Agent'
            else:
                print(f"🎯 Fallback to Dashboard for general data query: '{user_message[:50]}...'")
                return 'Dashboard_Visualization_Agent'
        
        # No clear match - let the LLM decide
        print(f"🤔 No clear routing match for: '{user_message[:50]}...', using LLM routing")
        return None

    def create_agent(self) -> Agent:
        # Use Provider via LiteLLM to avoid Gemini rate limits
        if USE_PROVIDER:
            model = LiteLlm(model=PROVIDER_MODEL)
            logger.info(f"🔧 Using Provider model: {PROVIDER_MODEL}")
        else:
            model = "gemini-2.5-flash"
            logger.info("🔧 Using Gemini model: gemini-2.5-flash")

        return Agent(
            model=model,
            name="COMRADES_DAL_Orchestrator",
            instruction=self.root_instruction,
            description="Orchestrates data access layer operations across specialized agents.",
            tools=[
                self.send_message,
                coordinate_data_access,
                get_access_status,
                generate_dal_report,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        available_agents_text = self.agents if self.agents else "No agents currently available"
 
        return f"""
        Role: You are the COMRADES DAL Orchestrator, coordinating data access layer operations across specialized agents.

        Available Agent Types:
        - GraphQL Query Agent: Generates and executes GraphQL queries from natural language
        - GraphQL_Schema_Mapping_Agent: Maps and transforms data schemas
        - Dashboard_Visualization_Agent: Creates visualizations and dashboards from data
        - Trace Pruning Agent: Analyzes traces, optimizes performance, and manages cloud costs

        IMPORTANT AGENT CAPABILITIES:

        Dashboard_Visualization_Agent:
        - Has INTELLIGENT FILE MATCHING: Can automatically find the right data file based on user requests
        - Has INTERNAL DATA ACCESS: Already has access to application metrics, endpoint metrics, and service metrics
        - Does NOT need CSV data provided - it will find and use appropriate data files automatically
        - Can handle visualization requests without additional data
        - Can handle requests like "create endpoint visualizations" or "show application performance" without additional data

        Trace Pruning Agent:
        - Handles cost monitoring and optimization queries
        - Manages namespace allocation costs, cloud costs, and cost alerts
        - Provides trace data sampling and optimization
        - Handles Kubernetes cost analysis and resource optimization

        ROUTING RULES:
        - For cost-related queries (allocation costs, namespace costs, cost alerts): Use Trace Pruning Agent
        - For visualization, dashboard, chart, or "show me" requests: Use Dashboard_Visualization_Agent
        - For GraphQL query generation: Use GraphQL Query Agent
        - For schema mapping/transformation: Use GraphQL_Schema_Mapping_Agent

        CRITICAL ROUTING INSTRUCTIONS:
        - Trace Pruning Agent handles:
          * "Show me namespace allocation costs"
          * "Are there any cost alerts?"
          * "What are my cloud costs?"
          * "Optimize trace data"
          * "Cost reduction workflows"

        - Dashboard_Visualization_Agent handles:
          * "Create endpoint visualizations"
          * "Show application performance metrics"
          * "Generate service health dashboard"
          * "Visualize performance trends"
          * "Create comprehensive dashboard"

        - GraphQL_Schema_Mapping_Agent handles:
          * "Map resilience vulnerabilities to optimizer entities"
          * "Transform schemas between systems"
          * "Schema mapping requests"

        - Do NOT ask for additional data when routing to agents
        - Pass user requests directly to the appropriate agent

        Core Directives:
        - Analyze requests and delegate tasks to appropriate agents
        - Use send_message tool to communicate with available agents
        - Coordinate multi-agent workflows for complex operations
        - If no agents are available, inform the user and suggest starting agents
        - Always check which agents are actually available before routing

        Today's Date: {datetime.now().strftime("%Y-%m-%d")}

        <Currently Available Agents>
        {available_agents_text}
        </Currently Available Agents>

        REMEMBER: 
        - Trace Pruning Agent handles cost and resource optimization queries
        - Dashboard_Visualization_Agent handles visualization and chart requests
        - Both agents have their own data access capabilities
        """

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """Enhanced stream with smart routing preprocessing."""

        # Check if we should route directly to a specific agent
        best_agent = self.determine_best_agent(query)

        if best_agent and best_agent in self.remote_agent_connections:
            print(f"🚀 Direct routing to {best_agent} for: {query[:50]}...")

            # Create a direct message to the best agent
            try:
                # Create a temporary tool context for direct routing
                class MockToolContext:
                    def __init__(self):
                        self.state = {
                            "task_id": str(uuid.uuid4()),
                            "context_id": str(uuid.uuid4())
                        }

                mock_context = MockToolContext()
                result = await self.send_message(best_agent, query, mock_context)

                yield {
                    "is_task_complete": True,
                    "content": result,
                }
                return

            except Exception as e:
                print(f"Direct routing failed: {e}, falling back to LLM routing")
                # Fall through to normal LLM routing

        # Normal LLM-based routing
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )

        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "Orchestrator is coordinating agents...",
                }

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Send message to specified agent with A2A-compatible artifact handling."""
        if not self.remote_agent_connections:
            error_msg = "No agents are currently available. Please start some agents and restart the orchestrator."
            return error_msg

        if agent_name not in self.remote_agent_connections:
            available_agents = list(self.remote_agent_connections.keys())
            error_msg = f"Agent '{agent_name}' not found. Available agents: {available_agents}. Please use exact agent names."
            return error_msg

        client = self.remote_agent_connections[agent_name]
        if not client:
            error_msg = f"Client not available for {agent_name}"
            return error_msg

        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        task_request = self._create_structured_task(agent_name, task, task_id)

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task_request}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )

        try:
            print(f"Sending message to {agent_name}...")
            
            send_response: SendMessageResponse = await client.send_message(message_request)
            print(f"send_response from {agent_name}:", send_response)

            if not isinstance(send_response.root, SendMessageSuccessResponse) or not isinstance(
                send_response.root.result, Task
            ):
                print(f"Received a non-success or non-task response from {agent_name}. Cannot proceed.")
                return f"Error: Non-success response from {agent_name}"

            response_content = send_response.root.model_dump_json(exclude_none=True)
            json_content = json.loads(response_content)

            print(f"=== PROCESSING RESPONSE FROM {agent_name} ===")
            print(f"JSON content keys: {json_content.keys()}")

            # Simple artifact extraction (following A2A sample pattern)
            resp = []
            if json_content.get("result", {}).get("artifacts"):
                artifacts = json_content["result"]["artifacts"]
                print(f"Found {len(artifacts)} artifacts")
                
                for artifact in artifacts:
                    if artifact.get("parts"):
                        resp.extend(artifact["parts"])
                        print(f"Added {len(artifact['parts'])} parts to response")

            # If we got parts, return them joined as text
            if resp:
                print(f"=== RETURNING {len(resp)} PARTS FROM {agent_name} ===")
                # Convert parts to text if they're objects
                text_parts = []
                for part in resp:
                    if isinstance(part, dict):
                        # Handle different part types
                        if part.get("type") == "text" and part.get("text"):
                            text_parts.append(part["text"])
                        elif part.get("content"):
                            text_parts.append(str(part["content"]))
                        else:
                            text_parts.append(str(part))
                    elif isinstance(part, str):
                        text_parts.append(part)
                    else:
                        text_parts.append(str(part))
                
                return "\n".join(text_parts) if text_parts else "No readable content in artifacts"
            
            # Fallback: check for direct content in result
            result = json_content.get("result", {})
            if result.get("content"):
                print(f"No artifacts, but found direct content from {agent_name}")
                return result["content"]
            
            print(f"No artifacts or content found from {agent_name}")
            return f"Task completed by {agent_name} (no content returned)"

        except Exception as e:
            error_msg = f"Error communicating with {agent_name}: {str(e)}"
            print(error_msg)
            return error_msg

    async def send_message2(self, agent_name: str, task: str, tool_context: ToolContext):
        """Send message to specified agent with better error handling."""
        if not self.remote_agent_connections:
            error_msg = "No agents are currently available. Please start some agents and restart the orchestrator."
            return error_msg

        if agent_name not in self.remote_agent_connections:
            available_agents = list(self.remote_agent_connections.keys())
            error_msg = f"Agent '{agent_name}' not found. Available agents: {available_agents}. Please use exact agent names."
            return error_msg

        client = self.remote_agent_connections[agent_name]
        if not client:
            error_msg = f"Client not available for {agent_name}"
            return error_msg

        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        task_request = self._create_structured_task(agent_name, task, task_id)

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task_request}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )

        try:
            print(f"Sending message to {agent_name}...")
            
            send_response: SendMessageResponse = await client.send_message(message_request)

            if not isinstance(send_response.root, SendMessageSuccessResponse) or not isinstance(
                send_response.root.result, Task
            ):
                error_msg = f"Error: Non-success response from {agent_name}"
                return error_msg

            response_content = send_response.root.model_dump_json(exclude_none=True)
            json_content = json.loads(response_content)

            print(f"=== PROCESSING RESPONSE FROM {agent_name} ===")

            artifacts_found = []
            if json_content.get("result", {}).get("artifacts"):
                artifacts = json_content["result"]["artifacts"]
                print(f"Found {len(artifacts)} artifacts")

                for i, artifact in enumerate(artifacts):
                    print(f"Processing artifact {i}: {type(artifact)}")

                    # Handle different artifact structures
                    if isinstance(artifact, dict):
                        content = None

                        if artifact.get("parts"):
                            parts = artifact["parts"]
                            if isinstance(parts, list) and len(parts) > 0:
                                content = str(parts[0])
                        elif artifact.get("content"):
                            content = artifact["content"]
                        elif artifact.get("text"):
                            content = artifact["text"]
                        elif artifact.get("data"):
                            content = str(artifact["data"])

                        # Apply size limits to prevent token overflow
                        if content:
                            if len(content) > 50000:  # 50KB limit
                                print(f"  Large artifact detected ({len(content)} chars), truncating...")
                                content = content[:10000] + "\n... (Large response truncated to prevent token limit)"
                            artifacts_found.append(content)

                    elif isinstance(artifact, str):
                        # Direct string artifact - also apply size limits
                        if len(artifact) > 50000:
                            print(f"  Large string artifact detected ({len(artifact)} chars), truncating...")
                            artifact = artifact[:10000] + "\n... (Large response truncated)"
                        artifacts_found.append(artifact)
                    else:
                        # Unknown artifact type
                        print(f"  Unknown artifact type: {type(artifact)}")
                        artifacts_found.append(str(artifact))

            # Also check for direct content in result
            result = json_content.get("result", {})
            if result.get("content") and not artifacts_found:
                print("No artifacts found, checking result.content")
                content = result["content"]
                if len(content) > 50000:
                    content = content[:10000] + "\n... (Large response truncated)"
                artifacts_found.append(content)

            # Format the response
            if artifacts_found:
                print(f"=== RETURNING {len(artifacts_found)} ARTIFACTS ===")
                # Join multiple artifacts with separators
                if len(artifacts_found) == 1:
                    return artifacts_found[0]
                else:
                    response_parts = []
                    for i, artifact in enumerate(artifacts_found):
                        response_parts.append(f"=== Artifact {i+1} ===\n{artifact}")
                    return "\n\n".join(response_parts)
            else:
                print("No artifacts found, returning completion message")
                return f"Task completed by {agent_name} (no artifacts returned)"

        except Exception as e:
            error_msg = f"Error communicating with {agent_name}: {str(e)}"
            return error_msg

    def _create_structured_task(self, agent_name: str, task: str, task_id: str) -> str:
        timestamp = datetime.now().isoformat()

        task_templates = {
            "GraphQL Query Agent": f"GRAPHQL QUERY REQUEST\nTask ID: {task_id}\nTimestamp: {timestamp}\n\n{task}",
            "GraphQL_Schema_Mapping_Agent": f"SCHEMA MAPPING REQUEST\nTask ID: {task_id}\nTimestamp: {timestamp}\n\n{task}",
            "Dashboard_Visualization_Agent": f"VISUALIZATION REQUEST\nTask ID: {task_id}\nTimestamp: {timestamp}\n\n{task}",
            "Trace Pruning Agent": f"TRACE ANALYSIS REQUEST\nTask ID: {task_id}\nTimestamp: {timestamp}\n\n{task}",
        }

        return task_templates.get(agent_name, f"REQUEST\nTask ID: {task_id}\nTimestamp: {timestamp}\n\n{task}")

    def get_available_agents(self) -> Dict[str, str]:
        """Get currently available agents."""
        return {name: card.description for name, card in self.cards.items()}


# Global orchestrator instance for lazy initialization
_orchestrator_instance: DALOrchestrator = None
_initialization_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None


async def _get_or_create_orchestrator() -> DALOrchestrator:
    """Lazily initialize the orchestrator on first use."""
    global _orchestrator_instance

    if _orchestrator_instance is not None:
        return _orchestrator_instance

    agent_urls = get_agent_urls()
    print(f"🚀 Initializing COMRADES DAL Orchestrator with agents at: {agent_urls}")
    _orchestrator_instance = await DALOrchestrator.create(agent_urls)
    print(f"✅ Orchestrator initialized with {len(_orchestrator_instance.cards)} connected agents")
    return _orchestrator_instance


def _get_initialized_orchestrator_sync():
    """
    Initialize the orchestrator synchronously for module-level root_agent.
    Handles both standalone and event-loop contexts.
    """
    async def _async_main():
        debug_log("_async_main() started")
        agent_urls = get_agent_urls()
        debug_log(f"🚀 Initializing with agents at: {agent_urls}")

        try:
            orchestrator_instance = await DALOrchestrator.create(agent_urls)
            debug_log(f"✅ DALOrchestrator.create() completed with {len(orchestrator_instance.cards)} agents")
            debug_log(f"   Connected agents: {list(orchestrator_instance.cards.keys())}")
            debug_log(f"   orchestrator.agents string: {orchestrator_instance.agents[:200] if orchestrator_instance.agents else 'EMPTY'}...")

            # Store the instance globally for later use
            global _orchestrator_instance
            _orchestrator_instance = orchestrator_instance

            return orchestrator_instance._agent
        except Exception as e:
            debug_log(f"❌ Error in _async_main: {e}")
            import traceback
            debug_log(traceback.format_exc())
            raise

    # Try multiple approaches to run the async initialization
    debug_log("=== _get_initialized_orchestrator_sync called ===")
    try:
        # First try: standard asyncio.run (works when no event loop exists)
        debug_log("🔄 Attempting asyncio.run() initialization...")
        result = asyncio.run(_async_main())
        debug_log(f"✅ asyncio.run() succeeded, got: {type(result)}")
        return result
    except RuntimeError as e:
        debug_log(f"⚠️  asyncio.run() failed: {e}")
        if "cannot be called from a running event loop" in str(e) or "no running event loop" in str(e).lower():
            debug_log("⚠️  Existing/no event loop detected, using alternative initialization...")
            try:
                # Second try: use existing event loop with nest_asyncio
                try:
                    loop = asyncio.get_running_loop()
                    debug_log(f"  Found running loop: {loop}")
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    debug_log(f"  Created new loop: {loop}")

                if loop.is_running():
                    debug_log("  Loop is running, using thread pool...")
                    # Create a new loop in a thread for initialization
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _async_main())
                        result = future.result(timeout=30)
                        debug_log(f"✅ Thread pool init succeeded: {type(result)}")
                        return result
                else:
                    debug_log("  Loop not running, using run_until_complete...")
                    result = loop.run_until_complete(_async_main())
                    debug_log(f"✅ run_until_complete succeeded: {type(result)}")
                    return result
            except Exception as e2:
                debug_log(f"⚠️  Alternative initialization failed: {e2}")
                import traceback
                debug_log(traceback.format_exc())
                # Final fallback: create uninitialized agent that will connect lazily
                debug_log("📋 Creating agent with lazy remote connection initialization...")
                return _create_lazy_agent()
        else:
            debug_log(f"❌ Unexpected RuntimeError: {e}")
            raise


def _create_lazy_agent() -> Agent:
    """
    Create an agent that initializes remote connections lazily on first tool call.
    This is a fallback when sync initialization isn't possible.
    """
    # Create a basic orchestrator without async initialization
    orchestrator = DALOrchestrator.__new__(DALOrchestrator)
    orchestrator.remote_agent_connections = {}
    orchestrator.cards = {}
    orchestrator.agents = "Agents will be discovered on first request..."

    # Use Provider via LiteLLM to avoid Gemini rate limits
    if USE_PROVIDER:
        model = LiteLlm(model=PROVIDER_MODEL)
        logger.info(f"🔧 Lazy agent using Provider model: {PROVIDER_MODEL}")
    else:
        model = "gemini-2.5-flash"
        logger.info("🔧 Lazy agent using Gemini model: gemini-2.5-flash")

    # Create the agent with lazy initialization in send_message
    agent = Agent(
        model=model,
        name="COMRADES_DAL_Orchestrator",
        instruction=lambda ctx: _lazy_instruction(orchestrator, ctx),
        description="Orchestrates data access layer operations across specialized agents.",
        tools=[
            lambda agent_name, task, tool_context: _lazy_send_message(orchestrator, agent_name, task, tool_context),
            coordinate_data_access,
            get_access_status,
            generate_dal_report,
        ],
    )

    # Store reference
    orchestrator._agent = agent

    return agent


async def _ensure_connections(orchestrator: DALOrchestrator):
    """Ensure remote agent connections are initialized."""
    if not orchestrator.remote_agent_connections:
        print("🔌 Lazily initializing remote agent connections...")
        await orchestrator._async_init_components(get_agent_urls())


def _ensure_connections_sync(orchestrator: DALOrchestrator):
    """Synchronously ensure connections are initialized."""
    if orchestrator.remote_agent_connections:
        logger.info("Connections already initialized")
        return  # Already initialized

    logger.info("🔌 Synchronously initializing remote agent connections...")

    # Use httpx sync client for synchronous initialization
    agent_urls = get_agent_urls()
    successful_connections = 0

    with httpx.Client(timeout=10) as client:
        for address in agent_urls:
            try:
                # Try to get agent card
                for endpoint in ["/.well-known/agent.json", "/.well-known/agent-card.json"]:
                    try:
                        url = f"{address.rstrip('/')}{endpoint}"
                        response = client.get(url)
                        if response.status_code == 200:
                            card_data = response.json()
                            card = AgentCard.model_validate(card_data)
                            card = card.model_copy(update={"memory": {"type": "ephemeral"}})

                            # Create connection
                            remote_connection = RemoteAgentConnections(
                                agent_card=card, agent_url=address
                            )
                            orchestrator.remote_agent_connections[card.name] = remote_connection
                            orchestrator.cards[card.name] = card
                            successful_connections += 1
                            logger.info(f"  ✅ Connected to {card.name} at {address}")
                            break
                    except Exception as e:
                        logger.debug(f"  Endpoint {endpoint} failed: {e}")
                        continue
            except Exception as e:
                logger.warning(f"  ❌ Failed to connect to {address}: {e}")

    if successful_connections > 0:
        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in orchestrator.cards.values()
        ]
        orchestrator.agents = "\n".join(agent_info)
        logger.info(f"✅ Initialized {successful_connections} agent connections")
    else:
        orchestrator.agents = "No agents currently available. Please check agent status."
        logger.warning("⚠️  No agents could be connected")


def _lazy_instruction(orchestrator: DALOrchestrator, context: ReadonlyContext) -> str:
    """Instruction that reflects current connection state."""
    debug_log(f"=== _lazy_instruction called, orchestrator.agents={orchestrator.agents[:50] if orchestrator.agents else 'empty'}... ===")

    # Ensure connections are initialized before generating instruction
    _ensure_connections_sync(orchestrator)

    debug_log(f"After ensure_connections_sync, orchestrator.agents={orchestrator.agents[:100] if orchestrator.agents else 'empty'}...")
    available_agents_text = orchestrator.agents if orchestrator.agents else "Initializing connections..."

    return f"""
    Role: You are the COMRADES DAL Orchestrator, coordinating data access layer operations across specialized agents.

    Available Agent Types:
    - GraphQL Query Agent: Generates and executes GraphQL queries from natural language
    - GraphQL_Schema_Mapping_Agent: Maps and transforms data schemas
    - Dashboard_Visualization_Agent: Creates visualizations and dashboards from data
    - Trace Pruning Agent: Analyzes traces, optimizes performance, and manages cloud costs

    IMPORTANT AGENT CAPABILITIES:

    Dashboard_Visualization_Agent:
    - Has INTELLIGENT FILE MATCHING: Can automatically find the right data file based on user requests
    - Has INTERNAL DATA ACCESS: Already has access to application metrics, endpoint metrics, and service metrics
    - Does NOT need CSV data provided - it will find and use appropriate data files automatically

    Trace Pruning Agent:
    - Handles cost monitoring and optimization queries
    - Manages namespace allocation costs, cloud costs, and cost alerts

    ROUTING RULES:
    - For cost-related queries (allocation costs, namespace costs, cost alerts): Use Trace Pruning Agent
    - For visualization, dashboard, chart, or "show me" requests: Use Dashboard_Visualization_Agent
    - For GraphQL query generation: Use GraphQL Query Agent
    - For schema mapping/transformation: Use GraphQL_Schema_Mapping_Agent

    Today's Date: {datetime.now().strftime("%Y-%m-%d")}

    <Currently Available Agents>
    {available_agents_text}
    </Currently Available Agents>
    """


async def _lazy_send_message(orchestrator: DALOrchestrator, agent_name: str, task: str, tool_context: ToolContext):
    """Send message with lazy connection initialization."""
    await _ensure_connections(orchestrator)
    return await orchestrator.send_message(agent_name, task, tool_context)


# Initialize the root agent
root_agent = _get_initialized_orchestrator_sync()
