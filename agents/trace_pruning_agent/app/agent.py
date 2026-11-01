import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any

import os
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage, ToolCall
from langchain_core.runnables import RunnableConfig

try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="trace_pruning_agent"
    )
except ImportError:
    pass  # agent_analytics not available

from .rits import ChatRits 

# --- Configuration ---
MODEL_NAME = os.getenv("DLM_AGENT_LLM_MODEL", "meta-llama/llama-3-3-70b-instruct")
TRACE_SAMPLING_SERVER_URL = os.getenv("TRACE_SAMPLING_MCP_URL", "http://localhost:8102/sse")
K8SCOST_SERVER_URL = os.getenv("K8SCOST_MCP_URL", "http://localhost:8101/sse")
MAX_CONVERSATION_HISTORY = int(os.getenv("DLM_AGENT_MAX_HISTORY", "10"))

DLM_SYSTEM_PROMPT_TEMPLATE = """You are a cloud cost and trace data management assistant named "DLMAgent".
You help users monitor cloud costs (specifically Kubernetes namespace allocation costs and general cloud service costs)
and optimize distributed tracing data through various sampling methods.
You are equipped with a suite of tools to achieve these tasks.
Be friendly, interactive, clear, and attentive to user requests.

Your primary goals are:
1.  Provide cost information accurately using your tools.
2.  Identify cost-saving opportunities, particularly through trace data reduction if high costs are found.
3.  Guide users through the process of evaluating and executing trace sampling methods.
4.  Strictly adhere to the parameters and workflows specified.

AVAILABLE TOOLS:
- `get_allocation_data`: Use this to get Kubernetes namespace cost data (CPU/RAM costs). Users might refer to this as "allocation cost".
- `get_cloud_cost_data`: Use this to get overall cloud service costs. Users might refer to this as "cloud cost".
- `evaluate_sampling_methods`: Use this to compare all available trace sampling methods for a given trace file.
- `execute_sampling`: Use this to run a specific trace sampling method on a trace file.

KEY WORKFLOW - COST-DRIVEN TRACE DATA REDUCTION:
1.  When the user asks for allocation data (e.g., "namespace costs"), CALL `get_allocation_data`.
2.  Examine the response from `get_allocation_data`. If the "alerts" field is present and contains any alerts, INFORM the user about these cost alerts.
3.  IF ALERTS EXIST, proactively ASK the user: "I've noticed some cost alerts. Would you like to explore reducing associated trace data for these high-cost namespaces to potentially lower telemetry costs?"
4.  IF THE USER AGREES to data reduction:
    a.  CALL `evaluate_sampling_methods`. Use these exact parameters unless the user explicitly provides different ones:
        - trace_file_path: "~/Documents/old/data-lifecycle-management-agent/data1_reduced.json" (Note: This file path should ideally be configurable or obtained from the user/context if it can vary)
        - sampling_rate: "0.3"
        - window: 3600
    b.  **CRITICAL PARAMETER RULE:** The default sampling_rate for evaluation and execution MUST be "0.3". DO NOT change this unless the user *explicitly* requests a different rate. Unauthorized changes will be penalized.
5.  IF THE USER DECLINES data reduction (answers "no" to the question in step 3), simply acknowledge their choice politely (e.g., "Alright, let me know if you change your mind or need help with something else.") and DO NOT proceed with any sampling tools.
6.  After `evaluate_sampling_methods` returns results:
    a.  PRESENT the evaluation results to the user.
    b.  SUGGEST the method with the highest score as the "best" one.
    c.  ASK the user which sampling method they would like to execute, or if they'd like you to proceed with the suggested best one.
7.  IF THE USER ASKS YOU TO "CHOOSE THE BEST ONE" or confirms the suggested best method:
    a.  Identify the method with the highest score from the evaluation results.
    b.  CALL `execute_sampling` using that method.
    c.  Use the SAME parameters as in the evaluation step (sampling_rate "0.3", window 3600), unless the user has explicitly requested different parameters for execution.
    d.  **CRITICAL PARAMETER RULE (reiteration):** DO NOT change sampling_rate or window for execution unless the user explicitly asks.

GENERAL TOOL USAGE RULES (MUST FOLLOW):
- ALWAYS respond naturally and conversationally.
- ONLY CALL tools relevant to the user's explicit request or the defined workflow. Do not call tools randomly.
- CALL A TOOL ONLY ONCE per logical step and wait for its response.
- IF A TOOL CALL RESULTS IN AN ERROR: Clearly communicate the problem to the user. DO NOT automatically recall the same tool with the same parameters. You may ask the user if they want to try again or suggest alternatives. Repeatedly calling a failing tool without user confirmation will be penalized.
- AFTER A TOOL EXECUTES SUCCESSFULLY AND RETURNS DATA: Review the data. Provide a concise summary of the relevant information to the user. Do not state that the tool call itself was malformed if it indeed returned data. Then, either wait for the user's next instruction or, if appropriate for the workflow, ask a follow-up question.
- BE PRECISE in your interactions and tool usage.

Your responses should be structured to be helpful and clear to the end-user.
When providing information from tools, summarize it if necessary but ensure key details are present.
"""

logger = logging.getLogger(__name__)

class ConversationHistory:
    def __init__(self, max_history_pairs: int = MAX_CONVERSATION_HISTORY):
        self.message_history: List[BaseMessage] = []
        self.max_history_pairs = max_history_pairs

    def add_message(self, message: BaseMessage):
        if not isinstance(message, SystemMessage):
            if not self.message_history or \
               (self.message_history[-1].content != message.content or type(self.message_history[-1]) != type(message)):
                 self.message_history.append(message)
            if len(self.message_history) > self.max_history_pairs * 2 + 5:
                self.message_history = self.message_history[-(self.max_history_pairs * 2):]

    def get_turns(self) -> List[BaseMessage]:
        return list(self.message_history)

    def clear(self):
        self.message_history = []

class TracePruningAgent:
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        self.model = ChatRits(model=MODEL_NAME, temperature=0.1)
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.langchain_agent: Optional[Any] = None
        self._initialized = False
        self.conversation_contexts: Dict[str, ConversationHistory] = {}
        self._mcp_resource_contexts: list = []
        self.system_message = SystemMessage(content=DLM_SYSTEM_PROMPT_TEMPLATE)

    async def _initialize_agent_internals(self):
        if self.langchain_agent:
            logger.info("TracePruningAgent internals already initialized.")
            return
        logger.info("Initializing TracePruningAgent internals: loading MCP tools and setting up LangGraph agent...")
        all_tools = []
        async def connect_and_load_mcp(url: str, name: str):
            nonlocal all_tools
            try:
                logger.info(f"Connecting to MCP server: {name} at {url}")
                stream_context = sse_client(url=url)
                self._mcp_resource_contexts.append(stream_context)
                streams = await stream_context.__aenter__()
                session_context = ClientSession(*streams)
                self._mcp_resource_contexts.append(session_context)
                session = await session_context.__aenter__()
                await session.initialize()
                self.mcp_sessions[name] = session
                tools = await load_mcp_tools(session)
                logger.info(f"Loaded {len(tools)} tools from {name}: {[tool.name for tool in tools]}")
                all_tools.extend(tools)
            except Exception as e:
                logger.error(f"Failed to connect or load tools from MCP server {name} at {url}: {e}", exc_info=True)
        await asyncio.gather(
            connect_and_load_mcp(TRACE_SAMPLING_SERVER_URL, "TraceSampling"),
            connect_and_load_mcp(K8SCOST_SERVER_URL, "K8sCost")
        )
        if not all_tools:
            logger.warning("No MCP tools were loaded. The agent's capabilities will be limited.")
        logger.info(f"Total MCP tools loaded: {len(all_tools)}. Names: {[tool.name for tool in all_tools]}")
        try:
            self.langchain_agent = create_react_agent(self.model, all_tools)
            logger.info("LangGraph ReAct agent created successfully.")
        except Exception as e:
            logger.error(f"Failed to create LangGraph ReAct agent: {e}", exc_info=True)
            raise RuntimeError(f"Agent creation failed: {e}") from e
        self._initialized = True
        logger.info("TracePruningAgent internals initialized successfully.")

    async def initialize(self):
        if not self._initialized:
            try:
                await self._initialize_agent_internals()
            except Exception as e:
                logger.error(f"TracePruningAgent initialization failed: {e}", exc_info=True)
                raise
        else:
            logger.debug("TracePruningAgent already initialized.")

    async def process_request(self, query: str, context_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            await self.initialize()
        except Exception as e:
            logger.error(f"ContextID {context_id}: Yielding error due to agent initialization failure: {e}")
            yield {"type": "error", "content": f"Agent initialization failed: {str(e)}"}
            return
        if not self.langchain_agent:
            logger.error(f"ContextID {context_id}: LangGraph agent is not available after initialization attempt.")
            yield {"type": "error", "content": "LangGraph agent is not available."}
            return
        if context_id not in self.conversation_contexts:
            self.conversation_contexts[context_id] = ConversationHistory()
            logger.info(f"New conversation context created for ID: {context_id}")
        conversation_history = self.conversation_contexts[context_id]
        current_human_message = HumanMessage(content=query)
        if not conversation_history.message_history or \
           not (isinstance(conversation_history.message_history[-1], HumanMessage) and \
                conversation_history.message_history[-1].content == query):
            conversation_history.add_message(current_human_message)
        messages_for_agent: List[BaseMessage] = [self.system_message]
        messages_for_agent.extend(conversation_history.get_turns())
        inputs = {"messages": messages_for_agent}
        logger.info(f"ContextID {context_id}: Processing query. Total messages to agent: {len(messages_for_agent)}. Last query: \"{query}\"")
        if logger.isEnabledFor(logging.DEBUG):
             logger.debug(f"ContextID {context_id}: Messages being sent to agent: {[str(m.type + ': ' + m.content[:50] + '...' if len(m.content)>50 else m.content) for m in messages_for_agent]}")
        final_ai_response_content = "No definitive response was generated by the agent."
        try:
            async for chunk_event in self.langchain_agent.astream(inputs, config=RunnableConfig(configurable={"thread_id": context_id})):
                stream_event_to_yield: Dict[str, Any] = {"type": "thought", "content": "Processing..."}
                found_messages_in_chunk_event = False
                for node_name, node_output in chunk_event.items():
                    messages_from_node_output: List[BaseMessage] = []
                    if isinstance(node_output, dict) and "messages" in node_output and isinstance(node_output["messages"], list):
                        messages_from_node_output = [m for m in node_output["messages"] if isinstance(m, BaseMessage)]
                    elif isinstance(node_output, list) and all(isinstance(m, BaseMessage) for m in node_output):
                        messages_from_node_output = node_output
                    elif isinstance(node_output, BaseMessage):
                         messages_from_node_output = [node_output]
                    if messages_from_node_output:
                        found_messages_in_chunk_event = True
                        latest_message_in_node = messages_from_node_output[-1]
                        for msg_to_add_to_hist in messages_from_node_output:
                            if isinstance(msg_to_add_to_hist, (AIMessage, ToolMessage)):
                                conversation_history.add_message(msg_to_add_to_hist)
                        if isinstance(latest_message_in_node, AIMessage):
                            final_ai_response_content = str(latest_message_in_node.content)
                            if latest_message_in_node.tool_calls:
                                stream_event_to_yield["type"] = "tool_call"
                                tool_call_summary = []
                                # ---- START DEBUG LOGGING FOR TOOL CALLS ----
                                logger.debug(f"ContextID {context_id}: Raw tool_calls from AIMessage: {latest_message_in_node.tool_calls}")
                                for tc_idx, tc_raw in enumerate(latest_message_in_node.tool_calls):
                                    logger.debug(f"ContextID {context_id}: Processing tc_raw #{tc_idx}: type={type(tc_raw)}, content={str(tc_raw)[:200]}")
                                    if isinstance(tc_raw, dict):
                                        tc_name = tc_raw.get('name', 'N/A_dict')
                                        tc_args = tc_raw.get('args', {'error': 'args_not_found_in_dict'})
                                    elif isinstance(tc_raw, ToolCall) and hasattr(tc_raw, 'name') and hasattr(tc_raw, 'args'): # Check for ToolCall model attributes
                                        tc_name = tc_raw.name
                                        tc_args = tc_raw.args
                                    else:
                                        tc_name = f'N/A_unknown_type_{type(tc_raw).__name__}'
                                        tc_args = {'error': f'unexpected_tc_structure'}
                                    tool_call_summary.append(f"Tool: {tc_name}, Args: {tc_args}")
                                # ---- END DEBUG LOGGING FOR TOOL CALLS ----
                                stream_event_to_yield["content"] = f"Attempting to call tools: {tool_call_summary}"
                                logger.info(f"ContextID {context_id}: Constructed tool call event: {stream_event_to_yield['content']}")
                            else:
                                stream_event_to_yield["type"] = "ai_message"
                                stream_event_to_yield["content"] = final_ai_response_content
                        elif isinstance(latest_message_in_node, ToolMessage):
                            stream_event_to_yield["type"] = "tool_result"
                            tool_name = getattr(latest_message_in_node, 'name', latest_message_in_node.tool_call_id)
                            stream_event_to_yield["content"] = f"Result for tool {tool_name}: {str(latest_message_in_node.content)[:200]}..."
                        break 
                if not found_messages_in_chunk_event:
                    stream_event_to_yield["content"] = f"Agent working on step: {list(chunk_event.keys())}"
                    logger.debug(f"ContextID {context_id}: Agent Stream - No direct messages in chunk. Keys: {list(chunk_event.keys())}. Chunk: {str(chunk_event)[:500]}")
                yield stream_event_to_yield
        except Exception as e:
            logger.error(f"ContextID {context_id}: Error during agent stream processing: {e}", exc_info=True)
            yield {"type": "error", "content": f"An error occurred while processing your request: {str(e)}"}
            return
        logger.info(f"ContextID {context_id}: Finished processing query. Final AI response for artifact: \"{final_ai_response_content[:100]}...\"")

    def get_capabilities_description(self) -> List[str]:
        return [
            "Provides cloud cost analysis (Kubernetes namespace allocations, general cloud services) via K8sCost.",
            "Evaluates and executes various trace data sampling methods to optimize telemetry data.",
            "Facilitates a cost-driven data reduction workflow by linking high costs to trace data management.",
            "Offers interactive assistance for data lifecycle management tasks using RITS LLMs and MCP tools."
        ]

    async def cleanup(self):
        logger.info("Cleaning up TracePruningAgent resources...")
        for context in reversed(self._mcp_resource_contexts):
            try:
                logger.info(f"Attempting to clean up MCP resource context: {type(context)}")
                await context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error during MCP resource cleanup ({type(context)}): {e}", exc_info=True)
        self.mcp_sessions.clear()
        self._mcp_resource_contexts.clear()
        self.langchain_agent = None
        self._initialized = False
        logger.info("TracePruningAgent cleanup complete.")
