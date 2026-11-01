import os
import json
import logging
import re
import asyncio
import time
from datetime import datetime
from typing import Any, AsyncIterable, Dict, List, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_litellm import ChatLiteLLM
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="graphql_query_agent"
    )
except ImportError:
    pass  # agent_analytics not available

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

perf_logger = logging.getLogger("performance")
perf_logger.setLevel(logging.INFO)

# Timeouts
HEARTBEAT_EVERY_SEC = 2
MAX_WAIT_SEC = 120
LLM_TIMEOUT_SEC = 60
SCHEMA_RETRIEVAL_TIMEOUT = 2

load_dotenv()

def log_timing(operation_name: str):
    """Decorator to log timing of operations."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                perf_logger.info(f"🟡 STARTING: {operation_name}")
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    perf_logger.info(f"🟢 COMPLETED: {operation_name} in {elapsed:.3f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    perf_logger.error(f"🔴 FAILED: {operation_name} after {elapsed:.3f}s - {e}")
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                perf_logger.info(f"🟡 STARTING: {operation_name}")
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    perf_logger.info(f"🟢 COMPLETED: {operation_name} in {elapsed:.3f}s")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    perf_logger.error(f"🔴 FAILED: {operation_name} after {elapsed:.3f}s - {e}")
                    raise
            return sync_wrapper
    return decorator

class ResponseFormat(BaseModel):
    status: Literal["input_required", "completed", "error"] = "completed"
    message: str
    graphql_query: Optional[str] = None
    title: Optional[str] = None
    report: Optional[str] = None

class ToolConverter:
    @staticmethod
    def to_langchain_tool(fn):
        return StructuredTool.from_function(
            func=fn,
            description="Execute a GraphQL query against the StepZen endpoint"
        )

@log_timing("GraphQL_Query_Execution")
def execute_graphql_query(query: str = None, args: list = None, kwargs: dict = None) -> str:
    """
    Execute a GraphQL query against the StepZen endpoint.

    Args:
        query: The GraphQL query string to execute (direct call)
        args: List of arguments (for LangChain tool calling)
        kwargs: Dict of keyword arguments (for LangChain tool calling)

    Returns:
        JSON string containing the query results or error information
    """
    # Handle both direct calls and LangChain tool calls
    if query is None and kwargs and 'query' in kwargs:
        query = kwargs['query']
    elif query is None:
        return json.dumps({"error": "No query provided"})

    logger.info(f"GraphQL Query Request: {query[:100]}...")

    url = os.getenv("GRAPHQL_ENDPOINT")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": (
            "apikey yangihissai::local.net+1000::"
            "86df16f80670c9145629d1e1656d3d4482803f3f395b2a9a3c5bb2a03de4baff"
        ),
    }

    start_time = time.time()
    try:
        logger.info("Making HTTP request to GraphQL endpoint...")
        res = requests.post(url=url, json={"query": query}, headers=headers, timeout=30)
        request_time = time.time() - start_time
        logger.info(f"HTTP request completed in {request_time:.3f}s, status: {res.status_code}")

        res.raise_for_status()
        response_data = res.json()

        logger.info(f"GraphQL response received: {len(str(response_data))} chars")
        return json.dumps(response_data)
    except requests.exceptions.Timeout:
        logger.error("GraphQL request timed out")
        return json.dumps({"error": "GraphQL query timed out"})
    except Exception as e:
        logger.error(f"GraphQL request failed: {e}")
        return json.dumps({"error": str(e)})


@log_timing("Schema_Index_Building")
def build_schema_index(base_schema_dir: str) -> Dict[str, str]:
    """Load GraphQL schemas with detailed logging."""
    logger.info(f"📁 Building schema index from: {base_schema_dir}")
    
    schema_map: Dict[str, str] = {}
    if not os.path.exists(base_schema_dir):
        logger.warning(f"⚠️ Schema directory does not exist: {base_schema_dir}")
        return schema_map

    file_count = 0
    total_size = 0
    
    for root, _dirs, files in os.walk(base_schema_dir):
        for file in files:
            if not file.endswith(".graphql"):
                continue
                
            file_path = os.path.join(root, file)
            logger.debug(f"📄 Reading schema file: {file}")
            
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                    # Much more aggressive size limit
                    if len(content) > 5000:  # 5KB limit per schema
                        original_size = len(content)
                        content = content[:5000] + "\n# ... (truncated)"
                        logger.info(f" Truncated {file} from {original_size} to 5000 chars")
                    relative_path = os.path.relpath(file_path, base_schema_dir)
                    schema_map[relative_path] = content
                    file_count += 1
                    total_size += len(content)
                    
            except Exception as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)

    logger.info(f"Schema index built: {file_count} files, {total_size:,} total chars")
    return schema_map

def split_camel_case(name: str) -> str:
    """Convert CamelCaseName -> Camel Case Name."""
    spaced = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", name))
    return " ".join(spaced.split())

# Tools
logger.info("Creating GraphQL query tool...")
execute_graphql_query_tool = ToolConverter.to_langchain_tool(execute_graphql_query)
TOOLS = [execute_graphql_query_tool]
logger.info(f"Tools created: {len(TOOLS)} tools available")

QUERY_GEN_PROMPT = """You are a GraphQL expert. Generate a query for this schema and request.

Schema:
{schema}

Request: {query}

Rules:
- For apm_applications: no boundaryScope/label unless app name specified
- Default timeframe: May 2025
- Use execute_graphql_query tool to test your query
- Keep query minimal

Response format:
### GraphQL Query
```graphql
[your query]
```

### Title
[brief title]

### Report
[one sentence explanation]
"""

class GraphQLQueryAgent:
    """LangGraph ReAct agent for GraphQL query generation."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(
        self,
        base_schema_dir: str = "./data/graphql",
        model_name: str = "mistralai/mistral-large",
        results_dir: str = "./data/results",
    ) -> None:
        logger.info("Initializing GraphQLQueryAgent...")
        start_time = time.time()
        
        os.makedirs(results_dir, exist_ok=True)

        # Check environment variables
        logger.info("Checking environment variables...")
        required_env = ["PROVIDER_API_KEY", "PROVIDER_URL", "PROVIDER_PROJECT_ID"]
        missing = [v for v in required_env if not os.getenv(v)]
        if missing:
            logger.error(f"Missing Provider env vars: {missing}")
            raise ValueError(f"Missing Provider env vars: {missing}")
        
        logger.info("All required environment variables found")

        # Initialize LLM
        self._initialize_llm(model_name)
        
        # Initialize LangGraph
        self._initialize_langgraph()
        
        # Initialize schema components
        self._initialize_schema_components(base_schema_dir)

        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        logger.info("🧵 Thread pool executor created")

        elapsed = time.time() - start_time
        logger.info(f"🎉 GraphQLQueryAgent initialization completed in {elapsed:.3f}s")

    @log_timing("LLM_Initialization")
    def _initialize_llm(self, model_name: str):
        """Initialize LLM with detailed logging."""
        logger.info(f"🤖 Initializing LLM with model: {model_name}")
        
        # Log environment details (without exposing secrets)
        provider_url = os.getenv("PROVIDER_URL")
        provider_project = os.getenv("PROVIDER_PROJECT_ID")
        logger.info(f"🔗 Provider URL: {provider_url}")
        logger.info(f"📋 Provider Project: {provider_project}")
        
        # Use correct Provider model format
        provider_model = f"provider/{model_name}"
        logger.info(f"🏷️ Using model identifier: {provider_model}")

        try:
            self.model = ChatLiteLLM(
                model=provider_model,
                temperature=0.0,
                max_tokens=1500,
                litellm_params={
                    "api_key": os.getenv("PROVIDER_API_KEY"),
                    "api_base": os.getenv("PROVIDER_URL"),
                    "project_id": os.getenv("PROVIDER_PROJECT_ID"),
                    "timeout": LLM_TIMEOUT_SEC,
                    "max_retries": 1,
                },
            )
            logger.info(f"✅ ChatLiteLLM initialized successfully")
            
            # Test if we can access the model (without making a full call)
            logger.info("🧪 Testing LLM accessibility...")
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            logger.error(f"📊 Error type: {type(e).__name__}")
            raise

    @log_timing("LangGraph_Initialization") 
    def _initialize_langgraph(self):
        """Initialize LangGraph with detailed logging."""
        logger.info("🕸️ Initializing LangGraph ReAct agent...")
        
        try:
            self.graph = create_react_agent(
                model=self.model,
                tools=TOOLS,
                store=InMemoryStore(),
                version="v2",
            )
            logger.info("✅ LangGraph ReAct agent created successfully")
        except Exception as e:
            logger.error(f"❌ LangGraph initialization failed: {e}")
            raise

    @log_timing("Schema_Components_Initialization")
    def _initialize_schema_components(self, base_schema_dir: str):
        """Initialize schema-related components."""
        logger.info("📚 Initializing schema components...")
        
        # Initialize sentence transformer
        logger.info("🧠 Loading sentence transformer model...")
        try:
            self.sentence_transformer = SentenceTransformer("all-mpnet-base-v2")
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.error(f"❌ Sentence transformer loading failed: {e}")
            raise
        
        # Load schemas
        self.schema_map = build_schema_index(base_schema_dir)
        
        # Pre-compute embeddings
        self._precompute_schema_embeddings()

    @log_timing("Schema_Embeddings_Precomputation")
    def _precompute_schema_embeddings(self):
        """Skip embeddings computation for speed."""
        logger.info("⚡ Skipping schema embeddings computation for speed")
        self._schema_embeddings_cache = {}

        # Simple keyword-based schema selection instead
        self._schema_keywords = {}
        for path in self.schema_map.keys():
            filename = os.path.basename(path).lower()
            if 'apm' in filename:
                self._schema_keywords[path] = ['apm', 'application', 'infrastructure']
            elif 'resilience' in filename:
                self._schema_keywords[path] = ['vulnerability', 'security', 'application']
            elif 'optimizer' in filename:
                self._schema_keywords[path] = ['entity', 'optimizer']
            else:
                self._schema_keywords[path] = ['general']

    def _precompute_schema_embeddings_old(self):
        """Pre-compute embeddings for faster schema retrieval."""
        logger.info("Pre-computing schema embeddings...")
        
        self._schema_embeddings_cache = {}
        
        for i, path in enumerate(self.schema_map.keys(), 1):
            logger.debug(f"Computing embedding {i}/{len(self.schema_map)}: {path}")
            
            readable = split_camel_case(os.path.basename(path).replace(".graphql", ""))
            emb = self.sentence_transformer.encode([readable], show_progress_bar=False)[0]
            self._schema_embeddings_cache[path] = emb
            
        logger.info(f"Pre-computed embeddings for {len(self._schema_embeddings_cache)} schemas")

    @log_timing("Schema_Retrieval")
    def _find_relevant_schema(self, question: str, k: int = 1) -> str:
        """Fast keyword-based schema selection."""
        if not self.schema_map:
            return ""

        question_lower = question.lower()

        # Simple keyword matching - much faster than embeddings
        if 'apm' in question_lower or 'application' in question_lower:
            # Find apm schema
            for path, schema in self.schema_map.items():
                if 'apm' in path.lower() and 'applications' in path.lower():
                    return schema[:8000]  # Truncate for speed

            # Fallback to any apm schema
            for path, schema in self.schema_map.items():
                if 'apm' in path.lower():
                    return schema[:8000]

        elif 'vulnerabilit' in question_lower:
            for path, schema in self.schema_map.items():
                if 'resilience' in path.lower() and 'vulnerabilit' in path.lower():
                    return schema[:8000]

        # Default: return first schema
        first_schema = list(self.schema_map.values())[0]
        return first_schema[:8000]

    def _find_relevant_schema_old(self, question: str, k: int = 1) -> str:
        """Find relevant schema with detailed logging."""
        logger.info(f"Finding relevant schema for: {question[:50]}...")
        
        if not self.schema_map:
            logger.warning("No schemas available")
            return ""

        try:
            # Encode user question
            logger.debug("Encoding user question...")
            user_emb = self.sentence_transformer.encode([question], show_progress_bar=False)[0]
            
            # Calculate similarities
            logger.debug("Calculating similarities...")
            scores: List[tuple[str, float]] = []
            
            for path, schema_emb in self._schema_embeddings_cache.items():
                sim = float(np.dot(user_emb, schema_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(schema_emb)))
                scores.append((path, sim))
            
            # Get best matches
            best_paths = [p for p, s in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]
            logger.info(f"Selected {len(best_paths)} best matching schemas")
            
            for i, (path, score) in enumerate(sorted(scores, key=lambda x: x[1], reverse=True)[:3]):
                logger.debug(f"   {i+1}. {path}: {score:.3f}")
            
            # Combine schemas
            combined_schema = "\n\n".join(self.schema_map[p] for p in best_paths)
            
            # Check size and truncate if needed
            original_size = len(combined_schema)
            if original_size > 15000:
                combined_schema = combined_schema[:15000] + "\n# ... (truncated for performance)"
                logger.info(f"Schema truncated from {original_size:,} to 15,000 chars")
            else:
                logger.info(f"Combined schema size: {original_size:,} chars")
            
            return combined_schema
            
        except Exception as e:
            logger.error(f"Schema retrieval failed: {e}")
            return ""

    @log_timing("LangGraph_Sync_Execution")
    def _run_sync(self, schema: str, question: str) -> tuple[str, str, str]:
        """Synchronous execution with detailed logging."""
        logger.info("Starting synchronous LangGraph execution...")
        
        prompt = QUERY_GEN_PROMPT.format(schema=schema, query=question)
        prompt_size = len(prompt)
        logger.info(f"Prompt size: {prompt_size:,} chars")
        
        inputs = {"messages": [("user", prompt)]}
        cfg = RunnableConfig(recursion_limit=5)  # Reduced from 100, then 10
        
        try:
            logger.info("🏃 Starting LangGraph stream...")
            event_count = 0
            
            events = []
            for event in self.graph.stream(inputs, stream_mode="values", config=cfg):
                event_count += 1
                events.append(event)
                logger.debug(f"📥 Received event #{event_count}")
                
                # Log the content of each event for debugging
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, "content"):
                        content_preview = str(last_message.content)[:100]
                        logger.debug(f"   Content preview: {content_preview}...")
                        
                        # Check if we have a GraphQL query in this event
                        if "```graphql" in str(last_message.content):
                            logger.info("Found GraphQL query in event!")
            
            logger.info(f"LangGraph execution completed with {event_count} events")
            return self._extract_final(events)
            
        except Exception as e:
            logger.error(f"LangGraph execution failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return "", "Error", f"Failed to generate query: {str(e)}"

    @staticmethod
    def _extract_final(events) -> tuple[str, str, str]:
        """Extract final result from LangGraph events with logging."""
        logger.info(f"🔍 Extracting final result from {len(events)} events...")
        
        for i, ev in enumerate(events[::-1]):  # Check from last event backwards
            logger.debug(f"🔍 Checking event #{len(events) - i}")
            
            if "messages" not in ev or not ev["messages"]:
                continue
                
            msg = ev["messages"][-1]
            if not hasattr(msg, "content"):
                continue
                
            content = msg.content
            logger.debug(f"📄 Message content length: {len(content)} chars")
            
            if "```graphql" in content:
                logger.info("✅ Found GraphQL query block!")
                
                # Extract GraphQL query
                gql_match = re.search(r"```graphql(.*?)```", content, re.S)
                gql = gql_match.group(1).strip() if gql_match else ""
                
                # Extract title (more flexible patterns)
                title_match = re.search(r"(?:Title|title):\s*(.+?)(?:\n|$)", content)
                title = title_match.group(1).strip() if title_match else "GraphQL Query"
                
                # Extract report (more flexible patterns)
                report_match = re.search(r"(?:Report|report):\s*(.+?)(?:\n|$)", content)
                report = report_match.group(1).strip() if report_match else "Query generated"
                
                if gql:  # Only return if we found a GraphQL query
                    return gql, title, report
        
        return "", "", ""

    def invoke(self, query: str, _context_id: str) -> Dict[str, Any]:
        """Synchronous invocation with detailed logging."""
        logger.info(f"Invoke called with query: {query}")
        overall_start = time.time()
        
        try:
            schema = self._find_relevant_schema(query)
            gql, title, report = self._run_sync(schema, query)
            
            elapsed = time.time() - overall_start
            logger.info(f"⏱️ Total invoke time: {elapsed:.3f}s")
            
            if gql:
                logger.info("✅ Invoke successful")
                return {
                    "graphql_query": gql,
                    "title": title,
                    "report": report,
                    "status": "completed",
                    "elapsed_seconds": elapsed
                }
            else:
                logger.warning("⚠️ Invoke completed but no query generated")
                return {
                    "status": "error",
                    "message": "Failed to generate query",
                    "elapsed_seconds": elapsed
                }
        except Exception as e:
            elapsed = time.time() - overall_start
            logger.error(f"❌ Invoke failed after {elapsed:.3f}s: {e}")
            return {
                "status": "error",
                "message": f"Invoke failed: {str(e)}",
                "elapsed_seconds": elapsed
            }

    @log_timing("Agent_Stream_Total")
    async def stream(self, query: str, context_id: str) -> AsyncIterable[Dict[str, Any]]:
        """Optimized streaming with detailed logging."""
        logger.info(f"Stream started for query: {query}")
        logger.info(f"Context ID: {context_id}")
        start_time = asyncio.get_event_loop().time()
        
        # Very fast schema retrieval
        yield {"is_task_complete": False, "content": "Finding schema..."}
        
        try:
            schema = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, self._find_relevant_schema, query
                ),
                timeout=SCHEMA_RETRIEVAL_TIMEOUT
            )
            
            schema_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Schema retrieval completed in {schema_time:.3f}s")
            
        except asyncio.TimeoutError:
            logger.error("Schema retrieval timed out")
            yield {
                "is_task_complete": True,
                "content": "Schema retrieval timed out. Please try again.",
            }
            return

        yield {"is_task_complete": False, "content": "Generating GraphQL query..."}

        # Query generation
        try:
            gql, title, report = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, self._run_sync, schema, query
                ),
                timeout=MAX_WAIT_SEC
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"Stream completed in {elapsed:.3f}s")
            
            if gql:
                full_response = (
                    f"**Query Generated in {elapsed:.1f}s**\n\n"
                    f"### GraphQL Query\n```graphql\n{gql}\n```\n\n"
                    f"### Title\n{title}\n\n"
                    f"### Report\n{report}"
                )
                
                yield {
                    "is_task_complete": True,
                    "content": full_response,
                    "graphql_query": gql,
                    "title": title,
                    "report": report,
                    "elapsed_seconds": elapsed
                }
            else:
                yield {
                    "is_task_complete": True,
                    "content": f"Failed to generate query after {elapsed:.1f}s. {report}",
                }
                
        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"Query generation timed out after {elapsed:.3f}s")
            yield {
                "is_task_complete": True,
                "content": f"Timeout after {elapsed:.1f}s. Please try a simpler query.",
            }
        except Exception as e:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(f"Stream error after {elapsed:.1f}s: {e}")
            yield {
                "is_task_complete": True,
                "content": f"Error after {elapsed:.1f}s: {str(e)}",
            }

# Test function for debugging
async def test_agent_detailed():
    """Detailed test function to see what's happening."""
    logger.info("Starting detailed agent test...")
    
    try:
        agent = GraphQLQueryAgent()
        logger.info("Agent created successfully")
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        return
    
    test_query = "Get all applications from APM"
    logger.info(f"Testing with query: {test_query}")
    
    try:
        overall_start = time.time()
        step_count = 0
        
        async for result in agent.stream(test_query, "debug-test"):
            step_count += 1
            elapsed = time.time() - overall_start
            
            content = result.get('content', '')
            is_complete = result.get('is_task_complete', False)
            
            logger.info(f"Step #{step_count} at {elapsed:.3f}s - Complete: {is_complete}")
            logger.info(f"Content: {content[:100]}...")
            
            if is_complete:
                logger.info(f"Test completed in {elapsed:.3f}s after {step_count} steps")
                
                if result.get('graphql_query'):
                    logger.info(f"GraphQL query generated: {len(result['graphql_query'])} chars")
                else:
                    logger.warning("No GraphQL query in final result")
                break
                
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    # Set up detailed logging for the test
    logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(test_agent_detailed())
