import os
import json
import requests
import re
import numpy as np
import asyncio
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from sentence_transformers import SentenceTransformer
from app.utils.provider_utils import get_wxai_response
from app.utils.schema_indexer import build_schema_index
from app.utils.toolkit_utils import parse_graphql_schema

from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from litellm import completion
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

# MCP imports
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

# MINIMAL Agent Analytics - Just like the calculator example
try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="schema_mapping_agent"
    )
except ImportError:
    pass  # agent_analytics not available

load_dotenv()
logger = logging.getLogger("gateway_ai_api")

# Updated MCP server URL to match your server
MCP_SERVER_URL = "http://127.0.0.1:3001/sse"  # Updated to match your server port

class MCPServer:
    def __init__(self):
        self.stream_contexts = []
        self.session_contexts = []
        self.sessions = []
    
    async def connect_to_sse_server(self, url):
        stream_context = sse_client(url=url)
        streams = await stream_context.__aenter__()
        session_context = ClientSession(*streams)
        session = await session_context.__aenter__()
        await session.initialize()
        
        self.stream_contexts.append(stream_context)
        self.session_contexts.append(session_context)
        self.sessions.append(session)
        return session
    
    async def cleanup(self):
        for session_context in self.session_contexts:
            if session_context:
                await session_context.__aexit__(None, None, None)
        for stream_context in self.stream_contexts:
            if stream_context:
                await stream_context.__aexit__(None, None, None)

# LLM wrapper for provider.ai via litellm
def call_llm(messages, model="provider/mistralai/mistral-large"):
    """Call provider.ai via litellm"""
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return f"Error calling LLM: {str(e)}"

# State definition for LangGraph
class AgentState(TypedDict):
    messages: Annotated[List[Any], "The messages in the conversation using LangChain message types"]
    utterance: str
    original_utterance: str  # Keep original for storage routing
    schemas: List[str]
    schema_map: Dict[str, str]
    schema_similarities: Optional[List[tuple]]
    source_schema: Optional[str]
    target_schema: Optional[str]
    source_schema_content: Optional[str]
    target_schema_content: Optional[str]
    schema_selection_reasoning: Optional[str]
    mapping_result: Optional[Dict[str, Any]]
    store_where: Optional[str]  # "graphdb", "wiki", "both", "none"
    storage_results: Optional[Dict[str, Any]]  # Results from storage operations
    error: Optional[str]

def split_camel_case(name: str):
    split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
    return " ".join(split)

# The vendored Map-Reflect agent replaces the schema-mapping-toolkit `solve`.
# We drive its field-mapping core directly: find_mappings already holds both full
# schemas, so no utterance/retrieval is needed -- we pass each schema's fields as
# source/destination properties and let the agent's LLM mapper align them.
_MAP_REFLECT_MODEL = os.getenv("SCHEMA_MAPPING_MODEL", "meta-llama/llama-3-3-70b-instruct")
_map_reflect_agent = None


def _get_map_reflect_agent():
    """Lazily build one MapReflectAgent and reuse it across calls.

    `_find_mappings` reads property_data/source/destination from the passed state
    rather than from `self`, so a single instance can map any schema pair. We init
    with a placeholder lookup and a scratch chroma dir we never query.
    """
    global _map_reflect_agent
    if _map_reflect_agent is None:
        # Lazy import: the in-process engine pulls heavy deps (chromadb, sentence-
        # transformers). The MAP_REFLECT_MCP_URL server path never touches this, so a
        # thin client only needs the `mcp` client, not the full mapping stack.
        from app.map_reflect.map_reflect_agent import MapReflectAgent
        import tempfile
        scratch = tempfile.mkdtemp(prefix="mapreflect_")
        lookup_path = os.path.join(scratch, "placeholder_lookup.json")
        with open(lookup_path, "w") as fh:
            json.dump(
                {"placeholder_P.f": {"product_name": "placeholder", "type_name": "P",
                                     "field_name": "f", "description": ""}},
                fh,
            )
        _map_reflect_agent = MapReflectAgent(
            property_lookup_file=lookup_path,
            model=_MAP_REFLECT_MODEL,
            chroma_dir=os.path.join(scratch, "chroma_db"),
        )
    return _map_reflect_agent


def _schema_to_lookup(schema_content, product):
    """Flatten a GraphQL schema into Map-Reflect's property-lookup format.

    Returns (lookup_dict, ordered_field_ids) keyed ``{product}_{Type}.{field}``.
    """
    type_defs, _enums, _scalars = parse_graphql_schema(schema_content)
    lookup, keys = {}, []
    for type_name, fields in type_defs.items():
        if type_name == "Query":
            continue
        for field_name, meta in fields.items():
            key = f"{product}_{type_name}.{field_name}"
            dtype = (meta or {}).get("data_type", "")
            desc = ((meta or {}).get("description") or "").strip() or f"{type_name}.{field_name} ({dtype})"
            lookup[key] = {
                "product_name": product,
                "type_name": type_name,
                "field_name": field_name,
                "description": desc,
            }
            keys.append(key)
    return lookup, keys


# Server path (the COMRADES default): when MAP_REFLECT_MCP_URL is set, the agent
# calls the deployed mcp_map_reflect MCP server instead of running map_reflect
# in-process -- mirroring how the DAL agent talks to mcp_dal. Unset -> in-process.
_MAP_REFLECT_MCP_URL = os.getenv("MAP_REFLECT_MCP_URL") or os.getenv("MCP_MAP_REFLECT_URL")


def _schema_types(schema_content):
    """Source type names declared in a GraphQL schema (excluding the Query root)."""
    type_defs, _enums, _scalars = parse_graphql_schema(schema_content)
    return [t for t in type_defs if t != "Query"]


def _unwrap_mcp(res, tool):
    """Unwrap an MCP tool result into a plain dict (structured output preferred)."""
    if getattr(res, "isError", False):
        msg = " ".join(str(getattr(c, "text", "")) for c in (res.content or []))
        raise RuntimeError(f"{tool} MCP error: {msg[:200]}")
    sc = getattr(res, "structuredContent", None)
    if isinstance(sc, dict):
        data = sc.get("result", sc)  # FastMCP wraps a non-dict return under "result"
    elif sc is not None:
        data = sc
    else:
        texts = [c.text for c in (res.content or []) if getattr(c, "type", "") == "text"]
        blob = "\n".join(texts).strip()
        data = json.loads(blob) if blob else {}
    # Tolerate an envelope like {"mappings": {...}, "reflection_feedback": ...}: unwrap
    # only when nothing at the top level already looks like a mapping list.
    if (isinstance(data, dict) and isinstance(data.get("mappings"), dict)
            and not any(isinstance(v, list) for v in data.values())):
        data = data["mappings"]
    return data if isinstance(data, dict) else {}


def _map_via_mcp(url, source_product, source_types, target_product):
    """Map each source type onto ``target_product`` via mcp_map_reflect's ``map_schema``.

    ``map_schema`` is per source type, so we call it once per type within a single MCP
    session and merge the results into one ``{src_field_id: [{mapProperty, ...}]}`` dict
    -- the same shape as the in-process candidate mappings, so the reshape is shared.
    Source/target must be products the server has pre-loaded (the itopsgraph products);
    for arbitrary schemas, unset MAP_REFLECT_MCP_URL to fall back to the in-process engine.
    """
    import anyio
    from mcp.client.streamable_http import streamablehttp_client

    async def _call():
        merged = {}
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for src_type in source_types:
                    res = await session.call_tool("map_schema", {
                        "source_product": source_product,
                        "source_type": src_type,
                        "target_product": target_product,
                    })
                    merged.update(_unwrap_mcp(res, "map_schema"))
        return merged

    return anyio.run(_call)


def find_mappings(source_graphql_schema, source_name, target_graphql_schema, target_name):
    """Map fields of the source GraphQL schema onto the target schema.

    Returns ``{"name": source_name, "properties": {source_field: [{mapProperty,...}]}}``,
    the same contract the previous toolkit-based implementation produced. Uses the
    deployed mcp_map_reflect server when MAP_REFLECT_MCP_URL is set, else the in-process
    vendored engine.
    """
    if _MAP_REFLECT_MCP_URL:
        src_types = _schema_types(source_graphql_schema)
        if not src_types:
            return {"name": source_name, "properties": {}}
        candidates = _map_via_mcp(_MAP_REFLECT_MCP_URL, source_name, src_types, target_name) or {}
    else:
        src_lookup, src_keys = _schema_to_lookup(source_graphql_schema, source_name)
        tgt_lookup, tgt_keys = _schema_to_lookup(target_graphql_schema, target_name)
        if not src_keys or not tgt_keys:
            return {"name": source_name, "properties": {}}
        state = {
            "user_utterance": f"Map the fields of the {source_name} schema to the {target_name} schema.",
            "property_data": {**src_lookup, **tgt_lookup},
            "source_properties": src_keys,
            "destination_properties": tgt_keys,
            "candidate_mappings": {},
            "final_mappings": {},
            "source_field_hints": [],
            "target_field_hints": [],
        }
        state = _get_map_reflect_agent()._find_mappings(state)
        candidates = state.get("candidate_mappings") or {}

    trimmed_result = {"name": source_name, "properties": {}}
    prefix = f"{source_name}_"
    for src_key, maps in candidates.items():
        if not isinstance(maps, list) or not maps:
            continue
        bare = src_key[len(prefix):] if src_key.startswith(prefix) else src_key
        trimmed_result["properties"][bare] = [
            {"mapProperty": m.get("mapProperty"),
             "justification": m.get("justification", ""),
             "confidence": m.get("confidence")}
            for m in maps if isinstance(m, dict)
        ]
    return trimmed_result

class GraphQLSchemaMappingAgent:
    """
    A GraphQL Mapping Agent using LangGraph for orchestration with MCP tool integration.
    """
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, base_schema_dir: str = "./data/graphql"):
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.schema_map = build_schema_index(base_schema_dir)
        self.mcp_tools = {} 
        self.mcp_server = MCPServer()  # Add this line
        self.graph = None

    async def async_init(self):
        """Async initialization for MCP tools and graph"""
        if self.graph:
            logger.info("Agent already initialized.")
            return
        
        logger.info("Initializing Agent: loading MCP tools...")
        
        try:
            # Connect to MCP server and maintain connection
            session = await self.mcp_server.connect_to_sse_server(MCP_SERVER_URL)
            
            # Load MCP tools
            from langchain_mcp_adapters.tools import load_mcp_tools
            mcp_tools = await load_mcp_tools(session)
            
            # Convert to simple dict
            self.mcp_tools = {tool.name: tool for tool in mcp_tools}
            logger.info(f"Loaded MCP tools: {list(self.mcp_tools.keys())}")
            print(f"[DEBUG] Loaded MCP tools: {list(self.mcp_tools.keys())}")  # TEMP

            
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            self.mcp_tools = {}
        
        self.graph = self._create_graph()
    
    async def cleanup(self):
        """Cleanup MCP connections"""
        if self.mcp_server:
            await self.mcp_server.cleanup()
        
    def _create_graph(self):
        """Create the LangGraph workflow with integrated tool functions."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("find_schemas", self._find_relevant_schemas_node)
        workflow.add_node("select_schemas", self._select_schemas_node)
        workflow.add_node("generate_mappings", self._generate_mappings_node)
        workflow.add_node("check_store_destination", self._check_store_destination)
        workflow.add_node("store_in_graphdb", self._store_in_graphdb_node)
        workflow.add_node("store_in_wiki", self._store_in_wiki_node)
        workflow.add_node("store_in_graphdb_and_wiki", self._store_in_graphdb_and_wiki_node)

        # Add edges
        workflow.set_entry_point("find_schemas")
        workflow.add_edge("find_schemas", "select_schemas")
        workflow.add_edge("select_schemas", "generate_mappings")
        workflow.add_edge("generate_mappings", "check_store_destination")

        # Conditional routing based on detected storage intent
        conditional_mapping = {
            "none": END,
            "graphdb": "store_in_graphdb",
            "wiki": "store_in_wiki",
            "both": "store_in_graphdb_and_wiki"
        }

        workflow.add_conditional_edges(
            "check_store_destination",
            self._route_storage_path,
            conditional_mapping
        )

        # Define edges for individual storage options
        workflow.add_edge("store_in_graphdb", END)
        workflow.add_edge("store_in_wiki", END)
        workflow.add_edge("store_in_graphdb_and_wiki", END)

        return workflow.compile()
    
    def _sanitize_utterance_for_mapping(self, utterance: str) -> str:
        """Remove storage instructions from utterance to avoid affecting mapping logic"""
        # Pattern to match storage-related instructions
        storage_patterns = [
            r"(and\s+)?(store|save|push|populate|update|put)\s+(in|into|to|the)?\s*(graphdb|graph|database|wiki|documentation)",
            r"(and\s+)?(store|save|populate)\s+(the\s+)?(result|mapping|data)\s+(in|into|to)\s*(graphdb|graph|database|wiki)",
            r"(and\s+)?(update|populate)\s+(the\s+)?(wiki|documentation|graphdb|graph)",
            r"store\s+(it|this|the\s+result)\s+(in|into|to)\s*(graphdb|wiki|graph)",
        ]
        
        cleaned_utterance = utterance
        for pattern in storage_patterns:
            cleaned_utterance = re.sub(pattern, "", cleaned_utterance, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and conjunctions
        cleaned_utterance = re.sub(r'\s+', ' ', cleaned_utterance)
        cleaned_utterance = re.sub(r'\s+(and|,)\s*$', '', cleaned_utterance, flags=re.IGNORECASE)
        
        return cleaned_utterance.strip()

    async def _check_store_destination(self, state: AgentState) -> AgentState:
        """Check where to store the mapping results based on utterance"""
        original_utterance = state.get("original_utterance", "").lower()
        actions = []

        # Check for GraphDB/graph database keywords
        graphdb_keywords = ["graphdb", "graph database", "graph db", "store in graph", "populate graphdb"]
        wiki_keywords = ["wiki", "documentation", "update wiki", "populate wiki", "generate wiki"]
        
        for keyword in graphdb_keywords:
            if keyword in original_utterance:
                actions.append("graphdb")
                break
                
        for keyword in wiki_keywords:
            if keyword in original_utterance:
                actions.append("wiki")
                break

        # Determine storage destination
        if "graphdb" in actions and "wiki" in actions:
            state["store_where"] = "both"
        elif "graphdb" in actions:
            state["store_where"] = "graphdb"
        elif "wiki" in actions:
            state["store_where"] = "wiki"
        else:
            state["store_where"] = "none"
        
        # Add system message about storage decision
        storage_decision = state["store_where"]
        if storage_decision != "none":
            system_content = f"Storage Decision: Will store mapping results in {storage_decision}"
            system_msg = SystemMessage(content=system_content)
            state["messages"].append(system_msg)
        
        return state

    def _route_storage_path(self, state: AgentState) -> str:
        """Route to appropriate storage path"""
        return state.get("store_where", "none")

    async def _store_in_graphdb_node(self, state: AgentState) -> AgentState:
        """Store mapping results in GraphDB using MCP tool"""
        try:
            mapping_result = state.get("mapping_result")
            source_schema = state.get("source_schema")
            target_schema = state.get("target_schema")
            
            if not mapping_result:
                state["error"] = "No mapping result to store in GraphDB"
                return state
            
            graphdb_tool = self.mcp_tools.get("populate_graphdb")
            if not graphdb_tool:
                state["error"] = "GraphDB tool not available"
                return state
            
            # Prepare data for GraphDB storage
            graphdb_data = {
                "mapping_name": mapping_result.get("name", f"{source_schema}_to_{target_schema}"),
                "source_schema": source_schema,
                "target_schema": target_schema,
                "mapping_properties": mapping_result.get("properties", {}),
            }
            
            # Call GraphDB tool
            graphdb_result = await graphdb_tool.ainvoke({"data": mapping_result})
            
            if not state.get("storage_results"):
                state["storage_results"] = {}
            state["storage_results"]["graphdb"] = graphdb_result
            
            # Add AI message about GraphDB storage
            ai_content = f"""✅ **GraphDB Storage Complete**

**Stored Mapping:**
- Name: {graphdb_data['mapping_name']}
- Source: {source_schema}
- Target: {target_schema}
- Properties: {len(graphdb_data['mapping_properties'])} mappings

**Storage Result:**
{graphdb_result}"""
            
            ai_msg = AIMessage(content=ai_content)
            state["messages"].append(ai_msg)
            
        except Exception as e:
            error_msg = f"Error storing in GraphDB: {str(e)}"
            state["error"] = error_msg
            error_ai_msg = AIMessage(content=f"❌ {error_msg}")
            state["messages"].append(error_ai_msg)
        
        return state

    async def _store_in_wiki_node(self, state: AgentState) -> AgentState:
        """Generate wiki documentation using MCP tool"""
        try:
            mapping_result = state.get("mapping_result")
            source_schema = state.get("source_schema")
            target_schema = state.get("target_schema")
            
            if not mapping_result:
                state["error"] = "No mapping result to document in wiki"
                return state
                
            wiki_tool = self.mcp_tools.get("generate_wiki")
            if not wiki_tool:
                state["error"] = "Wiki tool not available"
                return state
                    
            # Call Wiki tool
            wiki_result = await wiki_tool.ainvoke({"content": mapping_result})
            
            if not state.get("storage_results"):
                state["storage_results"] = {}
            state["storage_results"]["wiki"] = wiki_result
            
            # Add AI message about wiki generation
            ai_content = f"""📖 **Wiki Documentation Generated**
GraphQL Schema Mapping: {source_schema} to {target_schema}
**Generation Result:**
{wiki_result}"""
            
            ai_msg = AIMessage(content=ai_content)
            state["messages"].append(ai_msg)
            
        except Exception as e:
            error_msg = f"Error generating wiki documentation: {str(e)}"
            state["error"] = error_msg
            error_ai_msg = AIMessage(content=f"❌ {error_msg}")
            state["messages"].append(error_ai_msg)
        
        return state

    
    async def _store_in_graphdb_and_wiki_node(self, state: AgentState) -> AgentState:
        """Store mapping results in both GraphDB and Wiki using integrated functions"""
        try:
            # Store in GraphDB
            state = await self._store_in_graphdb_node(state)  # Add await

            # Store in Wiki
            state = await self._store_in_wiki_node(state)   # Add await

        except Exception as e:
            error_msg = f"Error storing in GraphDB and Wiki: {str(e)}"
            state["error"] = error_msg
            error_ai_msg = AIMessage(content=f"❌ {error_msg}")
            state["messages"].append(error_ai_msg)

        return state

    async def _find_relevant_schemas_node(self, state: AgentState) -> AgentState:
        """Node to find relevant schemas based on user utterance"""
        try:
            # Store original utterance for storage routing
            original_utterance = state["utterance"]
            state["original_utterance"] = original_utterance
            
            # Clean utterance for mapping logic
            cleaned_utterance = self._sanitize_utterance_for_mapping(original_utterance)
            state["utterance"] = cleaned_utterance
            
            schemas, schema_map, similarities = self._find_relevant_schema(cleaned_utterance)
            
            # Add system message about schema finding with detailed analysis
            similarity_info = "\n".join([f"  - {schema}: {score:.3f}" for schema, score in similarities[:5]])
            system_content = f"""Found {len(schemas)} relevant schemas for utterance: '{cleaned_utterance}'
{'(Original: "' + original_utterance + '")' if cleaned_utterance != original_utterance else ''}

Top 5 schema similarities:
{similarity_info}

Total schemas analyzed: {len(similarities)}"""
            
            system_msg = SystemMessage(content=system_content)
            state["messages"].append(system_msg)
            state["schemas"] = schemas
            state["schema_map"] = schema_map
            state["schema_similarities"] = similarities
            
        except Exception as e:
            state["error"] = f"Error finding schemas: {str(e)}"
            
        return state

    async def _select_schemas_node(self, state: AgentState) -> AgentState:
        """Node to select source and target schemas using LLM"""
        try:
            utterance = state["utterance"]  # Use cleaned utterance
            schemas = state["schemas"]
            similarities = state.get("schema_similarities", [])
            
            # Use the LLM wrapper for schema selection with detailed reasoning
            source_schema, target_schema, reasoning = self._select_schemas_with_llm(utterance, schemas, similarities)
            
            state["source_schema"] = source_schema
            state["target_schema"] = target_schema
            state["source_schema_content"] = state["schema_map"].get(source_schema, "")
            state["target_schema_content"] = state["schema_map"].get(target_schema, "")
            state["schema_selection_reasoning"] = reasoning
            
            # Add detailed AI message about schema selection with reasoning
            ai_content = f"""Schema Selection Complete:

**Selected Schemas:**
- Source Schema: `{source_schema}`
- Target Schema: `{target_schema}`

**Selection Reasoning:**
{reasoning}

**Available Schemas Considered:**
{', '.join(schemas[:10])}{'...' if len(schemas) > 10 else ''}

**Similarity Scores for Selected Schemas:**"""
            
            # Add similarity scores for selected schemas
            for schema, score in similarities:
                if schema in [source_schema, target_schema]:
                    ai_content += f"\n- {schema}: {score:.3f}"
            
            ai_msg = AIMessage(content=ai_content)
            state["messages"].append(ai_msg)
            
        except Exception as e:
            state["error"] = f"Error selecting schemas: {str(e)}"
            error_msg = AIMessage(content=f"Error during schema selection: {str(e)}")
            state["messages"].append(error_msg)
            
        return state

    async def _generate_mappings_node(self, state: AgentState) -> AgentState:
        """Node to generate mappings between schemas"""
        try:
            source_schema = state["source_schema"]
            target_schema = state["target_schema"]
            source_content = state["source_schema_content"]
            target_content = state["target_schema_content"]
            
            if source_content and target_content:
                mapping_result = find_mappings(
                    source_content, source_schema,
                    target_content, target_schema
                )
                state["mapping_result"] = mapping_result
                
                # Add detailed AI message about mapping completion
                ai_content = f"""Mapping Generation Complete

**Schemas Mapped:**
- From: `{source_schema}` 
- To: `{target_schema}`

**Mapping Summary:**
- Mapping Name: {mapping_result.get('name', 'N/A')}
- Properties Mapped: {len(mapping_result.get('properties', {}))}
- Non-empty Mappings: {len([k for k, v in mapping_result.get('properties', {}).items() if v])}

**Generated Mapping Structure:**
```json
{json.dumps(mapping_result, indent=2)[:500]}{'...' if len(json.dumps(mapping_result)) > 500 else ''}
```"""
                
                ai_msg = AIMessage(content=ai_content)
                state["messages"].append(ai_msg)
            else:
                error_content = "Missing schema content for mapping generation"
                state["error"] = error_content
                error_msg = AIMessage(content=error_content)
                state["messages"].append(error_msg)
                
        except Exception as e:
            error_content = f"Error generating mappings: {str(e)}"
            state["error"] = error_content
            error_msg = AIMessage(content=error_content)
            state["messages"].append(error_msg)
            
        return state

    def _find_relevant_schema(self, user_question: str, recall_limit=10) -> tuple[list, dict, list]:
        """Find relevant schemas using embeddings and return similarities"""
        if not self.schema_map:
            return [], {}, []
        
        # Prioritize 'gateway' keys
        gateway_keys = [k for k in self.schema_map.keys() if 'gateway' in k.lower()]
        
        user_embedding = self.sentence_transformer.encode([user_question], show_progress_bar=False)[0]
        key_scores = []

        for k in gateway_keys:
            doc = k.split("/")[1].replace("_", " ").replace(".graphql", "")
            doc = split_camel_case(doc)
            
            key_embedding = self.sentence_transformer.encode([doc], show_progress_bar=False)[0]
            cosine_sim = np.dot(user_embedding, key_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(key_embedding)
            )
            schema_name = k.split("/")[1].replace(".graphql", "")
            key_scores.append((schema_name, cosine_sim))

        key_scores_sorted = sorted(key_scores, key=lambda x: x[1], reverse=True)
        best_keys_with_scores = key_scores_sorted[:recall_limit]
        best_keys = [x[0] for x in best_keys_with_scores]
        
        best_schema_map = {}
        for k in gateway_keys:
            schema_name = k.split("/")[1].replace(".graphql", "")
            if schema_name in best_keys:
                best_schema_map[schema_name] = self.schema_map[k]
        
        return best_keys, best_schema_map, key_scores_sorted

    def _select_schemas_with_llm(self, utterance: str, schemas: List[str], similarities: List[tuple]) -> tuple[str, str, str]:
        """Select schemas using the LLM wrapper with detailed reasoning"""
        
        # Prepare schema information with similarities
        schema_info = []
        similarity_dict = {schema: score for schema, score in similarities}
        
        for schema in schemas[:10]:  # Limit to top 10 for context
            score = similarity_dict.get(schema, 0.0)
            schema_info.append(f"- {schema} (similarity: {score:.3f})")
        
        schema_selection_prompt = f"""
        Given a user utterance and a list of schemas for mapping, choose a source and a target schema.
        
        User utterance: "{utterance}"
        
        Available schemas with similarity scores:
        {chr(10).join(schema_info)}
        
        Please analyze the utterance and select:
        1. A source schema (what you're mapping FROM)
        2. A target schema (what you're mapping TO)
        
        Consider:
        - Keywords in the utterance that match schema names
        - Semantic similarity scores provided
        - Logical mapping direction based on the request
        
        Your response should be a JSON with the following fields:
        1. "source_schema": the name of the source schema
        2. "target_schema": the name of the target schema  
        3. "reasoning": detailed explanation of why these schemas were chosen, including:
           - How the utterance keywords relate to chosen schemas
           - Why this mapping direction makes sense
           - Similarity scores that influenced the decision
           - Any domain knowledge applied
        """
        
        messages = [
            {"role": "system", "content": "You are a schema selection assistant. Analyze the user request carefully and provide detailed reasoning for your schema selection choices. Respond only with valid JSON."},
            {"role": "user", "content": schema_selection_prompt}
        ]
        
        response = call_llm(messages)
        
        try:
            # Clean the response
            response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(response)
            return data["source_schema"], data["target_schema"], data.get("reasoning", "No reasoning provided")
        except Exception as e:
            # Fallback with basic reasoning
            if len(schemas) >= 2:
                fallback_reasoning = f"""Fallback selection due to JSON parsing error: {str(e)}
                
Selected based on highest similarity scores:
- Source: {schemas[0]} (score: {similarity_dict.get(schemas[0], 0.0):.3f})
- Target: {schemas[1]} (score: {similarity_dict.get(schemas[1], 0.0):.3f})

This is an automatic fallback when LLM response cannot be parsed."""
                return schemas[0], schemas[1], fallback_reasoning
            elif schemas:
                return schemas[0], schemas[0], "Only one schema available - using as both source and target"
            else:
                return "", "", "No schemas available for selection"

    async def run_agent(self, utterance: str) -> Dict[str, Any]:
        """Run the agent with a user utterance"""
        if not self.graph:
            raise RuntimeError("Agent not initialized. Call async_init() first.")
            
        initial_state = {
            "messages": [HumanMessage(content=utterance)],
            "utterance": utterance,
            "original_utterance": utterance,
            "schemas": [],
            "schema_map": {},
            "schema_similarities": None,
            "source_schema": None,
            "target_schema": None,
            "source_schema_content": None,
            "target_schema_content": None,
            "schema_selection_reasoning": None,
            "mapping_result": None,
            "store_where": None,
            "storage_results": None,
            "error": None
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "mapping_result": final_state.get("mapping_result"),
            "source_schema": final_state.get("source_schema"),
            "target_schema": final_state.get("target_schema"),
            "schema_selection_reasoning": final_state.get("schema_selection_reasoning"),
            "store_where": final_state.get("store_where"),
            "storage_results": final_state.get("storage_results"),
            "error": final_state.get("error"),
            "messages": final_state.get("messages", []),
            "schema_similarities": final_state.get("schema_similarities", [])
        }

    def print_conversation(self, result: Dict[str, Any]):
        """Print the conversation in a readable format"""
        print("\n" + "="*80)
        print("CONVERSATION FLOW")
        print("="*80)
        
        for i, message in enumerate(result.get("messages", [])):
            if isinstance(message, HumanMessage):
                print(f"\n🧑 USER MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
            elif isinstance(message, SystemMessage):
                print(f"\n🔧 SYSTEM MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
            elif isinstance(message, AIMessage):
                print(f"\n🤖 AI MESSAGE {i+1}:")
                print("-" * 40)
                print(message.content)
        
        print("\n" + "="*80)
        
        # Print storage results if available
        storage_results = result.get("storage_results")
        if storage_results:
            print("\nSTORAGE RESULTS:")
            print("-" * 40)
            for storage_type, result_data in storage_results.items():
                print(f"{storage_type.upper()}: {result_data}")

# Tool wrapper for LangChain integration
class GraphQLMappingTool(BaseModel):
    """Tool for GraphQL schema mapping"""
    utterance: str = Field(description="The user's natural language query for schema mapping")

def create_graphql_mapping_tool(agent: GraphQLSchemaMappingAgent):
    """Create a LangChain tool from the agent"""
    async def mapping_function(utterance: str) -> str:
        result = await agent.run_agent(utterance)
        if result["error"]:
            return f"Error: {result['error']}"
        
        # Include reasoning and storage results in the tool output
        output = {
            "mapping_result": result["mapping_result"],
            "source_schema": result["source_schema"],
            "target_schema": result["target_schema"],
            "selection_reasoning": result["schema_selection_reasoning"],
            "storage_location": result["store_where"],
            "storage_results": result["storage_results"]
        }
        return json.dumps(output, indent=2)
    
    return StructuredTool.from_function(
        func=mapping_function,
        name="graphql_mapping",
        description="Map between GraphQL schemas based on natural language description and optionally store results in GraphDB or Wiki",
        args_schema=GraphQLMappingTool
    )


### UNCOMMENT IF YOU WANT TO TEST INDIVIDUALLY
import asyncio

async def main():
    agent = GraphQLSchemaMappingAgent()
    try:
        await agent.async_init()

        # Test query
        test_query = "Map resilience applications to apm applications and store into wiki"
        print(f"Testing with: {test_query}")

        result = await agent.run_agent(test_query)
        agent.print_conversation(result)

        print("\nFinal Result:")
        print(json.dumps(result["mapping_result"], indent=2) if result["mapping_result"] else "No mapping result")
    finally:
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
