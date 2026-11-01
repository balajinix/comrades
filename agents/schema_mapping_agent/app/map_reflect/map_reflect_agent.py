"""
map_reflect_agent.py - LangGraph-based Map-Reflect Pipeline

DEPRECATED: This module has been superseded by map_reflect_runner.py.
Please use `python -m map_reflect.map_reflect_runner` instead.

The runner provides:
- Batch inference (10x speedup)
- All reflection modes (self, error, UQ, neighborhood)
- Unified CLI interface

Example:
    python -m map_reflect.map_reflect_runner \\
        --datasets ITOpsGraph \\
        --models provider/meta-llama/llama-3-3-70b-instruct \\
        --property-lookup-file data/itopsgraph_api_lookup.json \\
        --data-dir data \\
        --use-uncertainty \\
        --use-uq-reflection
"""
import warnings
warnings.warn(
    "map_reflect_agent.py is deprecated. Use map_reflect_runner.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import networkx as nx

# Workaround for old sqlite3 on some systems
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, TypedDict, Optional
import re
from dataclasses import dataclass
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
from pathlib import Path
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Try to import BM25 for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# Always import litellm (needed for Mistral models which work better with chat API)
from litellm import completion

# Try native SDK for Granite/Llama models
try:
    from provider_sdk import Credentials
    from provider_sdk.metanames import GenTextParamsMetaNames as GenParams
    from provider_sdk.foundation_models.utils.enums import DecodingMethods
    from provider_sdk.foundation_models import ModelInference
    USE_LITELLM = False
except ImportError:
    USE_LITELLM = True

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------- BM25 Hybrid Retrieval ----------

def tokenize_field_name(field_name: str) -> List[str]:
    """
    Tokenize a field name by splitting on _, ., and camelCase boundaries.
    Example: infra_Kind.KIND_PACKER_SBOM -> ["infra", "kind", "kind", "packer", "sbom"]
    """
    # First split on . and _
    parts = re.split(r'[._]', field_name)

    tokens = []
    for part in parts:
        # Split camelCase: insert space before uppercase letters that follow lowercase
        camel_split = re.sub(r'([a-z])([A-Z])', r'\1 \2', part)
        # Also split on consecutive uppercase followed by lowercase (e.g., HTTPServer -> HTTP Server)
        camel_split = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', camel_split)
        # Split and lowercase
        for token in camel_split.split():
            token_lower = token.lower().strip()
            if token_lower:
                tokens.append(token_lower)

    return tokens


class BM25Retriever:
    """BM25 retriever for lexical matching of field names."""

    def __init__(self, collection_name: str, index_dir: str):
        self.collection_name = collection_name
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, f"bm25_index_{collection_name}.pkl")
        self.bm25 = None
        self.doc_ids = []
        self.doc_texts = []
        self.tokenized_corpus = []

    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents."""
        self.doc_ids = []
        self.doc_texts = []
        self.tokenized_corpus = []

        for doc in documents:
            doc_id = doc.get("id", doc.get("doc_id", ""))
            doc_text = doc.get("doc", doc.get("text", doc_id))

            self.doc_ids.append(doc_id)
            self.doc_texts.append(doc_text)
            # Tokenize using field name tokenizer
            tokens = tokenize_field_name(doc_id) + tokenize_field_name(doc_text)
            self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
        logger.info(f"Built BM25 index with {len(self.doc_ids)} documents")

    def _save_index(self):
        """Persist BM25 index to disk."""
        data = {
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "tokenized_corpus": self.tokenized_corpus
        }
        os.makedirs(self.index_dir, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Saved BM25 index to {self.index_path}")

    def load_index(self) -> bool:
        """Load BM25 index from disk. Returns True if loaded successfully."""
        if not os.path.exists(self.index_path):
            return False

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            self.doc_ids = data["doc_ids"]
            self.doc_texts = data["doc_texts"]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.debug(f"Loaded BM25 index from {self.index_path} with {len(self.doc_ids)} documents")
            return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """Search for document IDs matching the query."""
        if self.bm25 is None:
            return []

        # Tokenize query the same way as documents
        query_tokens = tokenize_field_name(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                results.append(self.doc_ids[idx])

        return results


def reciprocal_rank_fusion(semantic_ids: List[str], bm25_ids: List[str], k: int = 60) -> List[str]:
    """
    Merge results using Reciprocal Rank Fusion.
    score(doc) = 1/(k + rank_semantic) + 1/(k + rank_bm25)
    Returns list of doc IDs sorted by fused score.
    """
    scores = {}

    # Add semantic scores
    for rank, doc_id in enumerate(semantic_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    # Add BM25 scores
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

    # Sort by fused score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs]


# Global BM25 retriever cache
_bm25_retrievers: Dict[str, BM25Retriever] = {}


# ---------- config loading ----------

def load_dataset_config(dataset: str = None, config_path: str = None) -> Dict[str, Any]:
    """Load dataset-specific configuration from config.yaml.

    Args:
        dataset: Dataset name (e.g., 'OMOP_MIMIC', 'ITOps')
        config_path: Path to config file (default: config.yaml in same directory)

    Returns:
        Configuration dict with hyperparameters
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    default_config = {
        "top_k": 15,
        "neighbor_k": 5,
        "n_results_multiplier": 10,
        "strict_schema_filter": False,
        "product_boost": 100,
        "product_penalty": -50,
        "type_exact_boost": 20,
        "type_partial_boost": 10,
        "type_field_boost": 5,
    }

    try:
        with open(config_path, 'r') as f:
            all_config = yaml.safe_load(f)

        # Get dataset-specific config or default
        if dataset and dataset in all_config:
            config = all_config[dataset]
            logger.info(f"Loaded config for dataset '{dataset}'")
        elif 'default' in all_config:
            config = all_config['default']
            logger.info(f"Using default config (dataset '{dataset}' not found)")
        else:
            config = default_config
            logger.warning(f"No config found, using hardcoded defaults")

        # Merge with defaults to ensure all keys exist
        return {**default_config, **config}

    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        return default_config

# Stable logger namespace
logger = logging.getLogger("gateway_ai_api.map_reflect")

# ---------- helpers ----------

def _strip_code_fences(s: str) -> str:
    """Remove Markdown code fences from model output before json.loads."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.replace("```json", "").replace("```", "").strip()

def _safe_json_loads(payload: str):
    """Try json.loads; return (obj_or_none, error_str_or_none)."""
    try:
        return json.loads(payload), None
    except Exception as e:
        return None, str(e)

# ---------- types ----------

class MappingState(TypedDict):
    user_utterance: str
    property_data: Dict[str, Any]
    source_properties: List[str]
    destination_properties: List[str]
    candidate_mappings: Dict[str, List[Dict[str, str]]]
    final_mappings: Dict[str, List[Dict[str, str]]]
    reflection_feedback: str
    graph: nx.Graph
    chroma_collection: Any
    source_context: str
    destination_context: str
    # New fields for improved schema selection
    source_product: str
    target_product: str
    source_type_hints: List[str]
    target_type_hints: List[str]
    # Field hints extracted from utterance for better field-level mapping
    source_field_hints: List[str]
    target_field_hints: List[str]
    # Retrieval mode flag (True when target must be found via retrieval)
    is_retrieval_mode: bool


@dataclass
class PropertyMapping:
    source_property: str
    destination_property: str
    justification: str
    confidence_score: float = 0.0

# ---------- agent ----------

class MapReflectAgent:
    def __init__(
        self,
        property_lookup_file: str,
        rebuild_index: bool = False,
        model: str = None,
        dataset: Optional[str] = None,
        chroma_dir: str = "../data/chroma_db",
        collection_name: Optional[str] = None,
        top_k: int = None,
        neighbor_k: int = None,
    ):
        """
        Initialize MapReflect agent.

        Args:
            property_lookup_file: Path to property lookup JSON file (required)
            rebuild_index: Whether to rebuild the ChromaDB index
            model: LLM model to use (if None, uses default from environment or fallback)
            dataset: Dataset name for namespaced collection (optional)
            chroma_dir: Directory for ChromaDB storage
            collection_name: Override collection name (optional)
            top_k: Number of top results to retrieve (overrides config if provided)
            neighbor_k: Number of neighbors to expand (overrides config if provided)
        """
        self.model = model
        self.dataset = dataset
        self.rebuild_index = rebuild_index

        # Load dataset-specific config
        self.config = load_dataset_config(dataset)
        logger.info("Loaded config for dataset=%s: %s", dataset, self.config)

        # Use provided values or fall back to config
        self.top_k = max(1, int(top_k)) if top_k is not None else self.config["top_k"]
        self.neighbor_k = max(0, int(neighbor_k)) if neighbor_k is not None else self.config["neighbor_k"]
        self.n_results_multiplier = self.config["n_results_multiplier"]
        self.strict_schema_filter = self.config["strict_schema_filter"]

        # Chroma client & embeddings
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load properties
        with open(property_lookup_file, "r") as f:
            self.property_data = json.load(f)

        # Namespace collection per dataset unless overridden
        self.collection_name = (
            collection_name
            if collection_name
            else (f"property_index_{self.dataset.lower()}" if self.dataset else "property_index")
        )

        logger.info(
            "MapReflect init | dataset=%s model=%s chroma_dir=%s collection=%s props=%d",
            self.dataset or "default", self.model, chroma_dir, self.collection_name, len(self.property_data)
        )

        # Pre-build property graph once (O(n²) operation - caching saves ~10s per sample)
        self._cached_property_graph = self._build_property_graph_once()
        logger.info("Pre-built property graph | nodes=%d edges=%d",
                    self._cached_property_graph.number_of_nodes(),
                    self._cached_property_graph.number_of_edges())

        # Build workflow
        self.graph = self._build_graph()

    # ---- LLM wrapper ----
    def _format_prompt_for_model(self, messages, model_id):
        """Format messages based on model-specific chat template.

        Args:
            messages: List of dicts with 'role' and 'content' keys
            model_id: Model identifier string

        Returns:
            Tuple of (formatted_prompt, stop_sequences)
        """
        model_lower = model_id.lower()

        # Granite 4 models (granite-4-h-small, etc.)
        if "granite-4" in model_lower or "granite4" in model_lower:
            prompt_parts = []
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
            prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
            return "".join(prompt_parts), ["<|end_of_text|>"]

        # Mistral models
        elif "mistral" in model_lower:
            system_msg = ""
            user_content = []
            for m in messages:
                if m.get('role') == 'system':
                    system_msg = m.get('content', '')
                elif m.get('role') == 'user':
                    user_content.append(m.get('content', ''))

            combined_user = "\n\n".join(user_content)
            if system_msg:
                prompt = f"[INST] {system_msg}\n\n{combined_user} [/INST]"
            else:
                prompt = f"[INST] {combined_user} [/INST]"
            return prompt, ["[/INST]", "</s>"]

        # Llama 3 models
        elif "llama" in model_lower:
            prompt_parts = ["<|begin_of_text|>"]
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                prompt_parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(prompt_parts), ["<|eot_id|>", "<|end_of_text|>"]

        # Granite 3.x models (granite-3-2-8b, granite-3-3-8b, etc.)
        elif "granite-3" in model_lower or "granite3" in model_lower:
            prompt_parts = []
            for m in messages:
                role = m.get('role', 'user')
                content = m.get('content', '')
                prompt_parts.append(f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n")
            prompt_parts.append("<|start_of_role|>assistant<|end_of_role|>")
            return "".join(prompt_parts), ["<|end_of_text|>"]

        # Default fallback for unknown models
        else:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(content)
            return "\n\n".join(prompt_parts), ["<|endoftext|>"]

    def call_llm(self, messages):
        """Call LLM using native Provider SDK or litellm fallback."""
        import time
        try:
            t0 = time.time()

            # Get credentials
            api_key = os.getenv("PROVIDER_API_KEY") or os.getenv("PROVIDER_APIKEY")
            project_id = os.getenv("PROVIDER_PROJECT_ID")
            url = os.getenv("PROVIDER_URL", "https://provider-host")

            if not api_key or not project_id:
                raise RuntimeError("PROVIDER_API_KEY and PROVIDER_PROJECT_ID required")

            # Remove provider/ prefix if present
            model_id = self.model.replace("provider/", "")

            # Use litellm for Mistral models (works better with chat API) or if native SDK not available
            use_litellm_for_model = USE_LITELLM or 'mistral' in model_id.lower()

            if use_litellm_for_model:
                t1 = time.time()
                logger.warning(f"LLM_TIMING | prompt_format: {t1-t0:.2f}s (litellm mode)")

                # Convert to litellm model format
                # Check for Mistral API key to decide routing
                mistral_api_key = os.getenv("MISTRAL_API_KEY")

                if 'mistral' in model_id.lower():
                    if mistral_api_key:
                        # Use direct Mistral API
                        if '/' in model_id:
                            model_name = model_id.split('/', 1)[1]  # e.g., mistral-medium-2505
                        else:
                            model_name = model_id
                        litellm_model = f"mistral/{model_name}"
                        logger.warning(f"LLM: Using Mistral API directly for {litellm_model}")
                    else:
                        # Route through Provider (no MISTRAL_API_KEY)
                        litellm_model = f"provider/{model_id}"
                        logger.warning(f"LLM: Using Provider for Mistral model {litellm_model}")
                else:
                    litellm_model = f"provider/{model_id}"

                response = completion(
                    model=litellm_model,
                    messages=messages,
                    temperature=0,  # Use 0 for deterministic responses
                    max_tokens=8000,
                )
                t2 = time.time()
                logger.warning(f"LLM_TIMING | generate: {t2-t1:.2f}s | total: {t2-t0:.2f}s")

                return response.choices[0].message.content

            # Format prompt based on model type
            prompt, stop_sequences = self._format_prompt_for_model(messages, model_id)

            t1 = time.time()
            logger.warning(f"LLM_TIMING | prompt_format: {t1-t0:.2f}s")

            # Call Provider - reuse cached model if available
            if not hasattr(self, '_cached_model') or self._cached_model_id != model_id:
                parameters = {
                    GenParams.MAX_NEW_TOKENS: 8000,
                    GenParams.TEMPERATURE: 0.1,
                    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                    GenParams.STOP_SEQUENCES: stop_sequences,
                }

                credentials = Credentials(url=url, api_key=api_key)
                self._cached_model = ModelInference(
                    model_id=model_id,
                    params=parameters,
                    credentials=credentials,
                    project_id=project_id
                )
                self._cached_model_id = model_id
                t2 = time.time()
                logger.warning(f"LLM_TIMING | model_init: {t2-t1:.2f}s (new)")
            else:
                t2 = time.time()
                logger.warning(f"LLM_TIMING | model_init: {t2-t1:.2f}s (cached)")

            response = self._cached_model.generate_text(prompt=prompt)
            t3 = time.time()
            logger.warning(f"LLM_TIMING | generate: {t3-t2:.2f}s | total: {t3-t0:.2f}s")

            return response[0] if isinstance(response, list) else response

        except Exception as e:
            error_str = str(e).lower()
            critical_patterns = ["authentication", "unauthorized", "api key", "project", "exceeded limit"]

            if any(p in error_str for p in critical_patterns):
                logger.error(f"CRITICAL LLM error - stopping execution: {e}")
                raise RuntimeError(f"Critical LLM error: {e}") from e

            logger.error(f"LLM error: {e}", exc_info=True)
            return f"Error calling LLM: {e}"


    # ---- LangGraph DAG ----
    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(MappingState)
        if self.rebuild_index:
            workflow.add_node("initialize_index", self._initialize_index)
        workflow.add_node("extract_contexts", self._extract_contexts)
        workflow.add_node("create_graph", self._create_property_graph)
        workflow.add_node("select_source_properties", self._select_source_properties)
        workflow.add_node("select_destination_properties", self._select_destination_properties)
        workflow.add_node("find_mappings", self._find_mappings)
        workflow.add_node("reflect_and_prune", self._reflect_and_prune)
        workflow.add_node("format_output", self._format_output)

        if self.rebuild_index:
            workflow.add_edge("initialize_index", "extract_contexts")
            workflow.set_entry_point("initialize_index")
        else:
            workflow.set_entry_point("extract_contexts")

        workflow.add_edge("extract_contexts", "create_graph")
        workflow.add_edge("create_graph", "select_source_properties")
        workflow.add_edge("select_source_properties", "select_destination_properties")
        workflow.add_edge("select_destination_properties", "find_mappings")
        # the following is taking too much time.
        #workflow.add_edge("find_mappings", "reflect_and_prune")
        #workflow.add_edge("reflect_and_prune", "format_output")
        workflow.add_edge("find_mappings", "format_output")
        workflow.add_edge("format_output", END)
        return workflow.compile()

    # ---- Nodes ----
    def _initialize_index(self, state: MappingState) -> MappingState:
        state["chroma_collection"] = self._create_collection_from_state(state)
        return state

    def _extract_contexts(self, state: MappingState) -> MappingState:
        user_utterance = state["user_utterance"]
        logger.info("Extracting contexts from utterance")

        # First, extract schema hints directly from utterance using pattern matching
        source_hints, target_hints = self._parse_utterance_for_schemas(user_utterance)

        # Check if this is a retrieval-style query (no target specified)
        is_retrieval_query = self._is_retrieval_query(user_utterance)

        if is_retrieval_query:
            # Retrieval mode: extract only source, target will be found via semantic search
            prompt = f"""Analyze this schema mapping request: "{user_utterance}"

            This is a RETRIEVAL query - the user wants to find target schemas that can be mapped to a source schema.
            Extract ONLY the source schema information. The target will be found via retrieval.

            Patterns:
            - "find types that can be mapped to [product] [type]" -> extract source product and type
            - "find APIs that can be mapped with [product] [api]" -> extract source product and api
            - "find tables that can be joined with [table]" -> extract source table
            - "find OMOP tables that map to [source]" -> extract source, target_product is "omop"

            Return ONLY a JSON object:
            {{"source_context": "[product] [specific type name]", "source_product": "[product name]", "source_types": ["type1", "type2"], "is_retrieval": true}}

            Your entire response must be ONLY the JSON object starting with {{ and ending with }}.
            """.strip()
        else:
            # Oracle mode: both source and target specified in utterance
            prompt = f"""Analyze this schema mapping request: "{user_utterance}"

            Extract the specific source and destination schema/type information.

            The request follows the pattern: "map [source_product] [source_type] with [target_product] [target_type]"

            Examples:
            - "map resilience api key with optimizer entity" -> source: "resilience ApiKey", dest: "optimizer Entity"
            - "map apm applications with resilience tokens" -> source: "apm Application", dest: "resilience Token"

            Return ONLY a JSON object:
            {{"source_context": "[product] [specific type name]", "destination_context": "[product] [specific type name]", "source_product": "[product name]", "target_product": "[product name]", "source_types": ["type1", "type2"], "target_types": ["type1", "type2"]}}

            Your entire response must be ONLY the JSON object starting with {{ and ending with }}.
            """.strip()

        messages = [
            {"role": "system", "content": "You are an expert in API schema mapping. Extract specific schema type names, not general domains."},
            {"role": "user", "content": prompt},
        ]
        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)

        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            ctx, err = _safe_json_loads(match.group())
        else:
            ctx, err = _safe_json_loads(raw)

        if err or not isinstance(ctx, dict):
            logger.warning("Context JSON parse failed; using parsed hints. err=%s raw=%s", err, raw[:200])
            state["source_context"] = " ".join(source_hints) if source_hints else user_utterance
            state["destination_context"] = " ".join(target_hints) if target_hints else user_utterance
            state["source_product"] = source_hints[0] if source_hints else ""
            state["target_product"] = target_hints[0] if target_hints else ""
            state["source_type_hints"] = source_hints[1:] if len(source_hints) > 1 else []
            state["target_type_hints"] = target_hints[1:] if len(target_hints) > 1 else []
            state["is_retrieval_mode"] = is_retrieval_query
        else:
            state["source_context"] = ctx.get("source_context", user_utterance)
            state["source_product"] = ctx.get("source_product", source_hints[0] if source_hints else "")
            state["source_type_hints"] = ctx.get("source_types", source_hints[1:] if len(source_hints) > 1 else [])
            state["is_retrieval_mode"] = ctx.get("is_retrieval", is_retrieval_query)

            if state["is_retrieval_mode"]:
                # In retrieval mode, destination context is derived from source for semantic search
                state["destination_context"] = state["source_context"]
                state["target_product"] = ""  # Will be found via retrieval
                state["target_type_hints"] = []
            else:
                state["destination_context"] = ctx.get("destination_context", user_utterance)
                state["target_product"] = ctx.get("target_product", target_hints[0] if target_hints else "")
                state["target_type_hints"] = ctx.get("target_types", target_hints[1:] if len(target_hints) > 1 else [])

        logger.info("Contexts | source=%r dest=%r src_product=%r tgt_product=%r",
                    state["source_context"], state["destination_context"],
                    state.get("source_product"), state.get("target_product"))
        return state

    def _extract_field_hints(self, state: MappingState) -> MappingState:
        """Extract specific field/attribute hints from utterance for better field-level mapping.

        This method analyzes the user's utterance to identify specific field names or
        attributes they want to map, improving field-level accuracy beyond just schema matching.
        """
        user_utterance = state["user_utterance"]
        logger.info("Extracting field hints from utterance")

        prompt = f"""Analyze this schema mapping request: "{user_utterance}"

Extract specific field or attribute names that the user wants to map between schemas.
Focus on nouns and field-like terms that indicate what data should be mapped.

Examples:
- "map resilience application name with optimizer entity" → source: ["name", "application"], target: ["entity"]
- "map apm timestamp to resilience created_on" → source: ["timestamp"], target: ["created_on"]
- "map resilience tags to optimizer entity tags" → source: ["tags"], target: ["tags"]
- "map resilience vulnerabilities severity to optimizer entity" → source: ["severity", "vulnerabilities"], target: ["entity"]
- "map resilience api key with apm call group" → source: ["api", "key", "apikey"], target: ["call", "group", "callgroup"]

Return ONLY a JSON object with field hints (lowercase, include both singular and compound forms):
{{"source_field_hints": ["field1", "field2"], "target_field_hints": ["field1", "field2"]}}

Your entire response must be ONLY the JSON object starting with {{ and ending with }}.
""".strip()

        messages = [
            {"role": "system", "content": "You are an expert in schema mapping. Extract field-level hints from natural language requests."},
            {"role": "user", "content": prompt},
        ]
        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)

        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            hints, err = _safe_json_loads(match.group())
        else:
            hints, err = _safe_json_loads(raw)

        if err or not isinstance(hints, dict):
            logger.warning("Field hints JSON parse failed; extracting from utterance directly. err=%s", err)
            # Fallback: extract words from utterance that look like field names
            words = re.findall(r'\b[a-z_]+\b', user_utterance.lower())
            # Filter out common words
            stop_words = {'map', 'with', 'to', 'from', 'the', 'and', 'or', 'a', 'an', 'in', 'on', 'for'}
            field_words = [w for w in words if w not in stop_words and len(w) > 2]
            state["source_field_hints"] = field_words
            state["target_field_hints"] = field_words
        else:
            state["source_field_hints"] = [h.lower() for h in hints.get("source_field_hints", [])]
            state["target_field_hints"] = [h.lower() for h in hints.get("target_field_hints", [])]

        logger.info("Field hints | source=%r target=%r",
                    state["source_field_hints"], state["target_field_hints"])
        return state

    def _parse_utterance_for_schemas(self, utterance: str) -> tuple:
        """Parse utterance to extract source and target schema hints.

        Handles patterns like:
        - "map resilience api key with optimizer entity"
        - "map apm applications with resilience tokens"

        Returns (source_hints, target_hints) where each is [product, type1, type2, ...]
        """
        utterance_lower = utterance.lower().strip()

        # Known product names in ITOps dataset
        products = ['resilience', 'optimizer', 'apm']

        # Patterns: "map X ... with Y ..." or "map X ... to Y ..."
        patterns = [
            r"map\s+(\w+)\s+(.+?)\s+with\s+(\w+)\s+(.+?)$",
            r"map\s+(\w+)\s+(.+?)\s+to\s+(\w+)\s+(.+?)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, utterance_lower)
            if match:
                src_product, src_rest, tgt_product, tgt_rest = match.groups()

                # Extract type hints from the rest
                src_types = self._extract_type_hints(src_rest)
                tgt_types = self._extract_type_hints(tgt_rest)

                return ([src_product] + src_types, [tgt_product] + tgt_types)

        # Fallback: find product names and split
        found_products = []
        for prod in products:
            if prod in utterance_lower:
                found_products.append(prod)

        if len(found_products) >= 2:
            # Assume first is source, second is target
            src_idx = utterance_lower.find(found_products[0])
            tgt_idx = utterance_lower.find(found_products[1])

            if src_idx < tgt_idx:
                src_part = utterance_lower[src_idx:tgt_idx]
                tgt_part = utterance_lower[tgt_idx:]
            else:
                src_part = utterance_lower[tgt_idx:src_idx]
                tgt_part = utterance_lower[src_idx:]
                found_products = [found_products[1], found_products[0]]

            src_types = self._extract_type_hints(src_part.replace(found_products[0], ''))
            tgt_types = self._extract_type_hints(tgt_part.replace(found_products[1], ''))

            return ([found_products[0]] + src_types, [found_products[1]] + tgt_types)

        return ([], [])

    def _is_retrieval_query(self, utterance: str) -> bool:
        """Check if the utterance is a retrieval-style query (no target specified).

        Retrieval queries ask to FIND targets, rather than specifying both source and target.

        Returns True for patterns like:
        - "find types that can be mapped to X"
        - "find APIs that can be mapped with X"
        - "find tables that can be joined with X"
        - "find OMOP tables that map to X"
        """
        utterance_lower = utterance.lower().strip()

        # Retrieval patterns - these don't specify a target
        retrieval_patterns = [
            r"^find\s+(types|apis|tables|schemas)\s+that\s+(can\s+be\s+)?mapp",
            r"^find\s+(types|apis|tables|schemas)\s+that\s+(can\s+be\s+)?join",
            r"^find\s+\w+\s+tables\s+that\s+map",  # "find OMOP tables that map to X"
            r"^retrieve\s+(types|apis|tables)",
            r"^search\s+for\s+(types|apis|tables)",
        ]

        for pattern in retrieval_patterns:
            if re.search(pattern, utterance_lower):
                return True

        # Oracle patterns - these specify both source and target
        oracle_patterns = [
            r"^map\s+\w+.*\s+(to|with)\s+\w+",  # "map X to Y" or "map X with Y"
        ]

        for pattern in oracle_patterns:
            if re.search(pattern, utterance_lower):
                return False

        # Default: if no "map X to Y" pattern found, treat as retrieval
        if not re.search(r"\s+(to|with)\s+", utterance_lower):
            return True

        return False

    def _extract_type_hints(self, text: str) -> list:
        """Extract potential schema type names from text.

        Converts phrases like 'api key' -> 'ApiKey', 'entity actions' -> ['Entity', 'Action']
        """
        # Remove common words
        stop_words = {'map', 'to', 'from', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as', 'filtered', 'stats'}

        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 1]

        # Convert to CamelCase type names
        type_hints = []
        i = 0
        while i < len(words):
            # Check for two-word type names (e.g., "api key" -> "ApiKey")
            if i + 1 < len(words):
                combined = words[i].capitalize() + words[i+1].capitalize()
                # Common two-word patterns
                if words[i] in ['api', 'call', 'market', 'entity', 'application']:
                    type_hints.append(combined)
                    i += 2
                    continue

            type_hints.append(words[i].capitalize())
            i += 1

        return type_hints

    def _build_property_graph_once(self) -> nx.Graph:
        """Build property graph once during init (O(n²) - cached for reuse)."""
        import time
        t0 = time.time()
        G = nx.Graph()
        props = self.property_data

        for k, v in props.items():
            G.add_node(k, **v)

        edge_count = 0
        keys = list(props.keys())
        for i, p1 in enumerate(keys):
            d1 = props[p1]
            for p2 in keys[i + 1:]:
                d2 = props[p2]
                weight = 0.0
                if d1.get("product_name") == d2.get("product_name"):
                    weight += 0.5
                if d1.get("type_name") == d2.get("type_name"):
                    weight += 0.3
                w1 = set(str(d1.get("field_name", "")).lower().split("."))
                w2 = set(str(d2.get("field_name", "")).lower().split("."))
                sim = len(w1 & w2) / max(len(w1 | w2), 1)
                weight += 0.2 * sim
                if weight > 0.1:
                    G.add_edge(p1, p2, weight=weight)
                    edge_count += 1

        logger.info("Built property graph in %.2fs | nodes=%d edges=%d",
                    time.time() - t0, G.number_of_nodes(), edge_count)
        return G

    def _create_property_graph(self, state: MappingState) -> MappingState:
        """Use pre-cached property graph instead of rebuilding."""
        state["graph"] = self._cached_property_graph
        logger.info("Using cached property graph | nodes=%d edges=%d",
                    self._cached_property_graph.number_of_nodes(),
                    self._cached_property_graph.number_of_edges())
        return state

    def _get_or_create_collection(self, state: MappingState):
        try:
            col = self.chroma_client.get_collection(name=self.collection_name)
            logger.info("Using existing Chroma collection: %s", self.collection_name)
            return col
        except Exception:
            return self._create_collection_from_state(state)

    def _select_source_properties(self, state: MappingState) -> MappingState:
        collection = state["chroma_collection"] or self._get_or_create_collection(state)
        state["chroma_collection"] = collection

        source_product = state.get("source_product", "").lower()
        type_hints = state.get("source_type_hints", [])

        # For strict schema filtering (OMOP), filter by product FIRST before semantic search
        if self.strict_schema_filter and source_product:
            # Get all properties matching the product
            matching_ids = self._get_properties_by_product(state["property_data"], source_product, type_hints)
            logger.info("Strict filter: found %d properties matching product=%s", len(matching_ids), source_product)

            if matching_ids:
                # Now do semantic search and intersect with type-filtered properties
                q = state["source_context"]
                q_emb = self.embedding_model.encode(q).tolist()
                # Search more results to increase chance of finding type-matching ones
                n_results = min(len(state["property_data"]), self.top_k * self.n_results_multiplier * 5)
                res = collection.query(query_embeddings=[q_emb], n_results=n_results)
                semantic_ids = res["ids"][0] if res.get("ids") else []

                # Intersect semantic results with product-filtered results
                filtered_ids = [pid for pid in semantic_ids if pid in matching_ids][:self.top_k]
                logger.info("Strict filter: %d semantic results intersected to %d", len(semantic_ids), len(filtered_ids))

                # If intersection is empty/too small, take type-filtered properties directly
                if len(filtered_ids) < self.top_k:
                    remaining = self.top_k - len(filtered_ids)
                    extras = [pid for pid in sorted(matching_ids) if pid not in filtered_ids][:remaining]
                    filtered_ids.extend(extras)
                    logger.info("Strict filter: added %d type-matched properties to reach %d", len(extras), len(filtered_ids))
            else:
                # Fallback if no product match
                logger.warning("Strict filter: no properties match product=%s, falling back to semantic", source_product)
                filtered_ids = []
        else:
            filtered_ids = []

        # Standard semantic search path (soft filtering or fallback)
        if not filtered_ids:
            q = state["source_context"]
            q_emb = self.embedding_model.encode(q).tolist()

            # Get more results initially, then filter by product/type/field hints
            n_results = min(self.top_k * self.n_results_multiplier, len(state["property_data"]))
            res = collection.query(query_embeddings=[q_emb], n_results=n_results)
            all_ids = res["ids"][0] if res.get("ids") else []

            # Get field hints for utterance-aware property selection
            source_field_hints = state.get("source_field_hints", [])

            filtered_ids = self._filter_properties_by_hints(
                all_ids, state["property_data"], source_product, type_hints,
                field_hints=source_field_hints
            )

            # If filtering returned nothing, fall back to top results
            if not filtered_ids:
                filtered_ids = set(all_ids[:self.top_k])
                logger.warning("No properties matched source hints, using top %d unfiltered", self.top_k)
            else:
                # Limit to top_k after filtering
                filtered_ids = set(list(filtered_ids)[:self.top_k])

        filtered_ids = set(filtered_ids)

        # 1-hop neighbors (only if neighbor_k > 0)
        nbrs = set()
        if self.neighbor_k > 0:
            for pid in filtered_ids:
                if pid in state["graph"]:
                    nbrs.update(state["graph"].neighbors(pid))
            # When strict filtering is enabled, filter neighbors by product AND type
            if self.strict_schema_filter and source_product:
                product_lower = source_product.lower()
                type_hints_lower = [h.lower() for h in (type_hints or [])]
                filtered_nbrs = set()
                for n in nbrs:
                    if not n.lower().startswith(product_lower + "_"):
                        continue
                    # Also filter by type hints if available
                    if type_hints_lower:
                        pdata = state["property_data"].get(n, {})
                        tname = pdata.get("type_name", "").lower()
                        if any(hint in tname for hint in type_hints_lower):
                            filtered_nbrs.add(n)
                    else:
                        filtered_nbrs.add(n)
                nbrs = filtered_nbrs
            nbrs = set(sorted(nbrs)[: self.neighbor_k])

        # Keep filtered_ids first (by semantic order), then append neighbors
        combined = list(filtered_ids) + [n for n in sorted(nbrs) if n not in filtered_ids]
        state["source_properties"] = combined
        logger.info("Selected source properties | filtered=%d +nbr=%d -> %d (product=%s, hints=%s, strict=%s)",
                    len(filtered_ids), len(nbrs), len(combined), source_product, type_hints, self.strict_schema_filter)
        return state

    def _get_properties_by_product(self, property_data: dict, product: str, type_hints: list = None) -> set:
        """Get all properties matching a product name (and optionally type hints).

        Used for strict schema filtering where we filter by product BEFORE semantic search.
        """
        matching = set()
        type_matched = set()
        product_lower = product.lower()
        type_hints_lower = [h.lower() for h in (type_hints or [])]

        for pid, pdata in property_data.items():
            pname = pdata.get("product_name", "").lower()
            tname = pdata.get("type_name", "").lower()

            # Product must match
            if product_lower not in pname and pname not in product_lower:
                continue

            matching.add(pid)

            # Track which ones match type hints
            if type_hints_lower:
                type_match = any(
                    hint in tname or tname in hint
                    for hint in type_hints_lower
                )
                if type_match:
                    type_matched.add(pid)

        # If type hints were provided and found matches, prefer those
        # Otherwise fall back to all product-matching properties
        if type_hints_lower and type_matched:
            logger.info("Type hint filter: %d of %d properties match type hints %s",
                       len(type_matched), len(matching), type_hints_lower)
            return type_matched

        return matching

    def _filter_properties_by_hints(self, property_ids: list, property_data: dict,
                                     product: str, type_hints: list,
                                     field_hints: list = None) -> set:
        """Re-rank properties by product name, type hints, and field hints.

        Uses a soft re-ranking approach: properties matching the product get boosted,
        but we don't strictly filter out non-matching properties (since the semantic
        relationship between utterances and schemas may not be literal).

        Args:
            property_ids: List of property IDs from ChromaDB (in semantic similarity order)
            property_data: Full property metadata dict
            product: Product name to filter by (e.g., 'resilience')
            type_hints: List of type name hints (e.g., ['ApiKey', 'Application'])
            field_hints: List of field name hints extracted from utterance (e.g., ['name', 'tags'])

        Returns:
            Set of re-ranked property IDs
        """
        # Get boost values from config
        product_boost = self.config.get("product_boost", 100)
        product_penalty = self.config.get("product_penalty", -50)
        type_exact_boost = self.config.get("type_exact_boost", 20)
        type_partial_boost = self.config.get("type_partial_boost", 10)
        type_field_boost = self.config.get("type_field_boost", 5)
        # New: field hint boost (high priority for utterance-aware matching)
        field_hint_exact_boost = self.config.get("field_hint_exact_boost", 50)
        field_hint_partial_boost = self.config.get("field_hint_partial_boost", 25)

        field_hints = field_hints or []
        scored = []

        for i, pid in enumerate(property_ids):
            pdata = property_data.get(pid, {})
            pname = pdata.get("product_name", "").lower()
            tname = pdata.get("type_name", "").lower()
            actual_field = pdata.get("field_name", "").lower()
            field_name = pid.split('.')[0].lower() if '.' in pid else pid.lower()
            # Full property path for matching
            full_path = pid.lower()

            # Base score from semantic similarity (inverse of position)
            base_score = len(property_ids) - i

            boost = 0

            # Product match (highest priority) - MUST match for source/target selection
            if product:
                if product in pname:
                    boost += product_boost  # Strong boost for matching product
                else:
                    boost += product_penalty  # Penalty for wrong product

            # Type hint matching (softer - these are often inexact)
            for hint in type_hints:
                hint_lower = hint.lower()
                # Exact type name match
                if hint_lower == tname:
                    boost += type_exact_boost
                elif hint_lower in tname or tname in hint_lower:
                    boost += type_partial_boost
                # Type name in field path
                if hint_lower in field_name:
                    boost += type_field_boost

            # NEW: Field hint matching (high priority for utterance-aware selection)
            for hint in field_hints:
                hint_lower = hint.lower()
                # Exact field name match (e.g., hint="name" matches field="name")
                if hint_lower == actual_field:
                    boost += field_hint_exact_boost
                    logger.debug("Field hint exact match: %s in %s", hint_lower, pid)
                # Field name contains hint (e.g., hint="name" matches field="display_name")
                elif hint_lower in actual_field or actual_field in hint_lower:
                    boost += field_hint_partial_boost
                    logger.debug("Field hint partial match: %s in %s", hint_lower, pid)
                # Hint appears anywhere in property path
                elif hint_lower in full_path:
                    boost += field_hint_partial_boost // 2
                    logger.debug("Field hint path match: %s in %s", hint_lower, pid)

            total_score = base_score + boost
            scored.append((pid, total_score))

        # Sort by total score descending
        scored.sort(key=lambda x: -x[1])

        # Return top properties (those with positive scores, or top N if none positive)
        positive_scored = [(pid, s) for pid, s in scored if s > 0]
        if positive_scored:
            return set(pid for pid, _ in positive_scored)
        else:
            # Fallback to top half of results
            return set(pid for pid, _ in scored[:len(scored)//2])

    def _create_collection_from_state(self, state: MappingState):
        logger.info("Building Chroma collection: %s", self.collection_name)
        # Delete existing collection entirely to ensure clean rebuild
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info("Deleted existing collection: %s", self.collection_name)
        except Exception as e:
            logger.debug("No existing collection to delete: %s", e)
        # Create fresh collection
        col = self.chroma_client.create_collection(name=self.collection_name, embedding_function=None)

        ids, docs, metas, embs = [], [], [], []
        props = state["property_data"]

        for pid, pdata in tqdm(
            props.items(),
            desc=f"Embedding properties -> {self.collection_name}",
            disable=logger.level > logging.INFO,
        ):
            ids.append(pid)
            doc = f"{pdata.get('product_name','')} {pdata.get('type_name','')} {pdata.get('field_name','')} {pdata.get('description','')}"
            docs.append(doc)
            metas.append({
                "product_name": pdata.get("product_name", ""),
                "type_name": pdata.get("type_name", ""),
                "field_name": pdata.get("field_name", "")
            })
            embs.append(self.embedding_model.encode(doc))

        # Add in batches to avoid ChromaDB batch size limit (max ~5461)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            col.add(
                ids=ids[i:end],
                documents=docs[i:end],
                metadatas=metas[i:end],
                embeddings=embs[i:end]
            )
        logger.info("Chroma collection populated | %d items in %d batches", len(ids), (len(ids) + batch_size - 1) // batch_size)
        return col

    def _select_destination_properties(self, state: MappingState) -> MappingState:
        collection = state["chroma_collection"]
        target_product = state.get("target_product", "").lower()
        type_hints = state.get("target_type_hints", [])
        is_retrieval_mode = state.get("is_retrieval_mode", False)
        source_product = state.get("source_product", "").lower()

        # RETRIEVAL MODE: Find destination properties via hybrid search (semantic + BM25)
        if is_retrieval_mode:
            logger.info("Retrieval mode: finding destination properties via hybrid search from source")

            # Use source properties to find similar destination properties
            source_props = state.get("source_properties", [])
            if not source_props:
                # Fall back to source context if no source properties selected yet
                q = state.get("source_context", state["user_utterance"])
            else:
                # Build query from source properties
                source_texts = []
                for pid in source_props[:5]:  # Use top 5 source properties
                    pdata = state["property_data"].get(pid, {})
                    source_texts.append(f"{pdata.get('type_name', '')} {pdata.get('field_name', '')} {pdata.get('description', '')}")
                q = " ".join(source_texts)

            q_emb = self.embedding_model.encode(q).tolist()

            # --- Semantic retrieval via ChromaDB ---
            # Search broadly to ensure we find cross-product matches
            # Cap at 5000 to avoid ChromaDB batch size limit (max ~5461)
            n_results = min(len(state["property_data"]), 5000)
            logger.info("Retrieval mode: querying ChromaDB n_results=%d, collection_count=%d",
                       n_results, collection.count())
            res = collection.query(query_embeddings=[q_emb], n_results=n_results)
            semantic_ids = res["ids"][0] if res.get("ids") else []
            logger.info("Retrieval mode: ChromaDB returned %d semantic results", len(semantic_ids))

            # --- BM25 retrieval (if available) ---
            bm25_ids = []
            if HAS_BM25:
                # Get or create BM25 retriever
                if self.collection_name not in _bm25_retrievers:
                    bm25_retriever = BM25Retriever(self.collection_name, self.chroma_client._persist_directory)
                    if not bm25_retriever.load_index():
                        # Build index from property data
                        logger.info(f"Building BM25 index for collection '{self.collection_name}'...")
                        documents = []
                        for pid, pdata in state["property_data"].items():
                            doc_text = f"{pdata.get('type_name', '')} {pdata.get('field_name', '')} {pdata.get('description', '')}"
                            documents.append({"id": pid, "doc": doc_text})
                        bm25_retriever.build_index(documents)
                    _bm25_retrievers[self.collection_name] = bm25_retriever

                bm25_retriever = _bm25_retrievers[self.collection_name]
                bm25_ids = bm25_retriever.search(q, top_k=n_results)
                logger.info("Retrieval mode: BM25 returned %d results", len(bm25_ids))

            # --- Merge with Reciprocal Rank Fusion ---
            if bm25_ids:
                all_ids = reciprocal_rank_fusion(semantic_ids, bm25_ids)
                logger.info("Retrieval mode: RRF merged to %d unique results", len(all_ids))
            else:
                all_ids = semantic_ids

            # Filter out properties from the source product
            filtered_ids = []
            for pid in all_ids:
                pdata = state["property_data"].get(pid, {})
                prop_product = pdata.get("product_name", "").lower()
                # Exclude same product as source
                if source_product and prop_product == source_product:
                    continue
                filtered_ids.append(pid)
                if len(filtered_ids) >= self.top_k:
                    break

            logger.info("Retrieval mode: found %d destination properties (excluded source_product=%s, searched %d total)",
                       len(filtered_ids), source_product, len(all_ids))

            # Set destination properties and return
            state["destination_properties"] = filtered_ids
            return state

        # ORACLE MODE: Target product is known
        # For strict schema filtering (OMOP), filter by product FIRST before semantic search
        if self.strict_schema_filter and target_product:
            # Get all properties matching the product
            matching_ids = self._get_properties_by_product(state["property_data"], target_product, type_hints)
            logger.info("Strict filter (dest): found %d properties matching product=%s", len(matching_ids), target_product)

            if matching_ids:
                # Now do semantic search and intersect with type-filtered properties
                q = state["destination_context"]
                q_emb = self.embedding_model.encode(q).tolist()
                # Search more results to increase chance of finding type-matching ones
                n_results = min(len(state["property_data"]), self.top_k * self.n_results_multiplier * 5)
                res = collection.query(query_embeddings=[q_emb], n_results=n_results)
                semantic_ids = res["ids"][0] if res.get("ids") else []

                # Intersect semantic results with product-filtered results
                filtered_ids = [pid for pid in semantic_ids if pid in matching_ids][:self.top_k]
                logger.info("Strict filter (dest): %d semantic results intersected to %d", len(semantic_ids), len(filtered_ids))

                # If intersection is empty/too small, take type-filtered properties directly
                if len(filtered_ids) < self.top_k:
                    remaining = self.top_k - len(filtered_ids)
                    extras = [pid for pid in sorted(matching_ids) if pid not in filtered_ids][:remaining]
                    filtered_ids.extend(extras)
                    logger.info("Strict filter (dest): added %d type-matched properties to reach %d", len(extras), len(filtered_ids))
            else:
                # Fallback if no product match
                logger.warning("Strict filter (dest): no properties match product=%s, falling back to semantic", target_product)
                filtered_ids = []
        else:
            filtered_ids = []

        # Standard semantic search path (soft filtering or fallback)
        if not filtered_ids:
            q = state["destination_context"]
            q_emb = self.embedding_model.encode(q).tolist()

            # Get more results initially, then filter by product/type/field hints
            n_results = min(self.top_k * self.n_results_multiplier, len(state["property_data"]))
            res = collection.query(query_embeddings=[q_emb], n_results=n_results)
            all_ids = res["ids"][0] if res.get("ids") else []

            # Get field hints for utterance-aware property selection
            target_field_hints = state.get("target_field_hints", [])

            filtered_ids = self._filter_properties_by_hints(
                all_ids, state["property_data"], target_product, type_hints,
                field_hints=target_field_hints
            )

            # If filtering returned nothing, fall back to top results
            if not filtered_ids:
                filtered_ids = set(all_ids[:self.top_k])
                logger.warning("No properties matched target hints, using top %d unfiltered", self.top_k)
            else:
                # Limit to top_k after filtering
                filtered_ids = set(list(filtered_ids)[:self.top_k])

        filtered_ids = set(filtered_ids)

        # 1-hop neighbors (only if neighbor_k > 0)
        nbrs = set()
        if self.neighbor_k > 0:
            for pid in filtered_ids:
                if pid in state["graph"]:
                    nbrs.update(state["graph"].neighbors(pid))
            # When strict filtering is enabled, filter neighbors by product AND type
            if self.strict_schema_filter and target_product:
                product_lower = target_product.lower()
                type_hints_lower = [h.lower() for h in (type_hints or [])]
                filtered_nbrs = set()
                for n in nbrs:
                    if not n.lower().startswith(product_lower + "_"):
                        continue
                    # Also filter by type hints if available
                    if type_hints_lower:
                        pdata = state["property_data"].get(n, {})
                        tname = pdata.get("type_name", "").lower()
                        if any(hint in tname for hint in type_hints_lower):
                            filtered_nbrs.add(n)
                    else:
                        filtered_nbrs.add(n)
                nbrs = filtered_nbrs
            nbrs = set(sorted(nbrs)[: self.neighbor_k])

        # Keep filtered_ids first (by semantic order), then append neighbors
        combined = list(filtered_ids) + [n for n in sorted(nbrs) if n not in filtered_ids]
        # Remove any that are already in source properties
        filtered = [x for x in combined if x not in state["source_properties"]]
        state["destination_properties"] = filtered
        logger.info("Selected destination properties | filtered=%d +nbr=%d -> %d (product=%s, hints=%s, strict=%s)",
                    len(filtered_ids), len(nbrs), len(filtered), target_product, type_hints, self.strict_schema_filter)
        return state

    def _find_mappings(self, state: MappingState) -> MappingState:
        srcs = state["source_properties"]
        dsts = state["destination_properties"]
        pdata = state["property_data"]
        user_utterance = state["user_utterance"]
        source_field_hints = state.get("source_field_hints", [])
        target_field_hints = state.get("target_field_hints", [])

        if not srcs or not dsts:
            logger.warning("Insufficient properties for mapping | src=%d dst=%d", len(srcs), len(dsts))
            state["candidate_mappings"] = {}
            return state

        logger.info("Finding mappings | evaluating %d source × %d dest = %d pairs",
                    len(srcs), len(dsts), len(srcs) * len(dsts))

        # Build prompt with utterance context for better field-level mapping
        prompt = [
            f'User Request: "{user_utterance}"',
            "",
        ]

        # Add field hints if available
        if source_field_hints or target_field_hints:
            prompt.append("Key fields/attributes mentioned in request:")
            if source_field_hints:
                prompt.append(f"  Source fields: {', '.join(source_field_hints)}")
            if target_field_hints:
                prompt.append(f"  Target fields: {', '.join(target_field_hints)}")
            prompt.append("")

        prompt.extend([
            "Based on the user's request above, evaluate which source properties should map to which destination properties.",
            "PRIORITIZE mappings that semantically match the user's intent - focus on properties whose names or descriptions relate to the fields mentioned in the request.",
            "",
            "Return ONLY a JSON array. No preamble, no explanation.",
            "Format: [{\"source_property\": \"id\", \"destination_property\": \"id\", \"valid\": true, \"confidence\": 0.9, \"justification\": \"reason\"}]",
            "Only include entries where valid=true.",
            "Your entire response must be ONLY the JSON array starting with [ and ending with ].",
            "\nCandidate Pairs:\n"
        ])

        for sp in srcs:
            si = pdata[sp]
            prompt.append(f"\nSource: {sp}\n - Product: {si['product_name']}\n - Type: {si['type_name']}\n - Field: {si['field_name']}\n - Description: {si['description']}")
            for dp in dsts:
                di = pdata[dp]
                prompt.append(
                    f"  Candidate Destination: {dp}\n"
                    f"   - Product: {di['product_name']}\n"
                    f"   - Type: {di['type_name']}\n"
                    f"   - Field: {di['field_name']}\n"
                    f"   - Description: {di['description']}"
                )

        messages = [
            {"role": "system", "content": "You are an expert in data mapping and schema alignment. Prioritize mappings that match the user's specific request. Return only valid JSON with no explanatory text."},
            {"role": "user", "content": "\n".join(prompt)}
        ]

        logger.debug("Evaluating %d source x %d dest = %d pairs", len(srcs), len(dsts), len(srcs) * len(dsts))

        raw = self.call_llm(messages)

        # Debug logging (only at DEBUG level)
        logger.debug("LLM Response length: %d characters", len(raw) if raw else 0)
        if len(raw) < 1000:
            logger.debug("Full LLM response:\n%s", raw)
        else:
            logger.debug("LLM response preview (first 500):\n%s", raw[:500])

        raw = _strip_code_fences(raw)

        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            parsed, err = _safe_json_loads(match.group())
        else:
            parsed, err = _safe_json_loads(raw)

        if err or not isinstance(parsed, list):
            logger.warning("Mapping JSON parse failed; err=%s", err)
            logger.info("Response length: %d characters", len(raw))
            logger.info("Response preview (first 500 chars):\n%s", raw[:500])
            logger.info("Response end (last 200 chars):\n%s", raw[-200:] if len(raw) > 200 else raw)
            
            # Try to extract JSON from verbose response
            # Look for JSON array anywhere in the response
            json_match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', raw)
            if json_match:
                try:
                    extracted_json = json_match.group()
                    logger.info("Found JSON array in response, attempting to parse...")
                    parsed = json.loads(extracted_json)
                    logger.info("Successfully recovered JSON array from verbose response (%d items)", len(parsed))
                except Exception as e2:
                    logger.warning("Failed to parse extracted JSON: %s", e2)
                    logger.debug("Extracted text was: %s", extracted_json[:500])
                    parsed = []
            else:
                logger.warning("No JSON array pattern found in response")
                parsed = []
    
        cand = defaultdict(list)
        valid_count = 0
        invalid_count = 0
        
        for m in parsed:
            if isinstance(m, dict):
                if m.get("valid") is True:
                    valid_count += 1
                    src = m.get("source_property")
                    dst = m.get("destination_property")
                    if not src or not dst:
                        logger.warning("Skipping mapping with missing src or dst: %s", m)
                        continue
                    cand[src].append({
                        "mapProperty": dst,
                        "justification": m.get("justification", ""),
                        "confidence": float(m.get("confidence", 0.5))
                    })
                else:
                    invalid_count += 1
    
        state["candidate_mappings"] = dict(cand)
        total_mappings = sum(len(v) for v in cand.values())
        
        logger.info("Parsed response: %d valid, %d invalid mappings", valid_count, invalid_count)
        
        if cand:
            logger.info("Candidate mappings found | %d sources with %d total mappings", 
                        len(cand), total_mappings)
        else:
            logger.warning("No candidate mappings found")
        
        return state

    def _create_reflection_prompt(self, candidate_mappings: Dict, property_data: Dict,
                                    user_utterance: str = "", field_hints: List[str] = None) -> str:
        """Create reflection prompt with utterance context for better filtering."""
        lines = []

        # Add user context at the top
        if user_utterance:
            lines.append(f'User Request: "{user_utterance}"')
            lines.append("")
            if field_hints:
                lines.append(f"Key fields mentioned: {', '.join(field_hints)}")
                lines.append("")

        lines.append("Review the following property mappings against the user's request.")
        lines.append("REMOVE mappings that don't align with what the user is asking for.")
        lines.append("KEEP mappings that match the user's intent and the fields they mentioned.")
        lines.append("")

        for sp, maps in candidate_mappings.items():
            sdesc = property_data.get(sp, {}).get("description", "")
            lines.append(f"\nSource: {sp} ({sdesc[:100]}...)")
            for m in maps:
                dp = m["mapProperty"]
                ddesc = property_data.get(dp, {}).get("description", "")
                lines.append(f"  -> {dp} ({ddesc[:100]}...)\n     Justification: {m['justification']}\n     Confidence: {m['confidence']}")

        lines.append("\nConsider: alignment with user's request, type compatibility, purpose similarity, business value.")
        return "\n".join(lines)

    def _reflect_and_prune(self, state: MappingState) -> MappingState:
        cand = state["candidate_mappings"]
        user_utterance = state["user_utterance"]
        source_field_hints = state.get("source_field_hints", [])
        target_field_hints = state.get("target_field_hints", [])
        all_field_hints = list(set(source_field_hints + target_field_hints))

        if not cand:
            state["final_mappings"] = {}
            state["reflection_feedback"] = "No candidate mappings were generated"
            logger.info("No candidates to reflect on")
            return state

        prompt = self._create_reflection_prompt(cand, state["property_data"],
                                                 user_utterance, all_field_hints)
        messages = [
            {"role": "system", "content": "You are an expert data mapper tasked with validating property mappings against user requirements. Review each mapping critically and provide feedback."},
            {"role": "user", "content": prompt}
        ]
        feedback = self.call_llm(messages)

        prune_prompt = [
            f'User Request: "{user_utterance}"',
            "",
            f"Feedback: {feedback}",
            "",
            "Based on this feedback and the user's original request, prune the candidate mappings below.",
            "ONLY KEEP mappings that align with what the user is asking for.",
            "Return ONLY a JSON object. No text before or after.",
            'Format: {"source_id": [{"mapProperty": "dest_id", "justification": "reason"}]}',
            "\nYour entire response must be ONLY the JSON object starting with { and ending with }.",
            "\nCandidate mappings:"
        ]
        for sp, maps in cand.items():
            prune_prompt.append(f"\nSource: {sp}")
            for m in maps:
                prune_prompt.append(f"  -> {m['mapProperty']} (Confidence: {m['confidence']})\n     Justification: {m['justification']}")

        messages = [
            {"role": "system", "content": "You are an expert in schema mapping. Filter mappings to match the user's intent. Output only the final valid mappings in JSON."},
            {"role": "user", "content": "\n".join(prune_prompt)}
        ]

        raw = self.call_llm(messages)
        raw = _strip_code_fences(raw)
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            final, err = _safe_json_loads(match.group())
        else:
            final, err = None, "No JSON found"

        if err or not isinstance(final, dict):
            logger.warning("Pruned JSON parse failed; falling back to confidence filter; err=%s raw=%s", err, raw[:200])
            final = {
                s: [{"mapProperty": m["mapProperty"], "justification": m.get("justification", "")}
                    for m in maps if float(m.get("confidence", 0.0)) >= 0.7]
                for s, maps in cand.items()
            }

        state["final_mappings"] = final
        state["reflection_feedback"] = str(feedback)
        logger.info("Final mappings after reflection | sources=%d", len(final))
        return state

    def _format_output(self, state: MappingState) -> MappingState:
        if not state.get("final_mappings"):
            state["final_mappings"] = state.get("candidate_mappings", {})
        return state

    # ---- public ----
    def run(self, user_utterance: str) -> Dict[str, Any]:
        import time
        logger.info("MapReflect run | utterance=%r model=%s", user_utterance, self.model)
        initial = MappingState(
            user_utterance=user_utterance,
            property_data=self.property_data,
            source_properties=[],
            destination_properties=[],
            candidate_mappings={},
            final_mappings={},
            reflection_feedback="",
            graph=nx.Graph(),
            chroma_collection=None,
            source_context="",
            destination_context="",
            source_product="",
            target_product="",
            source_type_hints=[],
            target_type_hints=[],
            # New fields for utterance-aware field-level mapping
            source_field_hints=[],
            target_field_hints=[],
            # Retrieval mode flag (set during context extraction)
            is_retrieval_mode=False
        )

        # Run with timing for each step
        state = initial
        steps = []

        # Add initialize_index step if rebuild_index is True
        if self.rebuild_index:
            steps.append(("initialize_index", self._initialize_index))

        steps.extend([
            ("extract_contexts", self._extract_contexts),
            ("extract_field_hints", self._extract_field_hints),  # NEW: Extract field hints from utterance
            ("create_graph", self._create_property_graph),
            ("select_source_properties", self._select_source_properties),
            ("select_destination_properties", self._select_destination_properties),
            ("find_mappings", self._find_mappings),
            ("format_output", self._format_output),
        ])

        total_start = time.time()
        for step_name, step_fn in steps:
            step_start = time.time()
            state = step_fn(state)
            step_time = time.time() - step_start
            logger.warning(f"TIMING | {step_name}: {step_time:.2f}s")
        total_time = time.time() - total_start
        logger.warning(f"TIMING | TOTAL: {total_time:.2f}s")

        out = {
            "mappings": state["final_mappings"],
            "source_properties": state["source_properties"],
            "destination_properties": state["destination_properties"],
            "reflection_feedback": state["reflection_feedback"]
        }
        logger.info("MapReflect done | final_sources=%d", len(out["mappings"]))
        return out

# Example usage
def main():
    logging.basicConfig(level=logging.WARNING)
    agent = MapReflectAgent(
        property_lookup_file="../data/itops_api_lookup.json",
        dataset="test",
        model="provider/mistralai/mistral-large"  # Example model specification
    )
    user_input = "map resilience vulnerabilities to optimizer entities"
    result = agent.run(user_input)
    print("\nFinal Mappings:")
    print(json.dumps(result["mappings"], indent=2))

if __name__ == "__main__":
    main()
