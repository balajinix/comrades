# app/agent.py
import os
import json
import tempfile
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from collections.abc import AsyncIterable
from datetime import datetime
from typing import Literal
import asyncio

try:
    from agent_analytics.instrumentation import agent_analytics_sdk
    agent_analytics_sdk.initialize_logging(
        logs_dir_path="./logs/analytics",
        log_filename="dashboard_visualization_agent"
    )
except ImportError:
    pass  # agent_analytics not available

from src.fetch_data import get_data_schema, filter_csv_by_col
from src.preprocessing import preprocess_data
from src.call_LLM import remove_redundant_columns, visualize_data
from src.generate_vis import process_spec, create_vegalite_chart

load_dotenv()

logger = logging.getLogger(__name__)


class ResponseFormat(BaseModel):
    status: Literal["input_required", "completed", "error"] = "completed"
    message: str
    output_dir: Optional[str] = None
    charts_generated: Optional[int] = 0
    data: Optional[Dict[str, Any]] = None


class DashboardVisualizationAgent:
    """Dashboard Visualization Agent using LangGraph pipeline."""
    
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain", "text/csv"]

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        load_dotenv()
        self.file_catalog = self._build_file_catalog()
        self.graph = self._create_graph()
        logger.info("Dashboard Visualization Agent initialized successfully")

    def _build_file_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Build a catalog of available CSV files with descriptions and keywords."""
        catalog = {
            "apm_application_metrics.csv": {
                "path": "data/apm_application_metrics.csv",
                "description": "Application performance metrics from APM including response times, throughput, and error rates",
                "keywords": [
                    "application", "app", "performance", "metrics", "apm", 
                    "response", "time", "throughput", "error", "rate", "latency",
                    "application_metrics", "app_performance"
                ],
                "domain": "application_monitoring"
            },
            "apm_endpoint_metrics.csv": {
                "path": "data/apm_endpoint_metrics.csv", 
                "description": "Endpoint-level metrics from APM showing API performance and usage patterns",
                "keywords": [
                    "endpoint", "api", "service", "apm", "metrics", 
                    "performance", "usage", "calls", "response", "latency",
                    "endpoint_metrics", "api_performance"
                ],
                "domain": "api_monitoring"
            },
            "apm_service_metrics.csv": {
                "path": "data/apm_service_metrics.csv",
                "description": "Service-level metrics from APM including dependencies and health status",
                "keywords": [
                    "service", "microservice", "dependency", "health", "apm",
                    "metrics", "status", "monitoring", "performance", "service_metrics"
                ],
                "domain": "service_monitoring"
            }
        }
        
        verified_catalog = {}
        for filename, info in catalog.items():
            file_path = info["path"]
            if not os.path.isabs(file_path):
                agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                file_path = os.path.join(agent_dir, file_path)
            
            if os.path.exists(file_path):
                info["absolute_path"] = file_path
                verified_catalog[filename] = info
                logger.info(f"Found data file: {filename} at {file_path}")
            else:
                logger.warning(f"Data file not found: {filename} at {file_path}")
        
        logger.info(f"File catalog built with {len(verified_catalog)} available files")
        return verified_catalog
    
    def _match_file_to_utterance(self, utterance: str) -> Tuple[Optional[str], float]:
        """Match user utterance to the most appropriate CSV file."""
        if not self.file_catalog:
            return None, 0.0
        
        utterance_lower = utterance.lower()
        utterance_words = set(utterance_lower.split())
        
        best_match = None
        best_score = 0.0
        
        for filename, info in self.file_catalog.items():
            score = 0.0
            
            # Check for exact keyword matches
            for keyword in info["keywords"]:
                if keyword.lower() in utterance_lower:
                    score += 2.0
            
            # Check for word overlaps
            keyword_words = set()
            for keyword in info["keywords"]:
                keyword_words.update(keyword.lower().split())
            
            word_overlap = len(utterance_words.intersection(keyword_words))
            score += word_overlap * 1.0
            
            # Bonus for filename matches
            filename_parts = filename.lower().replace('.csv', '').replace('_', ' ').split()
            filename_overlap = len(utterance_words.intersection(set(filename_parts)))
            score += filename_overlap * 3.0  # High weight for filename matches
            
            # Normalize score by number of keywords (avoid bias toward files with more keywords)
            if len(info["keywords"]) > 0:
                normalized_score = score / len(info["keywords"])
            else:
                normalized_score = score
            
            if normalized_score > best_score:
                best_score = normalized_score
                best_match = info["absolute_path"]
        
        return best_match, best_score
    
    def _get_available_files_info(self) -> str:
        """Get a user-friendly description of available data files."""
        if not self.file_catalog:
            return "No data files are currently available."
        
        info_lines = ["Available data files:"]
        for filename, info in self.file_catalog.items():
            info_lines.append(f"• {filename}: {info['description']}")
        
        return "\n".join(info_lines)
    
    def _parse_user_input(self, query: str) -> Dict[str, Any]:
        """Enhanced parsing with intelligent file matching."""
        matched_file, confidence = self._match_file_to_utterance(query)
        
        if matched_file and confidence > 0.5:
            csv_path = matched_file
            nl_query = query
        else:
            parts = query.strip().split('\n')
            csv_path = None
            nl_query = "Give me an overview of the data."
            
            for part in parts:
                part = part.strip()
                if part.endswith('.csv') or '/csv' in part.lower():
                    csv_path = part
                elif part and not part.endswith('.csv'):
                    nl_query = part
            
            # If still no CSV path found, try to find the best match anyway
            if not csv_path and self.file_catalog:
                matched_file, confidence = self._match_file_to_utterance(query)
                if matched_file:
                    csv_path = matched_file
                    nl_query = query
        
        # If we still don't have a file, list available options
        if not csv_path:
            available_files = list(self.file_catalog.keys())
            if available_files:
                first_file = list(self.file_catalog.values())[0]
                csv_path = first_file["absolute_path"]
                nl_query = query
        
        result = {
            "csv_path": csv_path,
            "nl_query": nl_query,
            "matched_confidence": confidence if matched_file else 0.0
        }
        
        return result

    def _create_graph(self):
        """Create the LangGraph for data visualization pipeline."""
        builder = StateGraph(state_schema=dict)
        
        # Add nodes
        builder.add_node("load_csv", RunnableLambda(self._load_csv_node))
        builder.add_node("get_schema", RunnableLambda(self._get_schema_node))
        builder.add_node("remove_redundant_columns", RunnableLambda(self._remove_columns_node))
        builder.add_node("preprocess_df", RunnableLambda(self._preprocess_node))
        builder.add_node("filter_data_schema", RunnableLambda(self._filter_schema_node))
        builder.add_node("generate_vis", RunnableLambda(self._generate_vis_node))
        builder.add_node("process_vis_spec", RunnableLambda(self._process_specs_node))
        builder.add_node("render_vega_charts", RunnableLambda(self._render_charts_node))
        
        # Set up the workflow
        builder.set_entry_point("load_csv")
        builder.add_edge("load_csv", "get_schema")
        builder.add_edge("get_schema", "remove_redundant_columns")
        builder.add_edge("remove_redundant_columns", "preprocess_df")
        builder.add_edge("preprocess_df", "filter_data_schema")
        builder.add_edge("filter_data_schema", "generate_vis")
        builder.add_edge("generate_vis", "process_vis_spec")
        builder.add_edge("process_vis_spec", "render_vega_charts")
        builder.set_finish_point("render_vega_charts")
        
        return builder.compile()

    def _load_csv_node(self, state):
        """Load the data to visualize.

        Two sources: a local CSV path (default), or---when ``input_csv`` is a
        ``dal:<natural-language query>`` sentinel---the central DAL agent, fetched
        over A2A. The latter makes this a true consumption agent that gets its
        data from the data-access layer rather than the filesystem.
        """
        try:
            csv_path = state["input_csv"]

            if isinstance(csv_path, str) and csv_path.startswith("dal:"):
                import asyncio
                from app.dal_client import fetch_from_dal
                nl = csv_path[len("dal:"):].strip()
                df, meta = asyncio.run(fetch_from_dal(nl))
                state["dal_meta"] = meta  # a2a_calls / success, for measurement
                if not meta.get("success") or df is None or df.empty:
                    raise ValueError(
                        f"DAL agent did not return usable data "
                        f"(a2a_calls={meta.get('a2a_calls')}): {meta.get('answer','')[:200]}")
                state["df"] = df
            else:
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
                df = pd.read_csv(csv_path, encoding='utf-8',
                                 na_values=['', 'NA', 'N/A', 'null', 'NULL'])
                if df.empty:
                    raise ValueError("CSV file is empty")
                state["df"] = df

            if "nl_query" not in state or not state["nl_query"]:
                state["nl_query"] = "Give me an overview of the data."

            return state
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def _get_schema_node(self, state):
        """Extract data schema from the dataframe."""
        try:
            state["schema"] = get_data_schema(state["df"])
            return state
        except Exception as e:
            raise Exception(f"Error getting schema: {str(e)}")

    def _remove_columns_node(self, state):
        """Remove redundant columns using LLM."""
        try:
            imp_cols = remove_redundant_columns(
                state["schema"], 
                state["nl_query"], 
                state["output_dir"]
            )
            
            available_cols = list(state["df"].columns)
            valid_imp_cols = [col for col in imp_cols if col in available_cols]
            
            if not valid_imp_cols:
                valid_imp_cols = available_cols
            
            state["imp_columns"] = valid_imp_cols
            return state
        except Exception as e:
            state["imp_columns"] = list(state["df"].columns)
            return state

    def _preprocess_node(self, state):
        """Preprocess the data."""
        try:
            df = preprocess_data(state["df"], state["imp_columns"])
            state["df"] = df
            return state
        except Exception as e:
            imp_cols = [col for col in state["imp_columns"] if col in state["df"].columns]
            state["df"] = state["df"][imp_cols] if imp_cols else state["df"]
            return state

    def _filter_schema_node(self, state):
        """Filter schema based on important columns."""
        try:
            state["filtered_schema"] = filter_csv_by_col(state["schema"], state["imp_columns"])
            with open(os.path.join(state["output_dir"], "data_schema.json"), "w") as f:
                json.dump(state["filtered_schema"], f, indent=2)
            return state
        except Exception as e:
            raise Exception(f"Error filtering schema: {str(e)}")

    def _generate_vis_node(self, state):
        """Generate visualization specifications."""
        try:
            spec_list = visualize_data(
                state["filtered_schema"], 
                state["nl_query"], 
                state["output_dir"]
            )
            state["spec_list"] = spec_list
            return state
        except Exception as e:
            state["spec_list"] = []
            return state

    def _process_specs_node(self, state):
        """Process and validate visualization specifications."""
        try:
            final_specs, error_count, total_count = process_spec(
                state["spec_list"], 
                state["df"], 
                state["output_dir"]
            )
            
            with open(os.path.join(state["output_dir"], "spec_list.json"), "w") as f:
                json.dump(final_specs, f, indent=2)
            state["final_specs"] = final_specs
            state["processing_stats"] = {
                "total_specs": total_count,
                "valid_specs": len(final_specs),
                "error_count": error_count
            }
            
            return state
        except Exception as e:
            state["final_specs"] = []
            return state

    def _render_charts_node(self, state):
        """Render the final charts."""
        try:
            charts_count = 0
            chart_files = []
            
            for index, spec in enumerate(state.get("final_specs", [])):
                try:
                    chart_file = create_vegalite_chart(spec, index, state["output_dir"])
                    if chart_file:
                        chart_files.append(chart_file)
                        charts_count += 1
                except Exception as chart_error:
                    continue
            
            state["charts_generated"] = charts_count
            state["chart_files"] = chart_files
            return state
        except Exception as e:
            state["charts_generated"] = 0
            state["chart_files"] = []
            return state

    def invoke(self, query: str, context_id: str) -> Dict[str, Any]:
        """Synchronous execution for single query."""
        try:
            parsed_input = self._parse_user_input(query)
            
            if not parsed_input["csv_path"]:
                available_info = self._get_available_files_info()
                return {
                    "status": "input_required",
                    "message": f"Could not determine which data file to use. {available_info}",
                    "output_dir": "",
                    "charts_generated": 0
                }
            
            confidence = parsed_input.get("matched_confidence", 0.0)
            output_dir = os.path.join(self.results_dir, f"viz_output_{context_id}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            initial_state = {
                "input_csv": parsed_input["csv_path"],
                "output_dir": output_dir,
                "nl_query": parsed_input["nl_query"]
            }
            
            result_state = self.graph.invoke(initial_state)
            charts_generated = result_state.get("charts_generated", 0)
            
            response_data = {
                "schema": result_state.get("filtered_schema"),
                "specs": result_state.get("final_specs"),
                "important_columns": result_state.get("imp_columns"),
                "chart_files": result_state.get("chart_files", []),
                "processing_stats": result_state.get("processing_stats", {}),
                "selected_file": os.path.basename(parsed_input["csv_path"]),
                "file_confidence": confidence
            }
            
            return {
                "status": "completed",
                "message": f"Successfully generated {charts_generated} visualizations from {os.path.basename(parsed_input['csv_path'])}",
                "output_dir": output_dir,
                "charts_generated": charts_generated,
                "data": response_data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "output_dir": "",
                "charts_generated": 0
            }

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        """Streaming execution with progress updates."""
        try:
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "Analyzing your request and finding the best data file..."
            }
            
            parsed_input = self._parse_user_input(query)
            
            if not parsed_input["csv_path"]:
                available_info = self._get_available_files_info()
                yield {
                    "is_task_complete": True,
                    "require_user_input": True,
                    "content": f"❓ Could not determine which data file to use. {available_info}\n\nPlease be more specific about what type of data you want to visualize."
                }
                return
            
            confidence = parsed_input.get("matched_confidence", 0.0)
            selected_file = os.path.basename(parsed_input["csv_path"])
            
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": f"🎯 Selected data file: {selected_file} (confidence: {confidence:.2f})\n📂 Processing your visualization request..."
            }
            
            output_dir = os.path.join(self.results_dir, f"viz_output_{context_id}")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            initial_state = {
                "input_csv": parsed_input["csv_path"],
                "output_dir": output_dir,
                "nl_query": parsed_input["nl_query"]
            }
            
            steps = [
                ("load_csv", "Loading CSV data..."),
                ("get_schema", "Analyzing data schema..."),
                ("remove_redundant_columns", "Identifying relevant columns..."),
                ("preprocess_df", "Preprocessing data..."),
                ("filter_data_schema", "Filtering schema..."),
                ("generate_vis", "Generating visualization specifications..."),
                ("process_vis_spec", "Processing specifications..."),
                ("render_vega_charts", "Rendering charts...")
            ]
            
            current_step = 0
            
            try:
                def run_pipeline():
                    return self.graph.invoke(initial_state)
                
                result_state = await asyncio.get_event_loop().run_in_executor(
                    None, run_pipeline
                )
                
                for step_name, step_message in steps:
                    current_step += 1
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Step {current_step}/{len(steps)}: {step_message}"
                    }
                    await asyncio.sleep(0.1)
                
                charts_generated = result_state.get("charts_generated", 0)
                processing_stats = result_state.get("processing_stats", {})
                chart_files = result_state.get("chart_files", [])
                
                response_data = {
                    "schema": result_state.get("filtered_schema"),
                    "specs": result_state.get("final_specs"),
                    "important_columns": result_state.get("imp_columns"),
                    "chart_files": chart_files,
                    "processing_stats": processing_stats,
                    "selected_file": selected_file,
                    "file_confidence": confidence
                }
                
                file_locations = "\n".join([f"• {os.path.abspath(f)}" for f in chart_files if f])
                chart_descriptions = []
                
                specs = result_state.get("final_specs", [])
                for i, spec in enumerate(specs):
                    chart_type = spec.get("type", "Chart")
                    chart_title = spec.get("title", f"Chart {i+1}")
                    chart_reason = spec.get("reason", "Data visualization")
                    chart_descriptions.append(f"• **{chart_type}**: {chart_title} - {chart_reason}")
                
                final_message = f"""🎉 Smart dashboard generation completed successfully!
    
    **Results Summary:**
    - **Data Source:** {selected_file} (selected with {confidence:.2f} confidence)
    - **Generated:** {charts_generated} interactive visualizations
    - **Processed:** {len(result_state.get('imp_columns', []))} important columns  
    - **Success Rate:** {processing_stats.get('valid_specs', 0)}/{processing_stats.get('total_specs', 1)} specifications
    
    **Generated Chart Files:**
    {file_locations}
    
    **Chart Types Created:**
    {chr(10).join(chart_descriptions) if chart_descriptions else "• Multiple visualization types generated"}
    
    **Important Columns Analyzed:**
    {', '.join(result_state.get('imp_columns', []))}

    **Total Processing Time:** Successfully completed all {len(steps)} pipeline steps"""
    
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": final_message,
                    "output_dir": output_dir,
                    "charts_generated": charts_generated,
                    "data": response_data
                }
                
            except Exception as graph_error:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": f"Error during visualization generation: {str(graph_error)}"
                }
                return
                
        except Exception as e:
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error: {str(e)}",
                "status": "error"
            }
