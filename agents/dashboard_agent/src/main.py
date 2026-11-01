from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import asyncio
import os
import tempfile
import json
from pathlib import Path

# Import your LangGraph components
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import pandas as pd

# Import all functions from your pipeline
from fetch_data import get_data_schema, filter_csv_by_col
from preprocessing import preprocess_data
from call_LLM import remove_redundant_columns, visualize_data
from generate_vis import process_spec, create_vegalite_chart

# FastAPI app initialization
app = FastAPI(title="Data Visualization Agent", version="1.0.0")

# Request/Response models for A2A communication
class AgentRequest(BaseModel):
    input_csv: str
    output_dir: Optional[str] = None
    nl_query: Optional[str] = "Give me an overview of the data."
    agent_id: Optional[str] = "data_viz_agent"

class AgentResponse(BaseModel):
    status: str
    message: str
    output_dir: Optional[str] = None
    charts_generated: Optional[int] = 0
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Define your LangGraph nodes (same as your original code)
def load_csv_node(state):
    print("DEBUG: load_csv_node received state:", state)
    try:
        df = pd.read_csv(state["input_csv"])
        state["df"] = df
        if "nl_query" not in state or not state["nl_query"]:
            state["nl_query"] = "Give me an overview of the data."
        return state
    except Exception as e:
        raise Exception(f"Error loading CSV: {str(e)}")

def get_schema_node(state):
    try:
        state["schema"] = get_data_schema(state["df"])
        return state
    except Exception as e:
        raise Exception(f"Error getting schema: {str(e)}")

def remove_columns_node(state):
    try:
        imp_cols = remove_redundant_columns(state["schema"], state["nl_query"], state["output_dir"])
        state["imp_columns"] = imp_cols
        return state
    except Exception as e:
        raise Exception(f"Error removing redundant columns: {str(e)}")

def preprocess_node(state):
    try:
        df = preprocess_data(state["df"], state["imp_columns"])
        state["df"] = df
        return state
    except Exception as e:
        raise Exception(f"Error preprocessing data: {str(e)}")

def filter_schema_node(state):
    try:
        state["filtered_schema"] = filter_csv_by_col(state["schema"], state["imp_columns"])
        with open(os.path.join(state["output_dir"], "data_schema.json"), "w") as f:
            json.dump(state["filtered_schema"], f)
        return state
    except Exception as e:
        raise Exception(f"Error filtering schema: {str(e)}")

def generate_vis_node(state):
    try:
        spec_list = visualize_data(state["filtered_schema"], state["nl_query"], state["output_dir"])
        state["spec_list"] = spec_list
        return state
    except Exception as e:
        raise Exception(f"Error generating visualizations: {str(e)}")

def process_specs_node(state):
    try:
        final_specs = process_spec(state["spec_list"], state["df"])
        with open(os.path.join(state["output_dir"], "spec_list.json"), "w") as f:
            json.dump(final_specs, f)
        state["final_specs"] = final_specs
        return state
    except Exception as e:
        raise Exception(f"Error processing specs: {str(e)}")

def render_charts_node(state):
    try:
        charts_count = 0
        for index, spec in enumerate(state["final_specs"]):
            create_vegalite_chart(spec, index, state["output_dir"])
            charts_count += 1
        state["charts_generated"] = charts_count
        return state
    except Exception as e:
        raise Exception(f"Error rendering charts: {str(e)}")

# Build the LangGraph
def create_graph():
    builder = StateGraph(state_schema=dict)
    builder.add_node("load_csv", RunnableLambda(load_csv_node))
    builder.add_node("get_schema", RunnableLambda(get_schema_node))
    builder.add_node("remove_redundant_columns", RunnableLambda(remove_columns_node))
    builder.add_node("preprocess_df", RunnableLambda(preprocess_node))
    builder.add_node("filter_data_schema", RunnableLambda(filter_schema_node))
    builder.add_node("generate_vis", RunnableLambda(generate_vis_node))
    builder.add_node("process_vis_spec", RunnableLambda(process_specs_node))
    builder.add_node("render_vega_charts", RunnableLambda(render_charts_node))
    
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

# Global graph instance
graph = create_graph()

# A2A Communication Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for the orchestrator to verify agent availability"""
    return {"status": "healthy", "agent": "data_viz_agent", "version": "1.0.0"}

@app.get("/capabilities")
async def get_capabilities():
    """Return agent capabilities for the orchestrator"""
    return {
        "agent_type": "data_visualization",
        "capabilities": [
            "csv_data_processing",
            "data_schema_analysis",
            "visualization_generation",
            "vega_lite_charts",
            "data_preprocessing"
        ],
        "input_formats": ["csv"],
        "output_formats": ["json", "vega_lite", "png", "svg"]
    }

@app.post("/process", response_model=AgentResponse)
async def process_data(request: AgentRequest):
    """Main processing endpoint for A2A communication"""
    try:
        # Validate input
        if not os.path.exists(request.input_csv):
            raise HTTPException(status_code=400, detail=f"Input CSV file not found: {request.input_csv}")
        
        # Create output directory if not provided
        output_dir = request.output_dir
        if not output_dir:
            output_dir = os.path.join(tempfile.gettempdir(), f"viz_output_{os.getpid()}")
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare state for LangGraph
        initial_state = {
            "input_csv": request.input_csv,
            "output_dir": output_dir,
            "nl_query": request.nl_query
        }
        
        # Execute the graph
        print(f"Processing request for CSV: {request.input_csv}")
        result_state = await asyncio.get_event_loop().run_in_executor(
            None, graph.invoke, initial_state
        )
        
        # Prepare response data
        response_data = {
            "schema": result_state.get("filtered_schema"),
            "specs": result_state.get("final_specs"),
            "important_columns": result_state.get("imp_columns")
        }
        
        return AgentResponse(
            status="success",
            message="Data visualization completed successfully",
            output_dir=output_dir,
            charts_generated=result_state.get("charts_generated", 0),
            data=response_data
        )
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return AgentResponse(
            status="error",
            message="Failed to process data visualization",
            error=str(e)
        )

@app.post("/process_async")
async def process_data_async(request: AgentRequest):
    """Async processing endpoint that returns immediately with a job ID"""
    # This is useful for long-running tasks
    # You can implement job tracking and status checking here
    job_id = f"job_{os.getpid()}_{hash(request.input_csv)}"
    
    # Start processing in background (implement proper job queue in production)
    asyncio.create_task(background_process(request, job_id))
    
    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Processing started in background"
    }

async def background_process(request: AgentRequest, job_id: str):
    """Background processing function"""
    try:
        # Same processing logic as synchronous version
        response = await process_data(request)
        # Store result somewhere (database, file, etc.) for later retrieval
        print(f"Job {job_id} completed: {response.status}")
    except Exception as e:
        print(f"Job {job_id} failed: {str(e)}")

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an async job"""
    # Implement job status tracking
    return {"job_id": job_id, "status": "completed", "message": "Job completed successfully"}

# Agent registration and discovery endpoints
@app.post("/register")
async def register_with_orchestrator(orchestrator_url: str):
    """Register this agent with the orchestrator"""
    # Implement registration logic with orchestrator
    return {"status": "registered", "orchestrator": orchestrator_url}

def find_available_port(start_port=8000, max_port=8100):
    """Find an available port starting from start_port"""
    import socket
    
    for port in range(start_port, max_port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("AGENT_HOST", "0.0.0.0")
    PORT = int(os.getenv("AGENT_PORT", "0"))  # 0 means find available port
    
    # Find available port if not specified
    if PORT == 0:
        PORT = find_available_port()
    
    print(f"Starting Data Visualization Agent on {HOST}:{PORT}")
    print("Agent Capabilities:")
    print("- CSV data processing")
    print("- Data visualization generation")
    print("- Vega-Lite chart creation")
    print("- Schema analysis")
    print(f"\nAgent URL: http://{HOST}:{PORT}")
    print(f"Health Check: http://{HOST}:{PORT}/health")
    print(f"Capabilities: http://{HOST}:{PORT}/capabilities")
    
    try:
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("Try specifying a different port: AGENT_PORT=8001 python3 main.py")