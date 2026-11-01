import json
import os
import logging
import aiohttp
import asyncio
import threading
import uvicorn
from mcp.types import TextContent
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, Any

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - Line:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sampling_server.log", mode='w')]
)

# ==============================================================================
# VISUALIZATION SERVER (runs on localhost:9001)
# ==============================================================================

# Use a lock to safely update the shared data between threads
data_lock = threading.Lock()
# This global variable will store the latest result to be displayed on the webpage
latest_result_for_visualization = {"type": "none", "content": {}}

# Initialize FastAPI app for the visualization server
app_vis = FastAPI()
# Initialize Jinja2 templates to render the HTML file
templates = Jinja2Templates(directory=".")

@app_vis.post("/update_data")
async def update_data(request: Request):
    """Receives data from the main MCP tools and stores it."""
    global latest_result_for_visualization
    data = await request.json()
    with data_lock:
        latest_result_for_visualization = data
    logging.info(f"Visualization data updated for type: {data.get('type')}")
    return {"status": "success", "message": "Data updated for visualization."}

@app_vis.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the index.html page with the latest data injected."""
    with data_lock:
        # Pass a copy of the data to the template
        data_to_render = latest_result_for_visualization.copy()
    return templates.TemplateResponse("index.html", {"request": request, "data": json.dumps(data_to_render)})

def run_visualization_server():
    """Runs the Uvicorn server for visualization in this function."""
    logging.info("Starting visualization server on http://localhost:9001")
    uvicorn.run(app_vis, host="0.0.0.0", port=9001, log_level="warning")


# ==============================================================================
# MAIN MCP TRACE SAMPLING SERVER (runs on localhost:8000)
# ==============================================================================

mcp = FastMCP("Trace Sampling Server", host="0.0.0.0", port=8102)

TVM_BASE_URL = os.getenv("TVM_BASE_URL", "http://127.0.0.1:8000")
# TVM_BASE_URL = "http://127.0.0.1:8000"
VIS_SERVER_URL = "http://localhost:9001"

async def send_data_to_visualizer(data: Dict[str, Any]):
    """Sends the result dictionary to the visualization server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{VIS_SERVER_URL}/update_data", json=data) as response:
                if response.status == 200:
                    logging.info("Successfully sent data to visualization server.")
                else:
                    logging.error(f"Failed to send data to visualizer: {response.status} - {await response.text()}")
    except aiohttp.ClientError as e:
        logging.error(f"Could not connect to the visualization server: {e}")


async def make_tvm_request(endpoint: str, data: Dict[str, Any] = None, timeout: int = 3600) -> Dict[str, Any]:
    """Make HTTP request to TVM API"""
    url = f"{TVM_BASE_URL}{endpoint}"
    
    try:
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(url, json=data, headers={'Content-Type': 'application/json', 'accept': 'application/json'}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status, 
                        detail=f"TVM API error: {response.status} - {error_text}"
                    )
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=500, detail=f"Connection error to TVM API: {str(e)}")
    except asyncio.TimeoutError: # This needs to be imported: `import asyncio`
        raise HTTPException(status_code=504, detail=f"TVM API request timed out after {timeout} seconds")

# This tool is responsible for running all the listed sampling methods in VM and return the information retention and similarity metrics, the llm can then pick the best method based on the metrics and requirements
@mcp.tool()
async def evaluate_sampling_methods(sampling_rate: float, window: int, trace_file_path: str):
    """Evaluate all sampling methods and compare their performance.
    
    Args:
        sampling_rate: Sampling rate for sampling
        window: Time window for sampling
        trace_file_path: Path to the input trace file
    
    Returns:
        Comparison of all sampling methods with their metrics
    """
    try:
        logging.info("="*80)
        logging.info("EVALUATE_SAMPLING_METHODS CALLED")
        
        request_data = {
            "sampling_rate": sampling_rate,
            "window": window,
            "trace_location": trace_file_path
        }
        
        logging.info(f"Sending request to TVM with data: {request_data}")
        data = await make_tvm_request("/evaluate", request_data)
        logging.info(f"Received response from TVM: {json.dumps(data, indent=2)}")
        
        result = {
            "status": "success",
            "data": data,
            "request_parameters": {
                "sampling_rate": sampling_rate, "window": window, "trace_file_path": trace_file_path
            }
        }

        # --- NEW: Send data to visualization server ---
        await send_data_to_visualizer({"type": "evaluate", "content": result})
        
        logging.info("EVALUATE_SAMPLING_METHODS COMPLETED SUCCESSFULLY")
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        logging.error(f"ERROR IN EVALUATE_SAMPLING_METHODS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error evaluating sampling methods: {str(e)}")


# This tool is responsible for executing a particular sampling method with the given parametre and output the sampling file.
@mcp.tool()
async def execute_sampling(sampling_method: str, sampling_rate: float, window: int, trace_file_path: str):
    """Execute a specific sampling method.
    
    Args:
        sampling_method: One of 'windowedStratifiedEntropyMaxSampling', 'clusterSampling', 'windowedStratifiedRandomSamplingSingle', 'windowedPureRandomSamplingSingle'
        sampling rate: Fraction of traces to sample
        window: Time window for sampling
        trace_file_path: Path to the input trace file
    
    Returns:
        Results from the specific sampling method
    """
    try:
        logging.info("="*80)
        logging.info("EXECUTE_SAMPLING CALLED")

        valid_methods = [
            'windowedStratifiedEntropyMaxSampling', 'clusterSampling', 
            'windowedStratifiedRandomSamplingSingle', 'windowedPureRandomSamplingSingle'
        ]
        
        if sampling_method not in valid_methods:
            raise HTTPException(status_code=400, detail=f"Invalid sampling method. Must be one of: {valid_methods}")
        
        request_data = {
            "sampling_method": sampling_method, "sampling_rate": sampling_rate,
            "window": window, "trace_location": trace_file_path
        }
        
        logging.info(f"Sending request to TVM with data: {request_data}")
        data = await make_tvm_request("/execute", request_data)
        logging.info(f"Received response from TVM: {json.dumps(data, indent=2)}")
        
        result = {
            "status": "success",
            "data": data,
            "request_parameters": {
                "sampling_method": sampling_method, "sampling_rate": sampling_rate,
                "window": window, "trace_file_path": trace_file_path
            }
        }

        # --- NEW: Send data to visualization server ---
        await send_data_to_visualizer({"type": "execute", "content": result})
        
        logging.info("EXECUTE_SAMPLING COMPLETED SUCCESSFULLY")
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
  
    except Exception as e:
        logging.error(f"ERROR IN EXECUTE_SAMPLING: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing sampling method: {str(e)}")


def start_mcp_server():
    """Starts the main MCP server."""
    mcp.run(transport="sse")

if __name__ == "__main__":
    # Start the visualization server in a separate thread
    # The `daemon=True` flag means the thread will exit when the main program exits
    vis_thread = threading.Thread(target=run_visualization_server, daemon=True)
    vis_thread.start()
    
    # Give the visualization server a moment to start up
    asyncio.run(asyncio.sleep(2))

    # Start the main MCP server in the main thread
    start_mcp_server()
