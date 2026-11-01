# mcp_tools.py
import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List
from fetch_data import filter_csv_by_col
from generate_vis import process_spec, create_vegalite_chart
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("MCP Tools Server")

@mcp.tool()
async def filter_data_schema_tool(data_schema: Dict[str, Any], imp_columns: List[str], output_dir: str) -> Dict[str, Any]:
    """Filter the CSV schema by important columns and save to data_schema.json"""
    try:
        filtered_schema = filter_csv_by_col(data_schema, imp_columns)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save filtered schema
        with open(os.path.join(output_dir, "data_schema.json"), "w") as f:
            json.dump(filtered_schema, f, indent=2)
        
        return {"filtered_schema": filtered_schema}
    except Exception as e:
        logging.error(f"Error filtering schema: {str(e)}")
        raise Exception(f"Error filtering schema: {str(e)}")

@mcp.tool()
async def process_vis_spec_tool(spec_list: List[Dict[str, Any]], df: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Process visualization specs using the dataframe and save to spec_list.json"""
    try:
        # Convert dict back to DataFrame if needed
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        
        final_specs = process_spec(spec_list, df, output_dir)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final specs
        with open(os.path.join(output_dir, "spec_list.json"), "w") as f:
            json.dump(final_specs, f, indent=2)
        
        return {"final_specs": final_specs}
    except Exception as e:
        logging.error(f"Error processing specs: {str(e)}")
        raise Exception(f"Error processing specs: {str(e)}")

@mcp.tool()
async def render_vega_charts_tool(final_specs: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """Render Vega-Lite charts from spec list and save to disk."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        final_specs = final_specs[0]
        # Render each chart
        for index, spec in enumerate(final_specs):
            create_vegalite_chart(spec, index, output_dir)
        
        return {
            "status": "success", 
            "message": f"Rendered {len(final_specs)} charts to {output_dir}",
            "charts_count": len(final_specs)
        }
    except Exception as e:
        logging.error(f"Error rendering charts: {str(e)}")
        raise Exception(f"Error rendering charts: {str(e)}")

# Run the MCP server
if __name__ == "__main__":
    # FastMCP should be run using its own run method
    mcp.run()