from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from graphdb_tool import populate_graphdb_graph
from wiki_tool import process_mapping_dict
import json

# Create MCP server
mcp = FastMCP("SchemaMappingTools", host="127.0.0.1", port=3001)

@mcp.tool()
def populate_graphdb(data: Dict[str, Any]) -> str:
    """Store mapping data in GraphDB graph database"""
    try:
        return populate_graphdb_graph(data)
    except Exception as e:
        return f"Error storing in GraphDB: {str(e)}"

@mcp.tool()
def generate_wiki(content:  Dict[str, Any]) -> str:
    """Generate wiki documentation from mapping data"""
    try:
        return process_mapping_dict(content)
    except Exception as e:
        return f"Error generating wiki: {str(e)}"

def start_mcp_server():
    """Starts the MCP server using FastMCP's async run method"""
    print("Starting MCP Server on port 3001...")
    mcp.run(transport="sse")

if __name__ == "__main__":
    start_mcp_server()
