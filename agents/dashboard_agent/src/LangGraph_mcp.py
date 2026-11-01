from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import pandas as pd
import json
import os
import asyncio

# ==== Step 1: Import all functions from your pipeline ====
from fetch_data import get_data_schema, filter_csv_by_col
from preprocessing import preprocess_data
from call_LLM import remove_redundant_columns, visualize_data
from generate_vis import process_spec, create_vegalite_chart

# ==== Step 2: Import MCP tools directly ====
from mcp_tools import (
    filter_data_schema_tool,
    process_vis_spec_tool,
    render_vega_charts_tool,
)

# ==== Step 3: Helper function to run async tools ====
def run_async_tool(tool_func, **kwargs):
    """Helper function to run async MCP tools synchronously"""
    return asyncio.run(tool_func(**kwargs))

# ==== Step 4: Define Each Node ====
def load_csv_node(state):
    print("DEBUG: load_csv_node received state:", state)
    df = pd.read_csv(state["input_csv"])
    state["df"] = df
    state["nl_query"] = "Give me an overview of the apm application metrics."
    return state

def get_schema_node(state):
    state["schema"] = get_data_schema(state["df"])
    return state

def remove_columns_node(state):
    imp_cols = remove_redundant_columns(state["schema"], state["nl_query"], state["output_dir"])
    state["imp_columns"] = imp_cols
    return state

def preprocess_node(state):
    df = preprocess_data(state["df"], state["imp_columns"])
    state["df"] = df
    return state

# ==== Step 5: MCP Tool Integration Nodes ====
def filter_schema_node(state):
    """Node that uses MCP filter_data_schema_tool"""
    result = run_async_tool(
        filter_data_schema_tool,
        data_schema=state["schema"],
        imp_columns=state["imp_columns"],
        output_dir=state["output_dir"]
    )
    state["filtered_schema"] = result["filtered_schema"]
    return state

def generate_vis_node(state):
    spec_list = visualize_data(state["filtered_schema"], state["nl_query"], state["output_dir"])
    state["spec_list"] = spec_list
    return state

def process_specs_node(state):
    """Node that uses MCP process_vis_spec_tool"""
    # Convert DataFrame to dict for JSON serialization
    df_dict = state["df"].to_dict() if hasattr(state["df"], 'to_dict') else state["df"]
    
    result = run_async_tool(
        process_vis_spec_tool,
        spec_list=state["spec_list"],
        df=df_dict,
        output_dir=state["output_dir"]
    )
    state["final_specs"] = result["final_specs"]
    return state

def render_charts_node(state):
    """Node that uses MCP render_vega_charts_tool"""
    result = run_async_tool(
        render_vega_charts_tool,
        final_specs=state["final_specs"],
        output_dir=state["output_dir"]
    )
    print(f"Charts rendered: {result['message']}")
    return state

# ==== Step 6: Build the LangGraph ====
def build_graph():
    builder = StateGraph(state_schema=dict)
    
    # Add nodes with RunnableLambda wrappers
    builder.add_node("load_csv", RunnableLambda(load_csv_node))
    builder.add_node("get_schema", RunnableLambda(get_schema_node))
    builder.add_node("remove_redundant_columns", RunnableLambda(remove_columns_node))
    builder.add_node("preprocess_df", RunnableLambda(preprocess_node))
    builder.add_node("filter_data_schema", RunnableLambda(filter_schema_node))
    builder.add_node("generate_vis", RunnableLambda(generate_vis_node))
    builder.add_node("process_vis_spec", RunnableLambda(process_specs_node))
    builder.add_node("render_vega_charts", RunnableLambda(render_charts_node))
    
    # Set entry point
    builder.set_entry_point("load_csv")
    
    # Add edges
    builder.add_edge("load_csv", "get_schema")
    builder.add_edge("get_schema", "remove_redundant_columns")
    builder.add_edge("remove_redundant_columns", "preprocess_df")
    builder.add_edge("preprocess_df", "filter_data_schema")
    builder.add_edge("filter_data_schema", "generate_vis")
    builder.add_edge("generate_vis", "process_vis_spec")
    builder.add_edge("process_vis_spec", "render_vega_charts")
    
    # Set finish point
    builder.set_finish_point("render_vega_charts")
    
    return builder.compile()

if __name__ == "__main__":
    # Build and run the graph
    graph = build_graph()
    
    # Run the graph
    graph.invoke({
        "input_csv": "~/Desktop/Data_Viz_agent/data-dashboard/data/apm_application_metrics.csv",
        "output_dir": "~/Desktop/Data_Viz_agent/data-dashboard/results/apm_application_metrics"
    })