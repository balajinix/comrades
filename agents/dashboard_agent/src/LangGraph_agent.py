from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import pandas as pd
import json
import os

# ==== Step 1: Import all functions from your pipeline ====
from fetch_data import get_data_schema, filter_csv_by_col
from preprocessing import preprocess_data
from call_LLM import remove_redundant_columns, visualize_data
from generate_vis import process_spec, create_vegalite_chart

# ==== Step 3: Define Each Node ====

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


def filter_schema_node(state):
    state["filtered_schema"] = filter_csv_by_col(state["schema"], state["imp_columns"])
    with open(os.path.join(state["output_dir"], "data_schema.json"), "w") as f:
        json.dump(state["filtered_schema"], f)
    return state


def generate_vis_node(state):
    spec_list = visualize_data(state["filtered_schema"], state["nl_query"], state["output_dir"])
    state["spec_list"] = spec_list
    return state


def process_specs_node(state):
    final_specs = process_spec(state["spec_list"], state["df"])
    with open(os.path.join(state["output_dir"], "spec_list.json"), "w") as f:
        json.dump(final_specs, f)
    state["final_specs"] = final_specs
    return state


def render_charts_node(state):
    for index, spec in enumerate(state["final_specs"]):
        create_vegalite_chart(spec, index, state["output_dir"])
    return state

# ==== Step 4: Build the LangGraph ====
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

# ==== Step 5: Compile and Run ====
graph = builder.compile()

if __name__ == "__main__":
    graph.invoke({
        "input_csv": "../data/apm_endpoint_metrics.csv",
        "output_dir": "../results/apm_endpoint_metrics"
    })

