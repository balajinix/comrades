from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import pandas as pd
import json
import os
import sys

sys.path.append("../src")

from fetch_data import get_data_schema, filter_csv_by_col
from preprocessing import preprocess_data
from call_LLM import remove_redundant_columns, visualize_data
from generate_vis import process_spec, create_vegalite_chart
from evaluate import evaluate_charts, convert_json_to_csv

# Step 1: Load CSV and preview data
state = {}
state["input_csv"] = "~/Desktop/Data_Viz_agent/data-dashboard/data/apm_endpoint_metrics.csv"
state["output_dir"] = "~/Desktop/Data_Viz_agent/data-dashboard/results/apm_endpoint_metrics"

df = pd.read_csv(state["input_csv"])
state["df"] = df

nl_df_path = "~/Desktop/Data_Viz_agent/data-dashboard/nl_queries/apm_endpoint_metrics_specific.csv"
# nl_df_path = "~/Desktop/Data_Viz_agent/data-dashboard/nl_queries/apm_application_metrics_specific.csv"
nl_df = pd.read_csv(nl_df_path)

base_dir = "~/Desktop/Data_Viz_agent/data-dashboard/results/apm_endpoint_metrics"

for nl_query in nl_df["query"]:
    state["nl_query"] = nl_query
    state["output_dir"] = os.path.join(base_dir, state["nl_query"])
    # Step 2: Schema and Redundant Column Removal
    state["schema"] = get_data_schema(state["df"])
    imp_cols = remove_redundant_columns(state["schema"], state["nl_query"], state["output_dir"])
    state["imp_columns"] = imp_cols
    state["df"] = preprocess_data(state["df"], state["imp_columns"])
    state["filtered_schema"] = filter_csv_by_col(state["schema"], state["imp_columns"])

    # LLM Call
    spec_list = visualize_data(state["filtered_schema"], state["nl_query"], state["output_dir"])
    state["spec_list"] = spec_list

    final_specs, error, total = process_spec(state["spec_list"], state["df"], state["output_dir"])
    state["final_specs"] = final_specs

    result_li = []

    # Render charts
    for index, spec in enumerate(state["final_specs"]):
        answer_json = create_vegalite_chart(spec, index, state["output_dir"])
        if answer_json == {}:
            error = error + 1
        result_li.append(answer_json)

    final_li_path = os.path.join(base_dir, state["nl_query"], "final_li_spec.json")

    with open(final_li_path, "w") as f:
        json.dump(result_li, f)

    vis_error_rate = error/total

    # run the evaluation for each chart that has been created
    eval_result = evaluate_charts(final_li_path, state["nl_query"], vis_error_rate)

    eval_result_path = os.path.join(base_dir, state["nl_query"], "final_eval_result.json")

    with open(eval_result_path, "w") as f:
        json.dump(eval_result, f)

    output_csv = os.path.join(base_dir, state["nl_query"], "final_result.csv")

    vis_error_path = os.path.join(base_dir, state["nl_query"], "vis_error_rate.json")

    vis_err_dic = {"vis_error_rate": vis_error_rate}

    with open(vis_error_path, "w") as f:
        json.dump(vis_err_dic, f)

    convert_json_to_csv(eval_result_path, output_csv, state["nl_query"])

