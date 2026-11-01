from fetch_data import get_data_schema, extract_data, filter_csv_by_col
from preprocessing import preprocess_data
from call_LLM import remove_redundant_columns, visualize_data
from generate_vis import process_spec, create_vegalite_chart
from evaluate import evaluate, parse_evaluation_response
import os
import pandas as pd
import json

class Data2Dashboard:
    def __init__(self, input_csv, output_dir, query_csv):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.query = query_csv
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        df = pd.read_csv(self.input_csv)
        df_query = pd.read_csv(self.query_csv)
        nl_query = "Give me an overview of the apm application metrics."

        for query in df_query[query]:

            schema = get_data_schema(df)
            imp_col_list = remove_redundant_columns(schema, nl_query, self.output_dir)
            df = preprocess_data(df, imp_col_list)
            schema = filter_csv_by_col(schema, imp_col_list)

            with open(os.path.join(self.output_dir, "data_schema.json"), "w") as f:
                json.dump(schema, f)

            spec_list = visualize_data(schema, nl_query, self.output_dir)
            spec_list = process_spec(spec_list, df)

            with open(os.path.join(self.output_dir, "spec_list.json"), "w") as f:
                json.dump(spec_list, f)

            for index, spec in enumerate(spec_list):
                create_vegalite_chart(spec, index, self.output_dir)

# Usage example (in separate script or main):
dashboard = Data2Dashboard("../data/apm_endpoint_metrics.csv", "../results/apm_endpoint_metrics", "../nl_queries/apm_application_metrics_generic.csv")

dashboard.run()