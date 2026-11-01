# Data2Dashboard

## About the Project

Data2Dashboard is an LLM-driven pipeline that translates natural language (NL) queries and tabular data into coherent, multi-chart dashboards. Built as a LangGraph agent compatible with the A2A framework, this system performs:

Schema-aware preprocessing using LLMs to remove redundant or sparse columns,
Chain-of-Thought (CoT) prompting to reason about appropriate visualizations, and
Post-processing with syntactic validation and semantic evaluation using Vision-Language Models (VLMs).
The ultimate goal is to streamline dashboard creation for Site Reliability Engineers (SREs), enabling adaptive, interpretable, and user-in-the-loop visual analytics without manual effort or expert knowledge in visualization design.

## Folder Structure and Running the Code

| Folder / File   | Description                                                                   |
|------------------|-------------------------------------------------------------------------------|
| `data/`          | Contains the 3 CSV files used in the pipeline.                                |
| `nl_queries/`    | Contains generic and specific NL queries for the 3 CSV files.                 |
| `notebooks/`     | Includes complete code walkthroughs with outputs.                             |
| `prompts/`       | Stores the prompts provided to the LLM and VLM.                               |
| `results/`       | Contains intermediate and final results from various stages of the pipeline.  |
| `src/`           | Main source code for the pipeline.                                            |
| `.env`           | Stores important API keys for RITS and Watson (ensure this is kept private). |

### Source folder

| File Name              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `call_LLM.py`          | Functions that handle calls to the LLM.                                     |
| `data2dashboard.py`    | Main driver file that integrates all functions given an NL query and data.  |
| `Evaluate.py`          | Evaluates the rendered charts.                                              |
| `fetch_data.py`        | Loads the dataframe and generates the data schema.                          |
| `generate_vis.py`      | Processes the Vega-Lite spec and renders the chart.                         |
| `langGraph_agent.py`   | Contains LangGraph agent logic.                                             |
| `LangGraph_mcp.py`     | LangGraph agent with Python functions converted into MCP tools.            |
| `llm_agent_no_cache.py`| Configuration file for the LLM without loading from cache.                 |
| `Main.py`              | Main file for creating and running the LangGraph agent.                     |
| `mcp_tools.py`         | Defines MCP tools from standard Python functions.                          |
| `preprocessing.py`     | Includes basic preprocessing utilities.                                     |
| `run_eval_nl_query.py` | Evaluates NL queries against their respective dataframes.                   |
| `test_agent.py`        | A test agent built using the A2A protocol.                                  |
| `VLM_pixtral.py`       | Defines the Pixtral-based Vision Language Model (VLM) class.                |

### Prompts folder

| File Name                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `remove_redundant_col.txt` | Prompt given to the LLM to identify and remove redundant columns from the dataframe. |
| `visualize_data.txt`       | Prompt given to the LLM to generate a list of visualizations for the dashboard.     |
| `evaluation_5metrics.txt`  | Prompt given to the VLM to evaluate the rendered charts across five metrics.        |

## Running the project

In order to run the pipeline only consisting of the python functions, run:

<pre><code>python3 data-dashboard/src/data2dashboard.py</code></pre>

In order to run the LangGraph agent, run:

<pre><code>python3 data-dashboard/src/LangGraph_agent.py</code></pre>

In order to run the LangGraph agent with mcp tools, run:

<pre><code>python3 data-dashboard/src/LangGraph_mcp.py</code></pre>

in order to run evaluation on all the queries, run:

<pre><code>python3 data-dashboard/src/run_eval_nl_query.py</code></pre>

## Results

We evaluated the system across 30 natural language queries, categorized into generic and specific types. The following table summarizes the performance metrics:

| Query Type | VER (%) | Legality Check (%) | Layout Check (%) | Scale & Ticks (0–5) | Readability (0–5) | Usability (0–5) |
|------------|---------|---------------------|-------------------|----------------------|-------------------|-----------------|
| Generic    | 40      | 100                 | 80                | 4                    | 3                 | 3               |
| Specific   | 50      | 80                  | 60                | 3                    | 2                 | 4               |


## Key Takeaways:

Generic queries yielded more visually clean and readable charts.
Specific queries were better aligned with task-specific needs, leading to higher usability despite complexity in rendering.
Conclusion

This project presents a novel approach to generating dashboards from NL queries using large language models. By combining:
- intelligent data preprocessing,
- reasoning-driven chart generation via CoT prompting, and
- robust post-processing with VLM evaluation,
- the system delivers promising results for real-world observability tasks. While specific queries remain challenging due to context and token limitations, the pipeline shows strong foundational performance.

Future work will focus on improving expressiveness, feedback loops, user interaction, and holistic dashboard storytelling.

## Conclusion

Data2Dashboard presents a robust pipeline for converting natural language queries into insightful, multi-chart dashboards using LLMs. By combining schema-aware preprocessing, reasoning-driven visualization, and semantic evaluation via VLMs, the system automates dashboard generation for observability tasks. While challenges remain with complex queries and data limits, the pipeline demonstrates strong real-world potential and lays the groundwork for more adaptive, user-guided visualization tools.