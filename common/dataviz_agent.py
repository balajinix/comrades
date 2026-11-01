import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from llm_agent import LLMAgent
import gc
import re
import numpy as np
import sys
import textwrap
from copy import deepcopy
import ast
from io import StringIO

def summarize_table(data, query):
    data_json = data.to_dict(orient='list')
    
    model_name = "ibm/granite-3-2-8b-instruct"
    
    prompt_template = (
        "Understand the data given below, which represents a table, where the key names are column names and "
        "the values are the column values. The data was fetched using the given query. Your task is to generate "
        "a summary using the following instructions.\n\n"
        "1. Generate one line describing the data using the given query.\n"
        "2. Generate one or two sentences highlighting any interesting statistical trends in the data.\n"
        "Do not generate anything else.\n\n"
        "Query:\n{query}\n\n"
        "Data:\n{data_json}"
    )
    
    prompt = prompt_template.format(query=query, data_json=json.dumps(data_json, default=str, indent=2))
    
    llm_agent = LLMAgent(service="PROVIDER", model_name=model_name, temperature=0.25)
    
    response = llm_agent.call_llm(prompt)
    
    return response

def identify_important_columns(data, query="SLI"):
    if data.empty:
        raise ValueError("The input DataFrame is empty.")

    #data_json = data.to_dict(orient='list')
    model_name = "ibm/granite-3-2-8b-instruct"
    
    prompt_template = """
    You are given column names of tabular data. 
    This data was retrieved using the query provided below.

    Your task is to analyze the column names and identify **combinations of column names** that, when visualized together, could reveal meaningful or interesting insights.

    Instructions:
    1. Return the result as a JSON object. Each key should be a string like "combo_1", "combo_2", etc., and the corresponding value should be a list of column names.
    Example:
    {{
        "combo_1": ["column_a", "column_b"],
        "combo_2": ["column_x", "column_y", "column_z"]
    }}
    2. Base your combinations on potential to uncover trends, correlations, clusters, or other useful insights.
    3. Only generate the JSON object. Make sure the JSON is valid and doesn't include extra commentary. Do not include any explanation, helpful message or any other text, such as here is the JSON object.

    Query:
    {query}

    Colums:
    {columns}
    """

    prompt = prompt_template.format(query=query, columns=columns)

    llm_agent = LLMAgent(service="PROVIDER", model_name=model_name, temperature=0.25)

    response = llm_agent.call_llm(prompt)
    return response


def save_plot(output_dir, plot_name):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches='tight')
    except Exception as e:
        print(f"[ERROR] Could not save plot '{plot_name}': {e}")
    finally:
        plt.clf()
        plt.close()

def detect_epoch_column(series):
    """Check if a numeric column is an epoch timestamp by converting and verifying validity."""
    if series.dtype.kind not in 'iuf':
        return None

    now = pd.Timestamp.now()
    min_reasonable_date = pd.Timestamp("2000-01-01")

    def is_within_epoch_range(ts_series):
        ts_series = ts_series.dropna()
        if ts_series.empty:
            return False
        valid = ts_series[(ts_series >= min_reasonable_date) & (ts_series <= now)]
        return len(valid) / len(ts_series) > 0.99

    detected_unit = None
    score = 0

    try:
        ts_seconds = pd.to_datetime(series, unit='s', errors='coerce')
        if is_within_epoch_range(ts_seconds):
            detected_unit = 's'
            score = 1
    except Exception:
        pass

    try:
        ts_milliseconds = pd.to_datetime(series, unit='ms', errors='coerce')
        if is_within_epoch_range(ts_milliseconds):
            # Prefer milliseconds if values are large
            if series.median() > 1e11 or score == 0:
                detected_unit = 'ms'
    except Exception:
        pass

    return detected_unit

def detect_and_plot(data, output_dir, base_name):
    """Detect data type and generate appropriate visualizations."""
    if data is None or data.empty:
        print("No data to visualize or DataFrame is empty.")
        return

    print("Data Summary:")
    try:
        print(data.info())
    except Exception as e:
        print(f"[ERROR] calling data.info(): {e}")

    num_cols = list(data.select_dtypes(include='number').columns)
    cat_cols = list(data.select_dtypes(include='object').columns)
    date_cols = list(data.select_dtypes(include='datetime64').columns)

      # Identify and convert epoch timestamp columns
    epoch_cols = {}  # Dictionary to store epoch columns and their unit (s/ms)
    for col in data.select_dtypes(include=['number']).columns:  # Only iterate over numeric columns
        epoch_type = detect_epoch_column(data[col])
        if epoch_type:
            data[col] = pd.to_datetime(data[col], unit=epoch_type).dt.date  # Convert and extract only date
            epoch_cols[col] = epoch_type

    # Merge date columns and converted timestamp columns into a single time-based column list
    time_cols = list(data.select_dtypes(include=['datetime']).columns) + list(epoch_cols.keys())
                                                                            
 
    # Merge date columns and converted timestamp columns into a single time-based column list
    time_cols = list(date_cols) + list(epoch_cols.keys())


    # Remove detected epoch columns from numerical columns
    num_cols = [col for col in num_cols if col not in epoch_cols]

    total_cols = len(data.columns)
    print('time cols',epoch_cols.keys())
    try:
        if total_cols == 1:
            case_1_single_column(data, output_dir, base_name)
        elif len(time_cols) == 1 and len(num_cols) >= 1 and len(cat_cols)==0:
            case_2_time_series(data, output_dir, base_name, time_cols, num_cols)  # Pass date info
        elif total_cols == 2 and len(cat_cols) == 1 and len(num_cols) == 1:
            case_3_one_cat_one_num(data, output_dir, base_name)
        elif total_cols == 2 and len(cat_cols) == 2:
            case_4_two_categorical(data, output_dir, base_name)
        elif total_cols == 2 and len(num_cols) == 2:
            case_5_two_numerical(data, output_dir, base_name)
        elif len(cat_cols) == 1 and len(num_cols) >= 2 and len(time_cols)==0:
                case_6_one_cat_multi_num(data, output_dir, base_name)
        elif len(cat_cols) >= 2 and len(num_cols) == 1 and len(time_cols)==0:
                case_7_multi_cat_one_num(data, output_dir, base_name)
        elif len(num_cols) >= 3 and len(cat_cols) == 0 and len(time_cols)==0:
                case_8_multi_numerical(data, output_dir, base_name)
        elif len(time_cols) >= 1 and len(cat_cols) == 1 and len(num_cols) >= 2:
            print(f"Case 10: Multiple time column detected. Dropping '{time_cols[0]}' and calling Case 6.")
            data = data.drop(columns=time_cols)  # Drop time column
            case_6_one_cat_multi_num(data, output_dir, base_name)  # Call case 6
        else:
            case_9_mulit_numerical_multi_cat_multi_date(data, output_dir, base_name)
    except Exception as e:
        print(f"[ERROR] in detect_and_plot logic: {e}")

    print("Visualizations generated and saved.")
    del data
    gc.collect()



    print("Data visualization process completed!")

# === Case 1: Single Column ===
def case_1_single_column(data, output_dir, base_name):
    """
    For each column in a single-column dataset:
    - Numerical: Create a histogram.
    - Categorical: Create a bar plot (or lollipop plot if desired).
    """
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            print(f"Case 1: Plotting histogram for numerical column '{col}'")
            try:
                plt.figure(figsize=(8, 6))
                sns.histplot(data[col].dropna(), kde=True)
                plt.title(f'Histogram of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                save_plot(output_dir, f'{base_name}_{col}_histogram.png')
            except Exception as e:
                print(f"[ERROR] in case_1 histogram for column '{col}': {e}")
                plt.close()
            finally:
                gc.collect()

        elif pd.api.types.is_object_dtype(data[col]):
            print(f"Case 1: Plotting bar plot for categorical column '{col}'")
            try:
                plt.figure(figsize=(10, 6))
                counts = data[col].value_counts()
                sns.barplot(x=counts.index, y=counts.values)
                plt.title(f'Bar Plot of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                save_plot(output_dir, f'{base_name}_{col}_barplot.png')
            except Exception as e:
                print(f"[ERROR] in case_1 barplot for column '{col}': {e}")
                plt.close()
            finally:
                gc.collect()


def case_2_time_series(data, output_dir, base_name, time_cols, num_cols):
    """
    Handles time-series data visualization.
    `date_cols`: List of datetime columns.
    `epoch_cols`: Dictionary of detected epoch timestamp columns and their units (s/ms).
    """
    print("Inside case 2 timeseries")
    print(num_cols)
    try:
        if len(time_cols) >= 1 and len(num_cols) >= 1:
            date_col = time_cols[0]  # Use the identified date column
        
            print(f"Case 2: Using date column '{date_col}'")

            if len(num_cols) == 1:
                num_col = num_cols[0]
                print(f"Case 2: Plotting area chart for '{num_col}' over '{date_col}'")
                try:
                    plt.figure(figsize=(10, 6))
                    plt.fill_between(data[date_col].dropna(), data[num_col].dropna(), alpha=0.4, color='blue')
                    plt.plot(data[date_col], data[num_col], marker='o', color='blue')
                    plt.title(f'Area Chart of {num_col} over {date_col}')
                    plt.xlabel(date_col)
                    plt.ylabel(num_col)
                    plt.xticks(rotation=45)
                    save_plot(output_dir, f'{base_name}_{num_col}_area_chart.png')
                except Exception as e:
                    print(f"[ERROR] in case_2 area chart: {e}")
                    plt.close()
                finally:
                    gc.collect()
            else:
                print("Case 2: Plotting multiple series with line plot and box plot.")
                try:
                    plt.figure(figsize=(12, 8))
                    for col in num_cols:
                        sns.lineplot(x=date_col, y=col, data=data, label=col)
                    plt.title('Line Plot for Multiple Numerical Series')
                    plt.xlabel(date_col)
                    plt.ylabel('Values')
                    plt.legend()
                    plt.xticks(rotation=45)
                    save_plot(output_dir, f'{base_name}_multiple_line_plot.png')
                except Exception as e:
                    print(f"[ERROR] in case_2 multiple series: {e}")
                    plt.close()
                finally:
                    gc.collect()

                try:
                    plt.figure(figsize=(12, 8))
                    melted_data = data.melt(id_vars=date_col, value_vars=num_cols, var_name='Series', value_name='Value')
                    sns.boxplot(x='Series', y='Value', data=melted_data)
                    plt.title('Box Plot for Multiple Numerical Series')
                    plt.xlabel('Series')
                    plt.ylabel('Values')
                    plt.xticks(rotation=45)
                    save_plot(output_dir, f'{base_name}_multiple_box_plot.png')
                except Exception as e:
                    print(f"[ERROR] in case_2 multiple series: {e}")
                    plt.close()
                finally:
                    gc.collect()
    except Exception as e:
        print(f"[ERROR] in case_2_time_series: {e}")
        plt.close()

# === Case 3: One Categorical, One Numerical ===
def case_3_one_cat_one_num(data, output_dir, base_name):
    """
    When exactly one categorical and one numerical column exist:
    - If there is one observation per category, use a bar plot.
      If there is high variation in bar heights, also show a pie chart.
    - Otherwise, use a box plot (or violin plot) to display distributions.
    """
    try:
        cat_cols = data.select_dtypes(include='object').columns
        num_cols = data.select_dtypes(include='number').columns

        if len(cat_cols) == 1 and len(num_cols) == 1:
            cat_col = cat_cols[0]
            num_col = num_cols[0]
            print(f"Case 3: Plotting relationship between '{cat_col}' and '{num_col}'")

            group_sizes = data.groupby(cat_col).size()

            try:
                if group_sizes.max() == 1:
                    # Prepare data as a proper DataFrame for Seaborn
                    bar_data = data[[cat_col, num_col]].set_index(cat_col).sort_index().reset_index()

                    # Create consistent color map
                    categories = bar_data[cat_col]
                    palette = sns.color_palette("tab20", len(categories))
                    color_map = dict(zip(categories, palette))

                    # Plot using DataFrame reference to fix label alignment
                    plt.figure(figsize=(14, 7))
                    sns.barplot(
                        data=bar_data,
                        x=cat_col,
                        y=num_col,
                        palette=color_map
                    )
                    plt.title(f'Bar Plot of {num_col} by {cat_col}')
                    plt.xlabel(cat_col)
                    plt.ylabel(num_col)
                    plt.xticks(rotation=60, ha='right')  # Helps prevent label overlap
                    plt.tight_layout()
                    save_plot(output_dir, f'{base_name}_{cat_col}_{num_col}_bar.png')
                    plt.close()


                    # Check for high variation in bar heights
                    bar_series = bar_data.set_index(cat_col)[num_col]
                    cv = bar_series.std() / bar_series.mean() if bar_series.mean() != 0 else 0

                    if cv > 1.0:
                        # High variation: also show pie chart
                        categories_sorted = bar_series.index
                        values_sorted = bar_series.values
                        colors_sorted = [color_map[cat] for cat in categories_sorted]

                        plt.figure(figsize=(10, 8))
                        if len(bar_series) > 8:
                            wedges, texts = plt.pie(
                                values_sorted,
                                startangle=140,
                                colors=colors_sorted
                            )
                            plt.legend(wedges, categories_sorted, title=cat_col, bbox_to_anchor=(1, 0.5), loc="center left", fontsize='small')
                        else:
                            def autopct_format(pct):
                                return f'{pct:.0f}%' if pct > 5 else ''

                            wedges, texts, autotexts = plt.pie(
                                values_sorted,
                                labels=None,  # Don't show labels directly on the pie
                                autopct=autopct_format,
                                startangle=140,
                                textprops={'fontsize': 10},
                                pctdistance=0.85,
                                colors=colors_sorted
                            )
                        plt.title(f'Pie Chart of {num_col} by {cat_col}', fontsize=14)
                        plt.tight_layout()
                        save_plot(output_dir, f'{base_name}_{cat_col}_{num_col}_pie.png')
                        plt.close()

                else:
                    # Multiple observations per group — use box plot
                   

                    # Wrap long labels for better display
                    data[cat_col] = data[cat_col].apply(lambda x: textwrap.fill(str(x), width=15))

                    plt.figure(figsize=(14, 8))
                    sns.violinplot(x=cat_col, y=num_col, data=data, inner='box')  # 'inner=box' shows a mini boxplot inside the violin
                    plt.title(f'Violin Plot of {num_col} by {cat_col}')
                    plt.xlabel(cat_col)
                    plt.ylabel(num_col)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    save_plot(output_dir, f'{base_name}_{cat_col}_{num_col}_violin.png')
                    plt.close()

            except Exception as e:
                print(f"[ERROR] in case_3_one_cat_one_num: {e}")
                plt.close()
            finally:
                gc.collect()

    except Exception as e:
        print(f"[ERROR] in case_3_one_cat_one_num: {e}")
        plt.close()

# === Case 4: Two Categorical Columns ===
def case_4_two_categorical(data, output_dir, base_name):
    """
    For two categorical columns, create a heatmap or count plot of their relationship.
    """
    try:
        cat_cols = data.select_dtypes(include='object').columns
        if len(cat_cols) == 2:
            print(f"Case 4: Plotting heatmap for categorical columns '{cat_cols[0]}' and '{cat_cols[1]}'")
            try:
                plt.figure(figsize=(12, 8))
                cross_tab = pd.crosstab(data[cat_cols[0]], data[cat_cols[1]])
                sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='d')
                plt.title(f'Heatmap of {cat_cols[0]} vs {cat_cols[1]}')
                plt.xlabel(cat_cols[1])
                plt.ylabel(cat_cols[0])
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                save_plot(output_dir, f'{base_name}_{cat_cols[0]}_{cat_cols[1]}_heatmap.png')
            except Exception as e:
                print(f"[ERROR] in case_4_two_categorical: {e}")
                plt.close()
            finally:
                gc.collect()
    except Exception as e:
        print(f"[ERROR] in case_4_two_categorical: {e}")
        plt.close()


# === Case 5: Two Numerical Columns ===
def case_5_two_numerical(data, output_dir, base_name):
    """
    For two numerical columns:
    - If scales differ significantly, create separate visualizations using Case 1.
    - Otherwise, use a scatter plot to show their relationship.
    """
    try:
        num_cols = data.select_dtypes(include='number').columns
        if len(num_cols) == 2:
            col1, col2 = num_cols[0], num_cols[1]
            range1 = data[col1].max() - data[col1].min()
            range2 = data[col2].max() - data[col2].min()

            # Check for scale difference
            if max(range1, range2) > 2 * min(range1, range2):
                print(f"Case 5: Significant scale difference detected. Creating separate histograms for '{col1}' and '{col2}'.")

                # Histogram for Column 1
                try:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data[col1].dropna(), kde=True)
                    plt.title(f'Histogram of {col1}')
                    plt.xlabel(col1)
                    plt.ylabel('Frequency')
                    save_plot(output_dir, f'{base_name}_{col1}_histogram.png')
                except Exception as e:
                    plt.close()
                finally:
                    gc.collect()

                # Histogram for Column 2
                try:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data[col2].dropna(), kde=True)
                    plt.title(f'Histogram of {col2}')
                    plt.xlabel(col2)
                    plt.ylabel('Frequency')
                    save_plot(output_dir, f'{base_name}_{col2}_histogram.png')
                except Exception as e:
                    plt.close()
                finally:
                    gc.collect()
            
            else:
                print(f"Case 5: Similar scales detected. Plotting scatter plot for '{col1}' and '{col2}'.")
                
                # Scatter Plot
                try:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=col1, y=col2, data=data)
                    plt.title(f'Scatter Plot of {col1} vs {col2}')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    save_plot(output_dir, f'{base_name}_{col1}_{col2}_scatter.png')
                except Exception as e:
                    plt.close()
                finally:
                    gc.collect()
    except Exception as e:
        print(f"[ERROR] in case_5_two_numerical: {e}")
        plt.close()


# === Case 6: Three or More Columns (One Categorical, Rest Numerical) ===
def case_6_one_cat_multi_num(data, output_dir, base_name):
    """
    For datasets with three or more columns with one categorical and at least two numerical columns:
    - If the categorical column has one observation per group, generate a grouped bar chart.
    - Else, if one numerical column is ordered, create an area plot or line plot for all numerical columns.
    - Otherwise, create a box plot, violin plot, or grouped scatter plot.
    """
    try:
        cat_cols = data.select_dtypes(include='object').columns
        num_cols = data.select_dtypes(include='number').columns
        data = data.loc[~(data[num_cols] == 0).all(axis=1)]
        
        if len(cat_cols) == 1 and len(num_cols) >= 2:
            cat_col = cat_cols[0]
            data = data.sort_values(by=cat_col)
            print(f"Case 6: Processing dataset with categorical '{cat_col}' and numerical columns {list(num_cols)}")
            
          
            if data[cat_col].nunique() == len(data):
                print(" - Unique values in categorical column detected. Using grouped bar chart.")
                try:
                    # Ensure only valid numerical columns are used
                    valid_num_cols = [col for col in num_cols if col in data.columns]

                    if not valid_num_cols:
                        print("[ERROR] No valid numerical columns found for bar plot.")
                        return

                    plt.figure(figsize=(14, 8))
                    melted = data.melt(id_vars=[cat_col], value_vars=valid_num_cols)
                    sns.barplot(x=cat_col, y='value', hue='variable', data=melted, dodge=True)
                    plt.title(f'Grouped Bar Plot for {cat_col}')
                    plt.xlabel(cat_col)
                    plt.ylabel('Value')
                    plt.xticks(rotation=45, ha='right')
                    plt.legend(title="Numerical Columns", bbox_to_anchor=(1.05, 1), loc='upper left')

                    plt.tight_layout()
                    save_plot(output_dir, f'{base_name}_{cat_col}_grouped_bar.png')
                    plt.close()
                except Exception as e:
                    print(f"[ERROR] in grouped bar plot: {e}")
                    plt.close()
                finally:
                    gc.collect()
                
                # Pie Charts for Each Numerical Column
                try:
                    for num_col in valid_num_cols:
                        if num_col not in data.columns:
                            print(f"[ERROR] Column not found: {num_col}, skipping pie chart.")
                            continue

                        summed_data = data.groupby(cat_col)[num_col].sum()
                        cv = summed_data.std() / summed_data.mean() if summed_data.mean() != 0 else 0

                        if cv > 1.0:
                            print(f" - High variation detected in {num_col}, generating pie chart.")
                            plt.figure(figsize=(10, 8))

                            if len(summed_data) > 8:
                                wedges, texts = plt.pie(summed_data, startangle=140)
                                plt.legend(summed_data.index, title=cat_col, bbox_to_anchor=(1, 0.5), loc="center left", fontsize='small')
                            else:
                                def autopct_format(pct):
                                    return f'{pct:.0f}%' if pct > 5 else ''  # Hide small values

                                wedges, texts, autotexts = plt.pie(
                                    summed_data,
                                    autopct=autopct_format,
                                    startangle=140,
                                    textprops={'fontsize': 10},
                                    pctdistance=0.85  # Push percentages outward
                                )
                                plt.legend(summed_data.index, title=cat_col, bbox_to_anchor=(1, 0.5), loc="center left", fontsize='small')

                            plt.title(f'Pie Chart for {num_col}', fontsize=14)
                            plt.tight_layout()
                            save_plot(output_dir, f'{base_name}_{num_col}_pie_chart.png')
                            plt.close()
                        else:
                            print(f" - Skipping pie chart for {num_col}, not enough variation (CV={cv:.2f})")
                except Exception as e:
                    print(f"[ERROR] in case_6 pie chart for {num_col}: {e}")
            

            elif any(data[col].dropna().is_monotonic_increasing or data[col].dropna().is_monotonic_decreasing for col in num_cols):
                ordered_col = next(
                    (col for col in num_cols if data[col].dropna().is_monotonic_increasing or data[col].dropna().is_monotonic_decreasing),
                    None
                )
                if ordered_col:
                    print(f" - Ordered numerical column detected: {ordered_col}. Using it as the x-axis for line plots.")
                    try:
                        plt.figure(figsize=(12, 8))
                        for col in num_cols:
                            if col != ordered_col:
                                sns.lineplot(x=ordered_col, y=col, data=data, marker='o', label=col)
                        plt.title(f'Line Plot of Numerical Columns vs {ordered_col}')
                        plt.xlabel(ordered_col)
                        plt.ylabel('Values')
                        plt.legend()
                        plt.xticks(rotation=45)
                        save_plot(output_dir, f'{base_name}_{ordered_col}_ordered_lineplot.png')
                    except Exception as e:
                        print(f"[ERROR] in ordered numerical line plot: {e}")
                        plt.close()
                    finally:
                        gc.collect()

            else:
                print(" - No ordered numerical column found. Creating box plot, violin plot, and grouped scatter plot.")
                melted = data.melt(id_vars=cat_col, value_vars=num_cols)
                try:
                    plt.figure(figsize=(12, 8))
                    sns.violinplot(x='variable', y='value', hue=cat_col, data=melted, inner='box')
                    plt.title(f'Violin Plot for Numerical Columns Grouped by {cat_col}')
                    plt.xlabel('Numerical Columns')
                    plt.ylabel('Values')
                    plt.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45)
                    plt.subplots_adjust(right=0.7)
                    save_plot(output_dir, f'{base_name}_{cat_col}_violin_plot.png')
                except Exception as e:
                    plt.close()
                finally:
                    gc.collect()
    except Exception as e:
        print(f"[ERROR] in case_6_one_cat_multi_num: {e}")
        plt.close()

# === Case 7: Three or More Columns (Several Categorical, One Numerical) ===
def case_7_multi_cat_one_num(data, output_dir, base_name):
    """
    For datasets with several categorical columns and one numerical column:
    - If there is one observation per group, generate a bar plot.
    - Otherwise, use a box plot.
    """
    try:
        cat_cols = data.select_dtypes(include='object').columns
        num_cols = data.select_dtypes(include='number').columns

        if len(cat_cols) >= 2 and len(num_cols) == 1:
            cat1, cat2 = cat_cols[0], cat_cols[1]
            num_col = num_cols[0]
            try:
                print(f"Case 7: Plotting for categorical columns '{cat1}', '{cat2}' and numerical '{num_col}'")
                if data.groupby([cat1, cat2]).size().max() == 1:
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x=cat1, y=num_col, hue=cat2, data=data)
                    plt.title(f'Bar Plot of {num_col} by {cat1} and {cat2}')
                else:
                    plt.figure(figsize=(12, 8))
                    sns.boxplot(x=cat1, y=num_col, hue=cat2, data=data)
                    plt.title(f'Box Plot of {num_col} by {cat1} and {cat2}')
                plt.xlabel(cat1)
                plt.ylabel(num_col)
                plt.xticks(rotation=45)
                save_plot(output_dir, f'{base_name}_{cat1}_{cat2}_{num_col}_comparison.png')
            except Exception as e:
                plt.close()
            finally:
                gc.collect()
    except Exception as e:
        print(f"[ERROR] in case_7_multi_cat_one_num: {e}")
        plt.close()


# === Case 8: Three or More Numerical Columns ===
def case_8_multi_numerical(data, output_dir, base_name):
    """
    For datasets with three or more numerical columns:
    - If one or multiple columns are monotonically increasing, use a line plot or stacked area plot.
    - If data is unordered, choose between a box plot or a violin plot.
    """
    num_cols = data.select_dtypes(include='number').columns
    if len(num_cols) < 3:
        return
    print(f"Case 8: Processing dataset with numerical columns: {list(num_cols)}")

    try:
        ordered_cols = [col for col in num_cols if data[col].dropna().is_monotonic_increasing]
         # Create a safe version of the column names to include in the filename
        col_str = '_'.join(num_cols)
        col_str_safe = re.sub(r'\W+', '_', col_str)  # Replace non-alphanumeric chars with underscores
        col_str_safe = col_str_safe[ :50]
        if ordered_cols:
            print(f" - Ordered numerical columns detected: {ordered_cols}. Using line plot and stacked area plot.")
            try:
                # Line Plot
                plt.figure(figsize=(12, 8))
                for col in num_cols:
                    try:
                        sns.lineplot(data=data, x=data.index, y=col, marker='o', label=col)
                    except Exception as e:
                        print(f"[ERROR] in lineplot for {col}: {e}")
                plt.title('Line Plot for Ordered Numerical Columns')
                plt.xlabel('Index')
                plt.ylabel('Values')
                plt.legend()
               
                filename = f'{base_name}_lineplot_{col_str_safe}.png'
                save_plot(output_dir, filename)
            except Exception as e:
                print(f"[ERROR] in case_8 line plot: {e}")
                plt.close()
            finally:
                gc.collect()

            # Stacked Area
            try:
                plt.figure(figsize=(12, 8))
                # We do a try, if the shapes are mismatched, we skip
                array_list = []
                for col in num_cols:
                    arr = data[col].dropna().values
                    array_list.append(arr)

                plt.stackplot(data.index, array_list, labels=num_cols, alpha=0.6)
                plt.title('Stacked Area Plot for Ordered Numerical Columns')
                plt.xlabel('Index')
                plt.ylabel('Values')
                plt.legend(loc='upper left')
                filename = f'{base_name}_stacked_area_{col_str_safe}.png'
                save_plot(output_dir, filename)

            except Exception as e:
                print(f"[ERROR] in stacked area plot: {e}")
                plt.close()
            finally:
                gc.collect()

        else:
            print(" - No ordered numerical columns detected. Creating box plot and violin plot.")
            # try:
            #     # Box Plot
            #     plt.figure(figsize=(12, 8))
            #     sns.boxplot(data=data[num_cols])
            #     plt.title('Box Plot for Multiple Numerical Columns')
            #     plt.xlabel('Numerical Columns')
            #     plt.ylabel('Values')
            #     plt.xticks(rotation=45)
            #     filename = f'{base_name}_boxplot_{col_str_safe}.png'
            #     save_plot(output_dir,filename)
            # except Exception as e:
            #     print(f"[ERROR] in case_8 box plot: {e}")
            #     plt.close()

            # Violin Plot
            try:
                plt.figure(figsize=(12, 8))
                sns.violinplot(data=data[num_cols])
                plt.title('Violin Plot for Multiple Numerical Columns')
                plt.xlabel('Numerical Columns')
                plt.ylabel('Values')
                plt.xticks(rotation=45)
                filename = f'{base_name}_violinplot_{col_str_safe}.png'
                save_plot(output_dir, filename)
            except Exception as e:
                print(f"[ERROR] in case_8 violin plot: {e}")
                plt.close()
            finally:
                gc.collect()

    except Exception as e:
        print(f"[ERROR] in case_8_multi_numerical: {e}")
        plt.close()



# === Case 9: Multi numerical, categorical and time columns ====    
def case_9_mulit_numerical_multi_cat_multi_date(data, output_dir, base_name):
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_axis_off()

        # Define max width for wrapping text
        max_width = 12

        # Wrap column headers and values
        wrapped_headers = ['\n'.join(textwrap.wrap(str(col), max_width)) for col in data.columns]
        wrapped_values = data.applymap(lambda x: '\n'.join(textwrap.wrap(str(x), max_width)))

        # Create the table
        table = plt.table(
            cellText=wrapped_values.values,
            colLabels=wrapped_headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.5, 1.5)

        # Style table cells
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
            cell.set_fontsize(9)
            if i == 0:
                cell.set_facecolor("#f2f2f2")
                cell.set_fontsize(11)
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor("white")

        # Create a unique suffix based on column names
        col_suffix = '_'.join(data.columns)
        # Sanitize filename (remove or replace unsafe characters)
        col_suffix = re.sub(r'[^a-zA-Z0-9_]+', '_', col_suffix)
        col_suffix = col_suffix[:50]
        # Build filename
        filename = f'{base_name}_case9_{col_suffix}.png'
        

        # Save the image
        save_plot(output_dir, filename)

        gc.collect()
    except Exception as e:
        print(f"[ERROR] in case_9_multi_catnumdate: {e}")
        plt.close()




def cross_join(left, right):
    new_rows = [] if right else left
    for left_row in left:
        for right_row in right:
            temp_row = deepcopy(left_row)
            for key, value in right_row.items():
                temp_row[key] = value
            new_rows.append(deepcopy(temp_row))
    return new_rows


def flatten_list(data):
    for elem in data:
        if isinstance(elem, list):
            yield from flatten_list(elem)
        else:
            yield elem


def json_to_dataframe(data_in):
    def flatten_json(data, prev_heading=''):
        if isinstance(data, dict):
            rows = [{}]
            for key, value in data.items():
                rows = cross_join(rows, flatten_json(value, prev_heading + '.' + key))
        elif isinstance(data, list):
            rows = []
            for item in data:
                [rows.append(elem) for elem in flatten_list(flatten_json(item, prev_heading))]
        else:
            rows = [{prev_heading[1:]: data}]
        return rows
    
    return pd.DataFrame(flatten_json(data_in))


def convert_json_to_csv_string(json_data, max_rows=500):
    """
    Converts json_data directly into a CSV string without reading/writing files.
    
    Args:
        json_data (dict): JSON data to convert (typically from GraphQL response)
        max_rows (int): Maximum number of rows to include in the CSV
        
    Returns:
        tuple: (csv_string, row_count) - The CSV as string and number of rows
    """
    # Extract data object if it exists
    if isinstance(json_data, dict) and "data" in json_data:
        data_obj = json_data.get("data", {})
        
        # Get the first key in the 'data' dict
        data_key = next(iter(data_obj), None)
        if data_key:
            data = data_obj[data_key]
        else:
            data = data_obj
    else:
        # If not in expected format, use as is
        data = json_data
    
    # Convert to dataframe using existing function
    df = json_to_dataframe(data)
    
    # Apply row limit
    df = df[:max_rows]
    
    # Convert to CSV string instead of writing to file
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_string = csv_buffer.getvalue()
    
    # Get row count for reporting
    row_count = len(df)
    
    # Clean up
    del df
    gc.collect()
    
    return csv_string, row_count


def json_to_csv(input_json_file, output_csv_file, max_rows=500):
    """
    Reads 'input_json_file', flattens it *generically* into a list of rows
    by exploding all arrays. Then writes up to max_rows lines to CSV.
    """
    with open(input_json_file, "r", encoding="utf-8") as f:
        json_obj = json.load(f)

    # Extract NL_query
    nl_query = json_obj.get("NL_query", "")
    # Access 'data' object
    data_obj = json_obj.get("data", {})

    # Get the first key in the 'data' dict
    data_key = next(iter(data_obj), None)
    data = data_obj[data_key]
    # print(data)
    df = json_to_dataframe(data)
    df = df[:max_rows]
    df.to_csv(output_csv_file, index=False, encoding='utf-8')
    print(f"CSV file created at: {output_csv_file}, containing {len(df)} rows (limit {max_rows}).")

    del df
    gc.collect()
    return nl_query

def read_data(file_path):
    """Read CSV data into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)
        detect_datetime_columns(df)
        return df
    except Exception as e:
        print(f"[ERROR] reading file '{file_path}': {e}")
        return None
    
def detect_datetime_columns(df):
    for col in df.columns:
        try:
            df[col] = df[col].apply(lambda x: parse(x) if isinstance(x, str) else x)
        except Exception:
            pass  # Skip columns that cannot be parsed

def find_non_numeric_columns(df):
    """Find non-numeric columns in a DataFrame."""
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return non_numeric_cols

def check_one_to_one_mapping(df, col1, col2):
    """
    Checks if two columns have a 1:1 mapping.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.

    Returns:
        bool: True if there is a 1:1 mapping, False otherwise.
    """
    # Check if the number of unique values is the same
    unique_col1 = df[col1].nunique()
    unique_col2 = df[col2].nunique()
    
    # Check if each value in col1 maps to exactly one unique value in col2
    mapping_col1 = df.groupby(col1)[col2].nunique().max() == 1
    mapping_col2 = df.groupby(col2)[col1].nunique().max() == 1

    return unique_col1 == unique_col2 and mapping_col1 and mapping_col2

def choose_better_column(df, col1, col2):
    """
    Chooses the more meaningful column based on:
    1. Fewer missing values
    2. More readable column name (prefers non-ID-like names)
    3. Shorter average string length (assuming simpler labels are better)
    
    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        col1 (str): First column name.
        col2 (str): Second column name.

    Returns:
        str: Name of the better column.
    """
    # Count missing values
    missing_col1 = df[col1].isnull().sum()
    missing_col2 = df[col2].isnull().sum()

    if missing_col1 < missing_col2:
        return col1
    elif missing_col2 < missing_col1:
        return col2

    # Prefer non-ID-like column names
    id_like_keywords = ["id", "code", "identifier", "number"]
    if any(keyword in col1.lower() for keyword in id_like_keywords) and not any(keyword in col2.lower() for keyword in id_like_keywords):
        return col2
    if any(keyword in col2.lower() for keyword in id_like_keywords) and not any(keyword in col1.lower() for keyword in id_like_keywords):
        return col1

    # Prefer the column with shorter average string length (simpler labels)
    avg_len_col1 = df[col1].dropna().astype(str).apply(len).mean()
    avg_len_col2 = df[col2].dropna().astype(str).apply(len).mean()

    return col1 if avg_len_col1 < avg_len_col2 else col2
    

def drop_columns(data):
    """Drop columns that are not important."""
    # Remove constant columns (columns with only one unique value)
    data = data.loc[:, data.nunique() > 1]
    
    # Drop columns with all NaN values
    data = data.dropna(axis=1, how='all')
    
    # Drop rows with any NaN values
    data = data.dropna(axis=0, how='any')
    
    # Drop duplicate rows
    #data = data.drop_duplicates()

    # Find 1:1 mapping of ID-like columns with other non-numeric cols and keep the better column
    non_numeric_cols = find_non_numeric_columns(data)
    id_like_keywords = ["id", "code", "identifier", "number"]

    # Ensure id_like_cols is always initialized
    id_like_cols = []  
    nonid_like_cols = []

    if len(non_numeric_cols) > 1:
        for col in non_numeric_cols:
            if any(keyword in col.lower() for keyword in id_like_keywords):
                id_like_cols.append(col)
            else:
                nonid_like_cols.append(col)

    if len(id_like_cols) > 0 and len(nonid_like_cols) > 0:
        temp = id_like_cols.copy()  # Make a copy to modify inside loop
        for col1 in id_like_cols:
            for col2 in nonid_like_cols:
                if check_one_to_one_mapping(data, col1, col2):
                    better_column = choose_better_column(data, col1, col2)
                    data = data.drop(columns=[col1 if better_column == col2 else col2])
                    temp.remove(col1)
                    break

        id_like_cols = []  # Reinitialize after processing
        for col1 in temp:
            for col2 in temp:
                if col1 != col2 and col2 not in id_like_cols and col1 not in id_like_cols:
                    if check_one_to_one_mapping(data, col1, col2):
                        better_column = choose_better_column(data, col1, col2)
                        id_like_cols.append(better_column)
                        break

        # Drop remaining columns that were not selected
        for col in list(set(temp) - set(id_like_cols)):
            data = data.drop(columns=col)

    return data

def process_data_viz(input_json, output_csv, output_dir):
    """
    End-to-end function:
    1. Converts JSON to CSV.
    2. Reads CSV into a DataFrame.
    3. Drops columns that are not important
    4. Generates and saves visualizations.
    """
    print("Step 1: Converting JSON to CSV...")
    try:
        nl_query = json_to_csv(input_json, output_csv)
    except Exception as e:
        print(f"[ERROR] while converting JSON to CSV: {e}")
        return

    print("Step 2: Reading CSV file...")
    data = read_data(output_csv)
    if data is None:
        print("[ERROR] reading CSV")
        return

    print("Step 3: Keep important columns and rows...")
    print('Drop columns')
    data = drop_columns(data) 
    print("Dropping rows with missing values")
    data = data.replace("", None).dropna() # drop rows containining None
    #data = data.drop_duplicates() # drop duplicate rows
    data = drop_columns(data) 
    print("Step 4: Generating visualizations...")
    base_name = os.path.splitext(os.path.basename(input_json))[0]  # e.g. "step_5"
    #data.to_csv(output_dir+'/' +base_name+'.csv', index=False)

    print("Step 4: Generating Summary")
    summary = summarize_table(data, query=nl_query)
    summary_file =  base_name + "_summary.txt"
    summary_path = os.path.join(output_dir, summary_file)
    with open(summary_path, "w") as file:
        file.write(summary)
    if not data.empty:
        print("Step 4: Finding important column combination")
        column_string = ", ".join(data.columns)
        response = identify_important_columns(column_string, query=nl_query)
        match = re.search(r'{[\s\S]*}', response)
        combos = {}
        if match:
            json_str = match.group(0)
            try:
                combos = json.loads(json_str)
                print("Parsed JSON:", combos)
            except json.JSONDecodeError as e:
                print("JSON parsing failed:", e)
        else:
            print("No JSON object found in the response.")


        print("Step 5: Generating visualizations...")
        for name, columns in combos.items():
            print(f"{name}: {columns}")
            try:
                filtered_data = data[columns].copy()
                detect_and_plot(filtered_data, output_dir, base_name)
                gc.collect()
            except KeyError as e:
                print(f"Columns not found in DataFrame: {e}")
        if not combos:
            try:
                detect_and_plot(data, output_dir, base_name)
                gc.collect()
            except KeyError as e:
                print(f"Columns not found in DataFrame: {e}")
        print("Data visualization process completed!")
    else:
        print("No data to visualize")
    

def list_json_files(directory, recursive=False):
    json_files = []
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".json"):
                    json_files.append(os.path.join(root, file))
    else:
        if os.path.isfile(directory) and directory.lower().endswith(".json"):
            # If the user gave a single JSON file instead of a directory
            json_files.append(directory)
        elif os.path.isdir(directory):
            for file in os.listdir(directory):
                if file.lower().endswith(".json"):
                    json_files.append(os.path.join(directory, file))
    return json_files

def process_multiple_jsons(json_files, csv_files_dir, output_dir, user_query=None):
    """
    Process multiple JSON files, each producing its own CSV and charts.
    All outputs go into a single directory named by a timestamp.
    
    Args:
        json_files (List[str]): A list of paths to JSON files to process.
        output_dir (str): The root directory under which we create a timestamped folder.
        user_query (str, optional): The user query that triggered these visualizations.

    Returns:
        str: The path to the timestamped output directory.
    """
    os.makedirs(csv_files_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for i, json_file in enumerate(json_files, start=1):
        print(f"\n=== Processing JSON file #{i}: {json_file} ===")

        # Use the base name of the JSON file for CSV file name
        json_basename = os.path.splitext(os.path.basename(json_file))[0]  # e.g. "step_5"
        csv_name = json_basename + ".csv"  # e.g. "step_5.csv"
        csv_path = os.path.join(csv_files_dir, csv_name)

        try:
            process_data_viz(json_file, csv_path, output_dir)
        except Exception as e:
            print(f"[ERROR] processing JSON '{json_file}': {e}")
        finally:
            gc.collect()

    print(f"\nAll JSON files processed. Charts and CSVs saved in: {output_dir}")
    return output_dir


# python dataviz_agent.py --json data.json --csv output.csv --output viz_output/
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Agent for Data Visualization")
    parser.add_argument("--json", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output visualization directory")
    args = parser.parse_args()

    
    json_files = list_json_files(args.json, recursive=False)
    if not json_files:
        print(f"No JSON files found in '{args.json}'. Exiting.")
        sys.exit(0)

    process_multiple_jsons(
        json_files=json_files,
        csv_files_dir=args.csv,
        output_dir=args.output,
        user_query=None
    )

def test_cols():
    def generate_sample_dataframe():
        data = pd.DataFrame({
            "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
            "product": ["A", "A", "B", "B", "C", "C", "A", "B"],
            "sales": [1200, 850, 900, 1100, 1500, 700, 1300, 950],
            "profit": [300, 150, 200, 250, 400, 100, 350, 220],
            "quarter": ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4"]
        })
        return data

    data = generate_sample_dataframe()
    column_string = ", ".join(data.columns)
    response = identify_important_columns(column_string, query="Product sales and profit by region and quarter")
    print(response)
    # Use regular expression to extract the JSON part from the string
    match = re.search(r'{[\s\S]*}', response)

    if match:
        json_str = match.group(0)
        try:
            combos = json.loads(json_str)
            print("Parsed JSON:", combos)
        except json.JSONDecodeError as e:
            print("JSON parsing failed:", e)
    else:
        print("No JSON object found in the response.")
