# src/generate_vis.py
import pandas as pd
import os
import json
from typing import List, Dict, Any, Tuple

# Note: You'll need to install vl_convert: pip install vl-convert-python
try:
    from vl_convert import vegalite_to_png
    VL_CONVERT_AVAILABLE = True
except ImportError:
    print("Warning: vl_convert not available. Chart rendering will be skipped.")
    VL_CONVERT_AVAILABLE = False

def extract_data(df: pd.DataFrame, col_name: str) -> list:
    """
    Extracts the values from a specific column in a DataFrame as a list.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to extract.

    Returns:
        list: A list of values from the specified column.
    """
    if col_name in df.columns:
        return df[col_name].tolist()
    else:
        raise ValueError(f"Column '{col_name}' not found in DataFrame.")

def add_data(json_spec: dict, data: pd.DataFrame) -> list:
    """
    Enhanced data integration with better error handling.
    """
    col_names = []
    success = True

    # Extract required columns from encoding
    if "encoding" in json_spec:
        encoding = json_spec["encoding"]
        
        # Handle x encoding
        if "x" in encoding and "field" in encoding["x"]:
            field_name = encoding["x"]["field"]
            if field_name in data.columns:
                col_names.append(field_name)
            else:
                print(f"Warning: Field '{field_name}' not found in data columns")
                success = False

        # Handle y encoding (can be list or single item)
        if "y" in encoding:
            y_encoding = encoding["y"]
            if isinstance(y_encoding, list):
                for item in y_encoding:
                    if "field" in item and item["field"] in data.columns:
                        col_names.append(item["field"])
                    else:
                        success = False
            else:
                if "field" in y_encoding:
                    if y_encoding["field"] == "count()":
                        # Handle count aggregation
                        json_spec["encoding"]["y"]["aggregate"] = "count"
                        if "field" in json_spec["encoding"]["y"]:
                            del json_spec["encoding"]["y"]["field"]
                    elif y_encoding["field"] in data.columns:
                        col_names.append(y_encoding["field"])
                    else:
                        print(f"Warning: Field '{y_encoding['field']}' not found in data columns")
                        success = False

        # Handle color encoding
        if "color" in encoding and "field" in encoding["color"]:
            field_name = encoding["color"]["field"]
            if field_name == "count()":
                json_spec["encoding"]["color"]["aggregate"] = "count"
                if "field" in json_spec["encoding"]["color"]:
                    del json_spec["encoding"]["color"]["field"]
            elif field_name in data.columns:
                col_names.append(field_name)
            else:
                print(f"Warning: Color field '{field_name}' not found in data columns")
                success = False

        # Handle angle encoding (for pie charts)
        if "angle" in encoding and "field" in encoding["angle"]:
            field_name = encoding["angle"]["field"]
            if field_name in data.columns:
                col_names.append(field_name)
            else:
                print(f"Warning: Angle field '{field_name}' not found in data columns")
                success = False

        # Handle theta encoding (for pie charts)
        if "theta" in encoding and "field" in encoding["theta"]:
            field_name = encoding["theta"]["field"]
            if field_name in data.columns:
                col_names.append(field_name)
            else:
                print(f"Warning: Theta field '{field_name}' not found in data columns")
                success = False

        # Handle r encoding (for pie charts)
        if "r" in encoding and "field" in encoding["r"]:
            field_name = encoding["r"]["field"]
            if field_name in data.columns:
                col_names.append(field_name)
            else:
                print(f"Warning: R field '{field_name}' not found in data columns")
                success = False

    if not success or not col_names:
        return []

    # Create data records
    final_records = []
    for index, row in data.iterrows():
        record = {}
        valid_record = True
        
        for col_name in col_names:
            value = row[col_name]
            if pd.isna(value):
                valid_record = False
                break
            record[col_name] = value
        
        if valid_record:
            final_records.append(record)

    return final_records

def process_spec(spec_list: List[Dict[str, Any]], data: pd.DataFrame, output_dir: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Enhanced spec processing with better error handling.
    """
    final_specs = []
    total = len(spec_list)
    error_count = 0

    print("Processing Vega-Lite specifications...")
    
    for i, spec_item in enumerate(spec_list):
        print(f"Processing spec {i+1}/{total}")
        
        # Extract the Vega-Lite spec
        vegalite_spec = spec_item.get("vegalite_spec", {})
        
        # Add data to the spec
        data_values = add_data(vegalite_spec, data)
        
        if not data_values:
            print(f"Warning: No valid data for spec {i+1}")
            error_count += 1
            continue
        
        # Add data to the spec
        vegalite_spec["data"] = {"values": data_values}
        spec_item["vegalite_spec"] = vegalite_spec
        
        final_specs.append(spec_item)
        print(f"Spec {i+1} processed successfully with {len(data_values)} data points")

    # Save processed specs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    spec_list_path = os.path.join(output_dir, "processed_specs.json")
    with open(spec_list_path, "w") as f:
        json.dump(final_specs, f, indent=2)

    print(f"Processed {len(final_specs)} valid specs out of {total} total specs")
    return final_specs, error_count, total

def create_vegalite_chart(spec_json: Dict[str, Any], index: int, output_dir: str) -> str:
    """
    Enhanced chart rendering with better error handling.
    Fixed to handle the correct parameters.
    """
    if not VL_CONVERT_AVAILABLE:
        print("Chart rendering skipped - vl_convert not available")
        return ""
    
    os.makedirs(output_dir, exist_ok=True)

    # Extract the Vega-Lite spec from the spec_json
    vegalite_spec = spec_json.get("vegalite_spec", {})
    
    # Fix common issues with chart specs
    if "mark" in vegalite_spec:
        # Fix pie chart mark
        if vegalite_spec["mark"] == "pie":
            vegalite_spec["mark"] = "arc"
        
        # Ensure proper mark format
        if isinstance(vegalite_spec["mark"], str):
            vegalite_spec["mark"] = {"type": vegalite_spec["mark"]}
    
    # Create filename from title or fallback
    chart_title = vegalite_spec.get("title", spec_json.get("title", f"chart_{index}"))
    chart_name = chart_title.replace(" ", "_").replace("/", "_").lower()
    output_path = os.path.join(output_dir, f"{chart_name}_{index}.png")

    try:
        # Convert spec to PNG
        png_bytes = vegalite_to_png(vegalite_spec)
        
        with open(output_path, "wb") as f:
            f.write(png_bytes)
            
        print(f"Chart saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Failed to save chart {index}: {e}")
        
        # Try to save the spec for debugging
        debug_path = os.path.join(output_dir, f"debug_spec_{index}.json")
        try:
            with open(debug_path, "w") as f:
                json.dump(vegalite_spec, f, indent=2)
            print(f"Debug spec saved to: {debug_path}")
        except:
            pass
        
        return ""
