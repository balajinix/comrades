from llm_agent_no_cache import LLMAgent
import json
import re
import os

import re
import json

base_prompt_dir = "~/Desktop/Data_Viz_agent/data-dashboard/prompts"

def extract_vegalit_json(llm_output: str):
    """
    Extracts chart_type, reason, and Vega-Lite JSON spec from the LLM output.

    Args:
        llm_output (str): Full LLM output text.

    Returns:
        List[Dict[str, Any]]: List of dicts with 'title', 'type', 'reason', 'vegalite_spec'
    """

    pattern = r"Chart_type:\s*(.*?)\s*Reason:\s*(.*?)\s*Vega-Lite 5 Specification:\s*```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, llm_output, re.DOTALL)

    extracted_specs = []

    for chart_type, reason, raw_json in matches:
        try:
            # Remove JS-style comments if any
            raw_json = re.sub(r'(?m)^\s*//.*$', '', raw_json)

            # Decode escape sequences
            raw_json_clean = bytes(raw_json.strip(), "utf-8").decode("unicode_escape")

            # Parse JSON
            vegalite_spec = json.loads(raw_json_clean)

            # Use title from the spec if available
            title = vegalite_spec.get("title", "")

            extracted_specs.append({
                "title": title,
                "type": chart_type.strip(),
                "reason": reason.strip(),
                "vegalite_spec": vegalite_spec
            })

        except json.JSONDecodeError as e:
            print(f"[Warning] Failed to parse JSON block:\n{e}\nRaw block:\n{raw_json[:200]}...")

    return extracted_specs


def extract_vegalite_specs(llm_output: str):
    """
    Extracts JSON strings following the phrase 'Vega-Lite 5 Specification'.
    It uses brace matching to extract complete JSON blocks.

    Args:
        llm_output (str): The string output from an LLM.

    Returns:
        List[str]: A list of JSON strings (not parsed).
    """
    specs = []
    search_phrase = "Vega-Lite 5 Specification"
    i = 0

    while i < len(llm_output):
        phrase_idx = llm_output.find(search_phrase, i)
        if phrase_idx == -1:
            break

        # Start scanning for '{' after the phrase
        json_start = llm_output.find('{', phrase_idx)

        if json_start == -1:
            break

        stack = ['{']
        j = json_start + 1

        while j < len(llm_output) and stack:
            if llm_output[j] == '{':
                stack.append('{')
            elif llm_output[j] == '}':
                stack.pop()
            j += 1

        if not stack:
            json_block = llm_output[json_start:j]
            specs.append(json_block)
        else:
            print(f"Warning: Unmatched braces after index {phrase_idx}. Skipping incomplete block.")
            break

        i = j  # Move to next position to find the next match

    return specs

def convert_strings_to_json(string_list):
    """
    Takes a list of escaped JSON strings (with `\\n`, `//` comments, etc.)
    and returns a list of valid JSON dictionaries.
    """
    parsed_jsons = []

    for i, raw_str in enumerate(string_list):
        try:
            # Decode escape sequences like \n, \"
            clean_str = bytes(raw_str, "utf-8").decode("unicode_escape")

            # ✅ Remove lines that are pure JS-style comments (not inline URLs)
            clean_str = re.sub(r'(?m)^\s*//.*$', '', clean_str)

            # Parse JSON
            json_obj = json.loads(clean_str)
            parsed_jsons.append(json_obj)

        except json.JSONDecodeError as e:
            print(f"Error parsing string at index {i}: {e}")

    return parsed_jsons

def extract_json_list(text: str) -> list:
    # Use regex to find content inside '''json ... '''
    match = re.search(r"''json\s*(.*?)\s*'''", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            return []
    else:
        print("No JSON block found.")
        return []

def remove_redundant_columns(data_schema, query, output_dir):
    prompt_path = os.path.join(base_prompt_dir,"remove_redundant_col.txt")

    with open(prompt_path) as f:
        prompt_template = f.read()

    schema_str = json.dumps(data_schema, indent=2, default=str)
    prompt = prompt_template.format(query=query, schema=schema_str)

    print("Prompt to remove redundant columns: ")
    print(prompt)
    print("-"*30)
    llm_agent = LLMAgent(service="PROVIDER", model_name="ibm/granite-3-2-8b-instruct", temperature=0.25)
    response = llm_agent.call_llm(prompt)
    print("The LLM response is: ")
    print(response)
    imp_columns = extract_json_list(response)
    print("The important columns according to the LLM are: ")
    print(imp_columns)
    print("-"*30)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, "imp_columns.json"), "w") as f:
        json.dump(imp_columns, f)
    return imp_columns


def visualize_data(report_json, query, output_dir):
    prompt_path = os.path.join(base_prompt_dir, "visualize_data_2.txt")

    with open(prompt_path) as f:
        prompt_template = f.read()

    num = len(report_json)
    prompt = prompt_template.format(question=query, table_context=json.dumps(report_json, default=str, indent=2), n = num)

    print("Prompt to suggest chart type: ")
    print(prompt)
    print("-"*30)
    llm_agent = LLMAgent(service="PROVIDER", model_name="ibm/granite-3-2-8b-instruct", temperature=0.25)
    response = llm_agent.call_llm(prompt)
    print("Response given by the LLM: ")
    print(response)
    spec_list = extract_vegalit_json(response)
    with open(os.path.join(output_dir, "spec_list_no_data.json"), 'w') as f:
        json.dump(spec_list, f)
    return spec_list