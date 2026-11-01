import logging
import numpy as np
from mistralai import Mistral
import json
import base64
import csv
import pandas as pd
import mimetypes
import base64

def encode_image_as_data_uri(image_path):
    # Guess MIME type
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"  # Default fallback

    # Encode image
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Return data URI
    return f"data:{mime_type};base64,{image_base64}"


logger = logging.getLogger(__name__)

class MistralChat:
    def __init__(
        self,
        api_key="YOUR_MISTRAL_API_KEY",  # Replace with your Mistral AI API key
        model_name="pixtral-12b-2409",  # Specify the model you want to use
        max_tokens=250,
        temperature=0.0,
        top_p=1.0,
    ):
        # Initialize Mistral API client
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


    def get_response(self, prompt, messages=None, max_retry=5, temperature=None):
        conn_success, llm_output = False, ""
        n_retry = 0

        # Format the conversation history
        if messages is None:
            messages = [{"role": "user", "content": [{"type":"text", "text":"Describe the image."}, {"type": "image_url", "image_url": "~/anon/Retail_environment/Images/img_35.png"}]}]
        else:
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # exit(0)

        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            
            try:
                logger.info("[INFO] Generating response from Mistral API...")
                    # Make the API call to Mistral
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                )

                # Process the response
                if response and len(response.choices[0].message.content) > 0:
                # Extract the content from the first choice
                    llm_output = response.choices[0].message.content
                    logger.info(f"[INFO] Raw model output:\n{llm_output}")

                    conn_success = True
                else:
                    logger.error(f"[ERROR] Invalid response format: {response}")


            except Exception as e:
                logger.error(f"[ERROR] Mistral API error: {e}")
                if n_retry >= max_retry:
                    break

        return conn_success, llm_output


class VLMPixtral:
    def __init__(self, *args, **kwargs):
        self.llm = MistralChat(api_key="YOUR_MISTRAL_API_KEY")

    def prompt_llm(self, prompt: str, image_path: str, temperature: float = 0.0):
        # with open(image_path, "rb") as image_file:
        #     image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        image_data_uri = encode_image_as_data_uri(image_path)

        # Prepare messages payload
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": image_data_uri 
                    }
                ]
            }
        ]

        # Feed prompt to LLM
        conn_success, llm_output = self.llm.get_response(
            prompt=prompt,
            messages=messages,
            max_retry=5,  # You can adjust retry count if needed
        )
        if not conn_success:
            # print(image_path)
            # raise Exception("Failed to connect to the LLM")
            return "invalid image"

        logger.info("\n" + "#" * 50)
        logger.info(f"LLM output:\n{llm_output}")

        return llm_output
    
prompt_path = "~/Desktop/Data_Viz_agent/data-dashboard/prompts/evaluation_5metrics.txt"

def evaluate(title: str, reason: str, image_path: str, query: str, type: str):
    """
    Evaluates a visualization using the VLMPixtral multimodal model.

    Args:
        title (str): Title of the visualization.
        reason (str): Reason for producing the visualization.
        image_path (str): Path to the image file.
        prompt_path (str): Path to the prompt template file (with {title} and {reason} placeholders).

    Returns:
        str: LLM's evaluation response.
    """
    # Load and format the prompt template
    with open(prompt_path, "r") as f:
        prompt_template = f.read()
    prompt_filled = prompt_template.format(title=title, reason=reason, query=query, type=type)

    # Instantiate your model
    evaluator = VLMPixtral()

    # Run the prompt and image through the model
    print("The image path is: ")
    response = evaluator.prompt_llm(prompt=prompt_filled, image_path=image_path)

    return response


import re

def parse_sre_evaluation_response(response: str) -> dict:
    """
    Parses a fixed-format LLM evaluation response for 5 visualization metrics.

    Args:
        response (str): The raw text response from the LLM.

    Returns:
        dict: A dictionary with keys for each metric, and sub-keys for response and justification.
    """
    # Define the fixed list of metric names
    metric_names = [
        "Legality Check",
        "Layout Check",
        "Scale and Ticks Check",
        "Overall Readability Rating",
        "Usability"
    ]

    result = {}

    for metric in metric_names:
        # Pattern: MetricName: value\nJustification: value
        pattern = rf"{metric}:\s*([^\n]+)\s+Justification:\s*(.*?)(?=\n\S|$)"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            response_value = match.group(1).strip()
            reason = match.group(2).strip()
            result[metric] = {
                "response": response_value,
                "reason": reason
            }
        else:
            result[metric] = {
                "response": None,
                "reason": "Not found in response"
            }

    return result

def evaluate_charts(final_li_path, nl_query, vis_error_rate):
    """
    Evaluates a visualization using the VLMPixtral multimodal model.

    Args:
        title (str): Title of the visualization.
        reason (str): Reason for producing the visualization.
        image_path (str): Path to the image file.
        prompt_path (str): Path to the prompt template file (with {title} and {reason} placeholders).

    Returns:
        str: LLM's evaluation response.
    """
    with open(final_li_path) as f:
        final_li = json.load(f)

    result_dic = []

    for item in final_li:
        # work on better parsing the response
        # for each item, there will be title, type, reason and image path
        response = evaluate(item["title"], item["reason"], item["image_path"], nl_query, item["type"])
        if response == "invalid image":
            continue
        print(response)
        eval_result = parse_sre_evaluation_response(response)
        item_dic = {}
        item_dic["title"] = item["title"]
        item_dic["reason"] = item["reason"]
        item_dic["image_path"] = item["image_path"]
        item_dic["type"] = item["type"]
        item_dic["semantic_eval"] = eval_result
        result_dic.append(item_dic)

    return result_dic

def convert_json_to_csv(input_file, output_file, nl_query):
    # Load the JSON data (replace this with your actual file if reading from one)
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Define the fieldnames for the CSV
    fieldnames = [
        "title",
        "reason",
        "image_path",
        "type",
        "legality check response",
        "legality check reason",
        "layout check response",
        "layout check reason",
        "scale and ticks check response",
        "scale and ticks check reason",
        "overall readability rating response",
        "overall readability rating reason",
        "usability response",
        "usability reason"
    ]   

    # Write to CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for item in data:
            semantic = item["semantic_eval"]
            writer.writerow({
                "title": item["title"],
                "reason": item["reason"],
                "image_path": item["image_path"],
                "type": item["type"],
                "legality check response": semantic["Legality Check"]["response"],
                "legality check reason": semantic["Legality Check"]["reason"],
                "layout check response": semantic["Layout Check"]["response"],
                "layout check reason": semantic["Layout Check"]["reason"],
                "scale and ticks check response": semantic["Scale and Ticks Check"]["response"],
                "scale and ticks check reason": semantic["Scale and Ticks Check"]["reason"],
                "overall readability rating response": semantic["Overall Readability Rating"]["response"],
                "overall readability rating reason": semantic["Overall Readability Rating"]["reason"],
                "usability response": semantic["Usability"]["response"],
                "usability reason": semantic["Usability"]["reason"],
            })

    df = pd.read_csv(output_file)

    df["nl_query"] = nl_query

    print(f"CSV file written to: {output_file}")





