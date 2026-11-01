"""
llm_agent.py

This module provides a generic LLMAgent class that supports calling LLM services 
from multiple remote hosting providers (e.g., RITS, OPENAI, PROVIDER). The service 
to use is specified via the command line or by client code when initializing LLMAgent.
"""

import os
import shelve
import traceback
import litellm
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

# RITS settings
RITS_API_KEY = os.getenv("RITS_API_KEY")

RITS_MODELS = {
    "ibm/granite-3-2-8b-instruct": {
        "model_id": "openai/ibm-granite/granite-3.2-8b-instruct",
        "api_base": "https://inference-host/granite-3-2-8b-instruct/v1",
        "max_new_tokens": 12800,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    },
    "ibm/granite-3-1-8b-instruct": {
        "model_id": "openai/ibm-granite/granite-3.1-8b-instruct",
        "api_base": "https://inference-host/granite-3-1-8b-instruct/v1",
        "max_new_tokens": 12800,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    },
    "ibm/granite-3-0-8b-instruct": {
        "model_id": "openai/ibm-granite/granite-3.0-8b-instruct",
        "api_base": "https://inference-host/granite-3-0-8b-instruct/v1",
        "max_new_tokens": 2048,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    },
    "ibm/granite-3-0-8b-base": {
        "model_id": "openai/ibm-granite/granite-3.0-8b-base",
        "api_base": "https://inference-host/granite-3-0-8b-base/v1",
        "max_new_tokens": 1280,
        "prompt_template": "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "prompt_begin": "<|start_of_role|>user<|end_of_role|>",
        "prompt_end": "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    },
    "Llama-3.1-8B-Instruct": {
        "model_id": "openai/meta-llama/Llama-3.1-8B-Instruct",
        "api_base": "https://inference-host/llama-3-1-8b-instruct",
        "max_new_tokens": 4096,
        "prompt_template": "<s>[INST] {user_input} [/INST]\n<s>[INST] ",
        "prompt_begin": "<s>[INST] ",
        "prompt_end": " [/INST]"
    },
    "granite-34b-code-instruct-8k": {
        "model_id": "openai/ibm-granite/granite-34b-code-instruct-8k",
        "api_base": "https://inference-host/granite-34b-code-instruct-8k",
        "max_new_tokens": 4096,
        "prompt_template": "<s>[INST] {user_input} [/INST]\n<s>[INST] ",
        "prompt_begin": "<s>[INST] ",
        "prompt_end": " [/INST]"
    }
}

class LLMAgent:
    """
    A generic agent for calling LLM services from multiple providers such as RITS, OPENAI, and PROVIDER.
    """

    def __init__(self, 
                 service: str = "RITS", 
                 model_name: str = "ibm/granite-3-2-8b-instruct", 
                 temperature: float = 0.25):
        """
        Initializes the LLMAgent.

        Args:
            service (str): The service to use (e.g., "RITS", "OPENAI", "PROVIDER").
            model_name (str): The model name identifier.
                              For RITS, this must exist in RITS_MODELS.
            temperature (float): Sampling temperature for the LLM call.
        """

        self.service = service.upper()
        self.temperature = temperature

        if self.service == "RITS" or self.service == "rits":
            if model_name not in list(RITS_MODELS.keys()):
                print(RITS_MODELS.keys())
                raise ValueError(f"Unknown RITS model: {model_name}")
            self.model_info = RITS_MODELS[model_name]
            self.model_id = self.model_info["model_id"]
            self.api_base = self.model_info["api_base"]
            self.prompt_template = self.model_info["prompt_template"]
            self.prompt_begin = self.model_info["prompt_begin"]
            self.prompt_end = self.model_info["prompt_end"]
            self.max_new_tokens = self.model_info.get("max_new_tokens", 1024)

            if not RITS_API_KEY:
                raise ValueError("RITS_API_KEY not found in environment")

        elif self.service == "OPENAI" or self.service == "openai":
            self.model_id = os.getenv("OPENAI_MODEL_ID")
            self.api_base = os.getenv("OPENAI_API_BASE")
            self.prompt_template = os.getenv("OPENAI_PROMPT_TEMPLATE", "{user_input}")
            self.max_new_tokens = int(os.getenv("OPENAI_MAX_NEW_TOKENS", "2048"))
            self.openapi_api_key = os.getenv("OPENAI_API_KEY")
            if not self.model_id or not self.api_base or not self.openapi_api_key:
                raise ValueError("OPENAI_MODEL_ID, OPENAI_API_BASE, or OPENAI_API_KEY not found in environment")

        elif self.service == "PROVIDER" or self.service == "provider":
            self.model_id = model_name
            self.max_new_tokens = int(os.getenv("PROVIDER_MAX_NEW_TOKENS", "8192"))
            self.provider_api_key = os.getenv("PROVIDER_APIKEY")
            self.provider_project_id = os.getenv("PROVIDER_PROJECT_ID")
            self.prompt_template = os.getenv("PROVIDER_PROMPT_TEMPLATE", "{user_input}")
            if not self.provider_api_key or not self.provider_project_id:
                raise ValueError("PROVIDER_APIKEY or PROVIDER_PROJECT_ID not found in environment")
        else:
            raise ValueError(f"Unsupported service: {self.service}")

    def format_prompt(self, refined_schema: str, user_input: str) -> str:
        """
        Formats the prompt based on the model's prompt template.

        Args:
            refined_schema (str): The refined GraphQL schema.
            user_input (str): The natural language input.

        Returns:
            str: The formatted prompt.
        """
        return self.prompt_template.format(
            user_input=f"Schema: {refined_schema} Utterance: {user_input}"
        )

    def call_llm(self, prompt: str) -> str:
        """ 
        Calls the selected LLM service and returns the generated completion text.
        """

        try:
            if self.service == "RITS":
                responses = litellm.batch_completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    provider="openai",
                    messages=[[{"role": "user", "content": prompt}]],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    api_key=RITS_API_KEY,
                    extra_headers={"RITS_API_KEY": RITS_API_KEY}
                )
                try:
                    response = responses[0]
                    generated_text = response.choices[0].message.content
                except Exception as e:
                    print(str(e))
                    return ""

            elif self.service == "OPENAI":
                responses = litellm.batch_completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    provider="openapi",
                    messages=[[{"role": "user", "content": prompt}]],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    api_key=self.openapi_api_key,
                    extra_headers={"OPENAI_API_KEY": self.openapi_api_key}
                )
                try:
                    response = responses[0]
                    generated_text = response.choices[0].message.content
                except Exception as e:
                    print(str(e))
                    return ""

            elif self.service == "PROVIDER" or self.service == "provider":
                from provider_sdk import Credentials
                from provider_sdk.metanames import GenTextParamsMetaNames as GenParams
                from provider_sdk.foundation_models.utils.enums import DecodingMethods
                from provider_sdk.foundation_models import ModelInference

                parameters = {
                    GenParams.MIN_NEW_TOKENS: 0,
                    GenParams.MAX_NEW_TOKENS: self.max_new_tokens,
                    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                    GenParams.REPETITION_PENALTY: 1,
                    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
                }
                credentials = Credentials(
                    url="https://provider-host",
                    api_key=self.provider_api_key
                )
                model = ModelInference(
                    model_id=self.model_id,
                    params=parameters,
                    credentials=credentials,
                    project_id=self.provider_project_id
                )
                try:
                    response = model.generate_text(prompt=[prompt])
                    generated_text = response[0]() if callable(response[0]) else response[0]
                except Exception as e:
                    print(str(e))
                    return ""
            else:
                raise ValueError(f"Service {self.service} is not implemented.")

            try:
                final_text = generated_text.strip()
            except Exception as e:
                final_text = ""

            return final_text

        except Exception as e:
            traceback.print_exc()
            return f"Error calling LLM: {str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize LLMAgent for a specific LLM service.")
    parser.add_argument("--service", type=str, default="PROVIDER", help="The LLM service to use (e.g., RITS, OPENAI, PROVIDER).")
    parser.add_argument("--model_name", type=str, default="granite-3-1-8b-instruct", help="Model name identifier.")
    parser.add_argument("--temperature", type=float, default=0.25, help="Sampling temperature.")
    args = parser.parse_args()

    agent = LLMAgent(service=args.service, model_name=args.model_name, temperature=args.temperature)
    test_prompt = "Test prompt for LLM service."
    output = agent.call_llm(test_prompt)
    print("LLM output:", output)