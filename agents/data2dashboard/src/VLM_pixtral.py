import logging
import numpy as np
from mistralai import Mistral
import json
import base64

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
                last_error = str(e)
                if n_retry >= max_retry:
                    break

        return conn_success, llm_output, last_error


class VLMPixtral:
    def __init__(self, *args, **kwargs):
        self.llm = MistralChat(api_key="YOUR_MISTRAL_API_KEY")

    def prompt_llm(self, prompt: str, image_path: str, temperature: float = 0.0):
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

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
                        "image_url": f"data:image/jpeg;base64,{image_base64}" 
                    }
                ]
            }
        ]

        # Feed prompt to LLM
        conn_success, llm_output, last_error = self.llm.get_response(
            prompt=prompt,
            messages=messages,
            max_retry=5,  # You can adjust retry count if needed
        )
        if not conn_success:
            print("The error is: ", last_error)
            exit(0)
            raise Exception("Failed to connect to the LLM")
        

        logger.info("\n" + "#" * 50)
        logger.info(f"LLM output:\n{llm_output}")

        return llm_output