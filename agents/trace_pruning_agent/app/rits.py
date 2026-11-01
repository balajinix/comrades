import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ChatRits(ChatOpenAI):
    """RITS chat model integration using langchain-openai.

    This class sets default parameters (like `api_key`, `base_url`) based on environment
    variables and the provided model name. It inherits from ChatOpenAI and overrides
    the initialization to support RITS-specific endpoints.
    """
    def __init__(self, **kwargs):
        """Initialize a ChatRits model with RITS-specific defaults.

        Args:
            **kwargs: Additional keyword arguments that can include:
                - model (str): The name of the RITS model.
                - model_name (str): Alternative key for specifying the RITS model.

        Raises:
            ValueError: If neither 'model' nor 'model_name' is provided.
            ValueError: If RITS_API_KEY environment variable is not set.
        """
        model_name = kwargs.get("model", kwargs.get("model_name"))
        if model_name is None:
            raise ValueError("model or model_name is required")

        # Get RITS API key from environment variables
        rits_api_key = os.getenv("RITS_API_KEY")
        if rits_api_key is None:
            raise ValueError("RITS_API_KEY environment variable is required")

        # Set default values for overriding fields
        kwargs.setdefault("model_name", model_name)
        kwargs.setdefault("api_key", "/")  # As per your code
        # Use the API key from environment variables
        kwargs.setdefault("default_headers", {"RITS_API_KEY": rits_api_key})
        kwargs.setdefault(
            "base_url",
            f"https://inference-host/"
            f"{model_name.split('/')[1].strip().lower().replace('.', '-')}/v1"
        )

        # Call parent constructor
        super().__init__(**kwargs)
