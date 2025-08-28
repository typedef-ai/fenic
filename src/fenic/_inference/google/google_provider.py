"""Google model provider implementation."""

from abc import abstractmethod
import logging
import os
from google import genai
from typing import Optional

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class GoogleModelProvider(ModelProviderClass):
    """Google implementation of ModelProvider."""

    @abstractmethod
    def get_client(self):
        pass

    async def validate_api_key(self) -> None:
        """Validate Google API key by listing models."""
        client = self.get_client()
        aio_client = client.aio
        _ = await aio_client.models.list()
        logger.debug(f"Google API key validation successful for {self._provider_type}")


class GoogleDeveloperModelProvider(GoogleModelProvider):
    """Google implementation of ModelProvider."""

    _instance: Optional["GoogleDeveloperModelProvider"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True 

    @property
    def name(self) -> str:
        return "google-developer"
    
    def get_client(self):
        # Native gen-ai client. 
        if "GEMINI_API_KEY" in os.environ:
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            return genai.Client()


class GoogleVertexModelProvider(GoogleModelProvider):
    """Google implementation of ModelProvider."""

    _instance: Optional["GoogleVertexModelProvider"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True 

    @property
    def name(self) -> str:
        return "google-vertex"
    
    def get_client(self):
        # Native gen-ai client. Passing `vertexai=True` automatically routes traffic
        # through Vertex-AI if the environment is configured for it.
        return genai.Client(vertexai=True)

