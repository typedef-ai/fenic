"""Google model provider implementation."""

import logging
import os
from google import genai

from fenic.core._inference.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class GoogleModelProvider(ModelProvider):
    """Google implementation of ModelProvider."""
    
    def __init__(self, provider_type: str = "google-developer"):
        """Initialize Google provider.
        
        Args:
            provider_type: Either "google-developer" or "google-vertex"
        """
        self._provider_type = provider_type
    
    @property
    def name(self) -> str:
        return self._provider_type
    
    def get_client(self):
        # Native gen-ai client. Passing `vertexai=True` automatically routes traffic
        # through Vertex-AI if the environment is configured for it.
        if self._provider_type == "google-developer":
            if "GEMINI_API_KEY" in os.environ:
                return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            else:
                return genai.Client()
        else:
            return genai.Client(vertexai=True)
    
    async def validate_api_key(self) -> None:
        """Validate Google API key by listing models."""
        client = self.get_client()
        aio_client = client.aio
        _ = await aio_client.models.list()
        logger.debug(f"Google API key validation successful for {self._provider_type}")


# Create singleton instances
google_developer_provider = GoogleModelProvider("google-developer")
google_vertex_provider = GoogleModelProvider("google-vertex")