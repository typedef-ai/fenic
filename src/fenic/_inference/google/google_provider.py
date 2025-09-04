"""Google model provider implementation."""

import logging
import os
from abc import abstractmethod

from google import genai

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class GoogleModelProvider(ModelProviderClass):
    """Google implementation of ModelProvider."""

    @abstractmethod
    def create_client(self):
        pass

    async def validate_api_key(self) -> None:
        """Validate Google API key by listing models."""
        client = self.create_client()
        aio_client = client.aio
        _ = await aio_client.models.list()
        logger.debug(f"Google API key validation successful for {self.name}")

    def create_aio_client(self):
        """Create a Google async client instance."""
        return self.create_client().aio


class GoogleDeveloperModelProvider(GoogleModelProvider):
    """Google Developer implementation of ModelProvider."""

    @property
    def name(self) -> str:
        return "google-developer"
    
    def create_client(self):
        """Create a Google Developer client instance."""
        if "GEMINI_API_KEY" in os.environ:
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        else:
            return genai.Client()


class GoogleVertexModelProvider(GoogleModelProvider):
    """Google Vertex implementation of ModelProvider."""

    @property
    def name(self) -> str:
        return "google-vertex"
    
    def create_client(self):
        """Create a Google Vertex client instance.

        Passing `vertexai=True` automatically routes traffic through Vertex-AI if the environment is configured for it.
        """
        return genai.Client(vertexai=True)
