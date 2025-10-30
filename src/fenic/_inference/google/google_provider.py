"""Google model provider implementation."""

import logging
import os
import warnings
from abc import abstractmethod

from google import genai
from google.genai._common import ExperimentalWarning as _GoogleExperimentalWarning
from google.genai.types import HttpOptions

from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic.core._inference.model_provider import ModelProviderClass

MAX_CLIENT_TIMEOUT_MS = MAX_MODEL_CLIENT_TIMEOUT * 1000

logger = logging.getLogger(__name__)


class GoogleModelProvider(ModelProviderClass):
    """Google implementation of ModelProvider."""

    def __init__(self):
        _suppress_google_experimental_warnings()

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
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=HttpOptions(timeout=MAX_CLIENT_TIMEOUT_MS))
        else:
            return genai.Client(http_options=HttpOptions(timeout=MAX_CLIENT_TIMEOUT_MS))


class GoogleVertexModelProvider(GoogleModelProvider):
    """Google Vertex implementation of ModelProvider."""

    @property
    def name(self) -> str:
        return "google-vertex"
    
    def create_client(self):
        """Create a Google Vertex client instance.

        Passing `vertexai=True` automatically routes traffic through Vertex-AI if the environment is configured for it.
        """
        return genai.Client(vertexai=True, http_options=HttpOptions(timeout=MAX_CLIENT_TIMEOUT_MS))


def _suppress_google_experimental_warnings() -> None:
    """Silence ExperimentalWarning from google-genai local tokenizer."""
    warnings.filterwarnings("ignore", category=_GoogleExperimentalWarning)
