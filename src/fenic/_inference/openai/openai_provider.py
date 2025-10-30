"""OpenAI model provider implementation."""

import logging

import httpx
from openai import AsyncOpenAI, OpenAI

from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class OpenAIModelProvider(ModelProviderClass):
    """OpenAI implementation of ModelProvider."""

    @property
    def name(self) -> str:
        return "openai"

    def create_client(self):
        """Create an OpenAI client instance."""
        return OpenAI(http_client=httpx.AsyncClient(timeout=MAX_MODEL_CLIENT_TIMEOUT))

    def create_aio_client(self):
        """Create an OpenAI async client instance."""
        return AsyncOpenAI(http_client=httpx.AsyncClient(timeout=MAX_MODEL_CLIENT_TIMEOUT))

    async def validate_api_key(self) -> None:
        """Validate OpenAI API key by listing models."""
        client = self.create_aio_client()
        _ = await client.models.list()
        logger.debug("OpenAI API key validation successful")
