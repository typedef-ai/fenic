"""Anthropic model provider implementation."""

import logging

import anthropic
import httpx

from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class AnthropicModelProvider(ModelProviderClass):
    """Anthropic implementation of ModelProvider."""

    def __init__(self, base_url: str | None = None):
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "anthropic"

    def create_client(self):
        """Create an Anthropic sync client instance."""
        return anthropic.Client(
            base_url=self._base_url,
            http_client=httpx.Client(timeout=MAX_MODEL_CLIENT_TIMEOUT),
        )

    def create_aio_client(self):
        """Create an Anthropic async client instance."""
        return anthropic.AsyncAnthropic(
            base_url=self._base_url,
            http_client=httpx.AsyncClient(timeout=MAX_MODEL_CLIENT_TIMEOUT),
        )
    
    async def validate_api_key(self) -> None:
        """Validate Anthropic API key by making a minimal completion request."""
        client = self.create_aio_client()
        _ = await client.models.list()
        logger.debug("Anthropic API key validation successful")
