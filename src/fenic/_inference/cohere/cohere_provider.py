"""Cohere model provider implementation."""

import logging

import cohere

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class CohereModelProvider(ModelProviderClass):
    """Cohere implementation of ModelProvider."""

    @property
    def name(self) -> str:
        return "cohere"

    def create_client(self):
        """Create a Cohere client instance."""
        return cohere.Client()

    def create_aio_client(self):
        """Create a Cohere async client instance."""
        return cohere.AsyncClientV2()
    
    async def validate_api_key(self) -> None:
        """Validate Cohere API key by making a minimal API call."""
        client = self.create_aio_client()
        _ = await client.models.list()
        logger.debug("Cohere API key validation successful")
