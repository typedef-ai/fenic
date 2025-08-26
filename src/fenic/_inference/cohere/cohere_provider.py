"""Cohere model provider implementation."""

import logging
import cohere

from fenic.core._inference.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class CohereModelProvider(ModelProvider):
    """Cohere implementation of ModelProvider."""
    
    @property
    def name(self) -> str:
        return "cohere"
    
    def get_client(self):
        """Get Cohere client instance."""
        return cohere.AsyncClientV2()
    
    async def validate_api_key(self) -> None:
        """Validate Cohere API key by making a minimal API call."""
        client = self.get_client()
        _ = await client.models.list()
        logger.debug("Cohere API key validation successful")


# Create singleton instance
cohere_provider = CohereModelProvider()