"""OpenAI model provider implementation."""

import logging
from openai import AsyncOpenAI

from fenic.core._inference.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class OpenAIModelProvider(ModelProvider):
    """OpenAI implementation of ModelProvider."""
    
    @property
    def name(self) -> str:
        return "openai"
    
    def get_client(self):
        """Get OpenAI client instance."""
        return AsyncOpenAI()
    
    async def validate_api_key(self) -> None:
        """Validate OpenAI API key by listing models."""
        client = self.get_client()
        _ = await client.models.list()
        logger.debug("OpenAI API key validation successful")


# Create singleton instance
openai_provider = OpenAIModelProvider()