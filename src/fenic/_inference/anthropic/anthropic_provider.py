"""Anthropic model provider implementation."""

import logging
from anthropic import AsyncAnthropic, Client

from fenic.core._inference.model_provider import ModelProvider

logger = logging.getLogger(__name__)


class AnthropicModelProvider(ModelProvider):
    """Anthropic implementation of ModelProvider."""
    
    @property
    def name(self) -> str:
        return "anthropic"
    
    def get_client(self):
        """Get Anthropic async client instance."""
        return AsyncAnthropic()
    
    def get_sync_client(self):
        """Get Anthropic sync client instance."""
        return Client()
    
    async def validate_api_key(self) -> None:
        """Validate Anthropic API key by making a minimal completion request."""
        client = self.get_client()
        _ = await client.completions.create(
            model="claude-3-haiku-20240307",  # Use cheapest available model for validation
            prompt="ping",
            max_tokens_to_sample=1,
            temperature=0
        )
        logger.debug("Anthropic API key validation successful")


# Create singleton instance
anthropic_provider = AnthropicModelProvider()