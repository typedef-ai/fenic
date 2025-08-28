"""Anthropic model provider implementation."""

import logging

import anthropic
from typing import Optional

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class AnthropicModelProvider(ModelProviderClass):
    """Anthropic implementation of ModelProvider."""
    
    _instance: Optional["AnthropicModelProvider"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True

    @property
    def name(self) -> str:
        return "anthropic"
    
    def get_client(self):
        """Get Anthropic async client instance."""
        return anthropic.Anthropic()
    
    async def validate_api_key(self) -> None:
        """Validate Anthropic API key by making a minimal completion request."""
        client = self.get_client()
        _ = await client.models.list()
        logger.debug("Anthropic API key validation successful")


# Create singleton instance
anthropic_provider = AnthropicModelProvider()