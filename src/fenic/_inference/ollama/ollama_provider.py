"""Ollama model provider implementation."""

import logging
import os
from typing import Optional

import ollama

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class OllamaModelProvider(ModelProviderClass):
    """Ollama implementation of ModelProvider for local models."""

    def __init__(self, host: Optional[str] = None):
        """Initialize Ollama provider.

        Args:
            host: Base URL for the Ollama server. Defaults to http://localhost:11434
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    @property
    def name(self) -> str:
        return "ollama"

    def create_client(self):
        """Create an Ollama sync client instance."""
        return ollama.Client(host=self.host)

    def create_aio_client(self):
        """Create an Ollama async client instance."""
        return ollama.AsyncClient(host=self.host)

    async def validate_api_key(self) -> None:
        """Validate Ollama connection by checking server accessibility."""
        try:
            client = self.create_aio_client()
            # Try to list models to verify server is accessible
            await client.list()
            logger.debug(f"Ollama server accessible at {self.host}")
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}. Make sure Ollama is running at {self.host}")
            # Don't raise - let the actual requests fail with better error messages