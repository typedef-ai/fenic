"""Cohere model provider implementation."""

import logging
from typing import Optional
import cohere

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class CohereModelProvider(ModelProviderClass):
    """Cohere implementation of ModelProvider."""
    _instance: Optional["CohereModelProvider"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True 

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
