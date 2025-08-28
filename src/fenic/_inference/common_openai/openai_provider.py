"""OpenAI model provider implementation."""

import logging
from openai import AsyncOpenAI
from typing import Optional

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class OpenAIModelProvider(ModelProviderClass):
    """OpenAI implementation of ModelProvider."""
    _instance: Optional["OpenAIModelProvider"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True

    @property
    def name(self) -> str:
        return "openai"

    @classmethod
    def get_client(cls):
        """Get OpenAI client instance."""
        return AsyncOpenAI()
    
    @classmethod
    async def validate_api_key(cls) -> None:
        """Validate OpenAI API key by listing models."""
        client = cls.get_client()
        _ = await client.models.list()
        logger.debug("OpenAI API key validation successful")

