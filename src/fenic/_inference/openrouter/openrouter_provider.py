import logging
import os
import threading
from functools import cached_property
from typing import Any, Dict, Optional

import requests
import httpx
from openai import AsyncOpenAI, OpenAI

from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)

class OpenRouterModelProvider(ModelProviderClass):
    """Lazy singleton provider that caches OpenRouter model parameters.

    Fetches models from `https://openrouter.ai/api/v1/models` using `requests`,
    translates to `CompletionModelParameters`, and caches on first access.
    """

    _instance: Optional["OpenRouterModelProvider"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._models_loaded = False
        self._models_lock = threading.Lock()
        # Register dynamic loader with model_catalog
        try:
            model_catalog.register_dynamic_provider(ModelProvider.OPENROUTER, self._load_models_once)
        except Exception:
            logger.error("Failed to register OpenRouter dynamic loader", exc_info=True)

    @property
    def name(self) -> str:
        return "openrouter"

    @cached_property
    def _headers(self) -> Dict[str, str]:
        key = os.environ.get("OPENROUTER_API_KEY")
        headers = {"Accept": "application/json", "X-Title": "typedef-fenic"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @cached_property
    def client(self):
        """Return an OpenAI SDK client configured for OpenRouter."""
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    @cached_property
    def aio_client(self):
        """Return an Async OpenAI SDK client configured for OpenRouter."""
        return AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    # Compatibility with ModelProviderClass abstract methods
    def create_client(self):
        return self.client

    def create_aio_client(self):
        return self.aio_client

    def _translate_model(self, model_obj: Dict[str, Any]) -> Optional[CompletionModelParameters]:
        pricing = model_obj.get("pricing") or {}
        top_provider = model_obj.get("top_provider") or {}

        try:
            input_cost = float(pricing.get("prompt", 0.0))
            output_cost = float(pricing.get("completion", 0.0))
        except (TypeError, ValueError):
            return None

        cached_read_cost = 0.0
        try:
            if pricing.get("input_cache_read") is not None:
                cached_read_cost = float(pricing.get("input_cache_read"))
        except (TypeError, ValueError):
            cached_read_cost = 0.0

        context_len = None
        if isinstance(top_provider.get("context_length"), int):
            context_len = int(top_provider.get("context_length"))
        elif isinstance(model_obj.get("context_length"), int):
            context_len = int(model_obj.get("context_length"))
        if context_len is None:
            return None

        max_tokens = None
        if isinstance(top_provider.get("max_completion_tokens"), int):
            max_tokens = int(top_provider.get("max_completion_tokens"))
        if not max_tokens:
            max_tokens = min(8192, context_len)

        supported_params = set(model_obj.get("supported_parameters") or [])
        supports_reasoning = ("reasoning" in supported_params) or ("include_reasoning" in supported_params)
        supports_custom_temperature = ("temperature" in supported_params)

        return CompletionModelParameters(
            input_token_cost=input_cost,
            output_token_cost=output_cost,
            context_window_length=context_len,
            max_output_tokens=max_tokens,
            cached_input_token_read_cost=cached_read_cost,
            supports_profiles=True,
            supports_reasoning=supports_reasoning,
            supports_custom_temperature=supports_custom_temperature,
            supported_parameters=supported_params,
        )

    def _load_models_once(self):
        if self._models_loaded:
            return
        with self._models_lock:
            if self._models_loaded:
                return
            url = "https://openrouter.ai/api/v1/models"
            resp = requests.get(url, headers=self._headers, timeout=30)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"OpenRouter models request failed: {resp.status_code} {resp.text}"
                )
            payload = resp.json() or {}
            models = payload.get("data") or []
            loaded = 0
            for model in models:
                params = self._translate_model(model)
                if not params:
                    continue
                model_id = model.get("id")
                canonical = model.get("canonical_slug")
                if isinstance(model_id, str):
                    # Register into global catalog so standard lookups succeed
                    try:
                        model_catalog.add_model(ModelProvider.OPENROUTER, model_id, params, snapshots=[canonical] if isinstance(canonical, str) else None)
                    except Exception:
                        logger.error(f"Failed to add OpenRouter model to catalog: {model_id}", exc_info=True)
                    loaded += 1
            self._models_loaded = True
            logger.info(f"OpenRouter model cache initialized with {loaded} models")

    async def validate_api_key(self) -> None:
        self._load_models_once()
        logger.debug("OpenRouter API key validation successful")
