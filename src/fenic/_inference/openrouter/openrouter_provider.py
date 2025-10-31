import logging
import os
import threading
from functools import cached_property
from typing import Any, Dict, Optional

import httpx
import requests
from openai import AsyncOpenAI, OpenAI

from fenic._constants import MAX_MODEL_CLIENT_TIMEOUT
from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._inference.model_provider import ModelProviderClass
from fenic.core.error import SessionError, ValidationError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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
        # Register dynamic loader with model_catalog
        model_catalog.register_dynamic_provider(ModelProvider.OPENROUTER, self._load_models_once)

    @property
    def name(self) -> str:
        return "openrouter"

    @cached_property
    def client(self):
        """Return an OpenAI SDK client configured for OpenRouter."""
        return OpenAI(
            default_headers=self._headers,
            base_url=OPENROUTER_BASE_URL,
            http_client=httpx.AsyncClient(timeout=MAX_MODEL_CLIENT_TIMEOUT),
        )

    @cached_property
    def aio_client(self):
        """Return an Async OpenAI SDK client configured for OpenRouter."""
        return AsyncOpenAI(
            default_headers=self._headers,
            base_url=OPENROUTER_BASE_URL,
            http_client=httpx.AsyncClient(timeout=MAX_MODEL_CLIENT_TIMEOUT),
        )

    def create_client(self):
        """Return an OpenAI SDK client configured for OpenRouter."""
        return self.client

    def create_aio_client(self):
        """Return an Async OpenAI SDK client configured for OpenRouter."""
        return self.aio_client

    def _translate_model(
        self, model_obj: Dict[str, Any]
    ) -> Optional[CompletionModelParameters]:
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
        if isinstance(top_provider.get("context_length"), int) and top_provider.get("context_length") > 0:
            context_len = int(top_provider.get("context_length"))
        elif isinstance(model_obj.get("context_length"), int) and model_obj.get("context_length") > 0:
            context_len = int(model_obj.get("context_length"))
        if not context_len:
            return None

        max_tokens = None
        if isinstance(top_provider.get("max_completion_tokens"), int):
            max_tokens = int(top_provider.get("max_completion_tokens"))
        if not max_tokens:
            max_tokens = min(8192, context_len)

        supported_params = set(model_obj.get("supported_parameters") or [])
        supports_reasoning = ("reasoning" in supported_params) or (
            "include_reasoning" in supported_params
        )
        supports_custom_temperature = "temperature" in supported_params
        supports_verbosity = "verbosity" in supported_params

        return CompletionModelParameters(
            input_token_cost=input_cost,
            output_token_cost=output_cost,
            context_window_length=context_len,
            max_output_tokens=max_tokens,
            cached_input_token_read_cost=cached_read_cost,
            supports_profiles=True,
            supports_reasoning=supports_reasoning,
            supports_custom_temperature=supports_custom_temperature,
            supports_verbosity=supports_verbosity,
            supports_pdf_parsing=True, # Even if the model doesn't support pdf file processing, OpenRouter can use its separate processing engines
            supported_parameters=supported_params,
        )

    def _load_models_once(self):
        if self._models_loaded:
            return
        url = f"{OPENROUTER_BASE_URL}/models/user"
        if not self._headers.get("Authorization"):
            url = f"{OPENROUTER_BASE_URL}/models"
        resp = requests.get(url, headers=self._headers, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter models request failed: {resp.status_code} {resp.text}"
            )
        payload = resp.json() or {}
        models = payload.get("data") or []
        if not models:
            raise SessionError("No OpenRouter models found. Ensure the OpenRouter account is configured to include at least one provider.")
        added_models = []
        untranslated_models = []
        for model in models:
            params = self._translate_model(model)
            if not params:
                logging.warning(
                    f"Could not extract Completion Parameters from OpenRouter model: {model}"
                )
                untranslated_models.append(model)
                continue
            model_id = model.get("id")
            if isinstance(model_id, str):
                # Register into global catalog so standard lookups succeed
                model_catalog.add_model(ModelProvider.OPENROUTER, model_id, params)
                added_models.append(model_id)
        if not added_models:
            raise SessionError("Failed to process and load OpenRouter models")
        self._models_loaded = True
        logger.info(f"OpenRouter model cache initialized with {len(added_models)} models")
        if untranslated_models:
            logger.warning(f"Failed to process and load OpenRouter models: {untranslated_models}")

    async def validate_api_key(self) -> None:
        url = f"{OPENROUTER_BASE_URL}/key"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValidationError("OpenRouter API Key is not set. Ensure the `OPENROUTER_API_KEY` is populated to use OpenRouter models.")
        resp = requests.get(url, headers=self._headers, timeout=30)
        if resp.status_code >= 400:
            raise ValidationError(
                f"OpenRouter key request failed: {resp.status_code} {resp.text}"
            )
        logger.debug("OpenRouter API key validation successful")

    @cached_property
    def _headers(self) -> Dict[str, str]:
        key = os.environ.get("OPENROUTER_API_KEY")
        headers = {
            "Accept": "application/json",
            "HTTP-Referer": "https://github.com/typedef-ai/fenic",
            "X-Title": "fenic (by typedef.ai)",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers


openrouter_provider = OpenRouterModelProvider()
