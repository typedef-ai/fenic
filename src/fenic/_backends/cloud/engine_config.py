from __future__ import annotations

import copy
import os
import pickle  # nosec: B403

from fenic import SessionConfig
from fenic._constants import API_KEY_SUFFIX
from fenic.core._inference.model_catalog import ModelProvider
from fenic.core._resolved_session_config import (
    ResolvedAnthropicModelConfig,
    ResolvedGoogleModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
)
from fenic.core.error import ConfigurationError, InternalError


class CloudSessionConfig:
    """Configuration required for cloud session.

    Attributes:
        session_config (SessionConfig): The session configuration created by the user.
        model_api_keys (Dict[str, str]): A dictionary of model API keys.

    Upon initialization, will read the API keys required for the SemanticConfig from the environment variables
    """

    def __init__(self, session_config: SessionConfig):
        self.model_api_keys = {}
        self.unresolved_session_config = copy.deepcopy(session_config)
        self.session_config = copy.deepcopy(session_config._to_resolved_config())
        self.unresolved_session_config.cloud = None
        self.session_config.cloud = None

        semantic_config = self.session_config.semantic
        env_keys = dict(os.environ)

        def get_and_store_key(env_var: str, provider: ModelProvider):
            if env_var not in env_keys:
                raise ConfigurationError(
                    f"{env_var} is not set. Please set it in your environment to use {provider} models."
                )
            self.model_api_keys[env_var] = env_keys[env_var]

        providers: set[ModelProvider] = set()

        if semantic_config.language_models:
            for model_config in semantic_config.language_models.model_configs.values():
                providers.add(get_model_provider_for_config(model_config))

        if semantic_config.embedding_models:
            for model_config in semantic_config.embedding_models.model_configs.values():
                providers.add(get_model_provider_for_config(model_config))

        for provider in providers:
            if provider == ModelProvider.GOOGLE_DEVELOPER:
                google_api_key = f"GOOGLE{API_KEY_SUFFIX}"
                gemini_api_key = f"GEMINI{API_KEY_SUFFIX}"
                if google_api_key in env_keys:
                    self.model_api_keys[google_api_key] = env_keys[google_api_key]
                elif gemini_api_key in env_keys:
                    self.model_api_keys[google_api_key] = env_keys[gemini_api_key]
                else:
                    raise ConfigurationError(
                        f"{google_api_key} or {gemini_api_key} must be set to use {provider} models."
                    )
            elif provider == ModelProvider.GOOGLE_VERTEX:
                get_and_store_key("GOOGLE_CLOUD_PROJECT", provider)
                get_and_store_key("GOOGLE_CLOUD_LOCATION", provider)
            else:
                key = f"{provider.value.upper()}{API_KEY_SUFFIX}"
                get_and_store_key(key, provider)


    @staticmethod
    def deserialize(data: bytes) -> CloudSessionConfig:
        return pickle.loads(data)  # nosec: B301

    def serialize(self) -> bytes:
        return pickle.dumps(self)


def get_model_provider_for_config(model_config: ResolvedModelConfig) -> ModelProvider:
    if isinstance(model_config, ResolvedOpenAIModelConfig):
        return ModelProvider.OPENAI
    elif isinstance(model_config, ResolvedAnthropicModelConfig):
        return ModelProvider.ANTHROPIC
    if isinstance(model_config, ResolvedGoogleModelConfig):
        return model_config.model_provider
    else:
        raise InternalError(f"Unsupported model {model_config} in semantic config.")
