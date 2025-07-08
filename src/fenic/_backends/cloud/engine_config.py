from __future__ import annotations

import copy
import os
import pickle  # nosec: B403

from fenic import SessionConfig
from fenic._constants import API_KEY_SUFFIX
from fenic._inference.model_catalog import ModelProvider
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

        # Read keys from environment variables ending with _API_KEY
        env_keys = {
            key: value
            for key, value in os.environ.items()
            if key.endswith(API_KEY_SUFFIX)
        }
        language_model_providers: set[ModelProvider] = set([
            get_model_provider_for_config(model_config) for model_config in semantic_config.language_models.values()
        ])
        for language_model_provider in language_model_providers:
            if language_model_provider == ModelProvider.GOOGLE_GLA:
                google_api_key = f"GOOGLE{API_KEY_SUFFIX}"
                gemini_api_key = f"GEMINI{API_KEY_SUFFIX}"
                if google_api_key not in env_keys.keys() and gemini_api_key not in env_keys.keys():
                    raise ConfigurationError(
                        f"{google_api_key} is not set. Please set it in your environment to use {language_model_provider} models."
                    )
                if google_api_key in env_keys.keys():
                    self.model_api_keys[google_api_key] = env_keys[google_api_key]
                if gemini_api_key in env_keys.keys():
                    self.model_api_keys[google_api_key] = env_keys[gemini_api_key]
            elif language_model_provider == ModelProvider.GOOGLE_VERTEX:
                project_environment_var = "GOOGLE_CLOUD_PROJECT"
                location_environment_var = "GOOGLE_CLOUD_LOCATION"
                if project_environment_var not in env_keys.keys():
                    raise ConfigurationError(
                        f"{project_environment_var} is not set. Please set it in your environment to use {language_model_provider} models."
                    )
                if location_environment_var not in env_keys.keys():
                    raise ConfigurationError(
                        f"{location_environment_var} is not set. Please set it in your environment to use {language_model_provider} models."
                    )
                self.model_api_keys[project_environment_var] = env_keys[project_environment_var]
                self.model_api_keys[location_environment_var] = env_keys[location_environment_var]
            else:
                completions_api_key = f"{language_model_provider.value.upper()}{API_KEY_SUFFIX}"
                if completions_api_key not in env_keys.keys():
                    raise ConfigurationError(
                        f"{completions_api_key} is not set. Please set it in your environment to use {language_model_provider} models."
                    )
                self.model_api_keys[completions_api_key] = env_keys[completions_api_key]

        if self.session_config.semantic.embedding_models:
            embedding_model_providers: set[ModelProvider] = set([
                get_model_provider_for_config(model_config) for model_config in semantic_config.embedding_models.values()
            ])
            for embedding_model_provider in embedding_model_providers:
                embeddings_api_key = (
                f"{embedding_model_provider.value.upper()}{API_KEY_SUFFIX}"
            )
                if embeddings_api_key not in env_keys.keys():
                    raise ConfigurationError(
                        f"{embeddings_api_key} is not set. Please set it in your environment to use {embedding_model_provider.value.upper()} models."
                    )
                self.model_api_keys[embeddings_api_key] = env_keys[embeddings_api_key]

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
