"""Internal configuration classes for resolved session settings.

This module defines internal configuration classes that represent the fully resolved
state of a session after processing user-provided configuration. These classes are
used internally after the user creates a SessionConfig in the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from fenic.core._inference.model_catalog import ModelProvider

ReasoningEffort = Literal["low", "medium", "high"]

# --- Enums ---

class CloudExecutorSize(str, Enum):
    SMALL = "INSTANCE_SIZE_S"
    MEDIUM = "INSTANCE_SIZE_M"
    LARGE = "INSTANCE_SIZE_L"
    XLARGE = "INSTANCE_SIZE_XL"


# --- Model Configs ---

@dataclass
class ResolvedAnthropicModelProfile:
    thinking_token_budget: Optional[int] = None

@dataclass
class ResolvedGoogleModelProfile:
    thinking_token_budget: Optional[int] = None
    embedding_dimensionality: Optional[int] = None
    embedding_task_type: Optional[str] = None



@dataclass
class ResolvedOpenAIModelProfile:
    reasoning_effort: Optional[ReasoningEffort] = None

@dataclass
class ResolvedCohereModelProfile:
    embedding_dimensionality: Optional[int] = None
    embedding_task_type: Optional[str] = None

@dataclass
class ResolvedOpenAIModelConfig:
    model_name: str
    rpm: int
    tpm: int
    model_provider: ModelProvider = ModelProvider.OPENAI
    profiles: Optional[dict[str, ResolvedOpenAIModelProfile]] = None
    default_profile: Optional[str] = None


@dataclass
class ResolvedAnthropicModelConfig:
    model_name: str
    rpm: int
    input_tpm: int
    output_tpm: int
    model_provider: ModelProvider = ModelProvider.ANTHROPIC
    profiles: Optional[dict[str, ResolvedAnthropicModelProfile]] = None
    default_profile: Optional[str] = None

@dataclass
class ResolvedGoogleModelConfig:
    model_name: str
    model_provider: Literal[ModelProvider.GOOGLE_DEVELOPER, ModelProvider.GOOGLE_VERTEX]
    rpm: int
    tpm: int
    profiles: Optional[dict[str, ResolvedGoogleModelProfile]] = None
    default_profile: Optional[str] = None

@dataclass
class ResolvedCohereModelConfig:
    model_name: str
    rpm: int
    tpm: int
    model_provider: ModelProvider = ModelProvider.COHERE
    profiles: Optional[dict[str, ResolvedCohereModelProfile]] = None
    default_profile: Optional[str] = None

ResolvedModelConfig = Union[ResolvedOpenAIModelConfig, ResolvedAnthropicModelConfig, ResolvedGoogleModelConfig, ResolvedCohereModelConfig]


# --- Semantic / Cloud / Session Configs ---

@dataclass
class ResolvedSemanticConfig:
    language_models: Optional[ResolvedLanguageModelConfig] = None
    embedding_models: Optional[ResolvedEmbeddingModelConfig] = None

@dataclass
class ResolvedLanguageModelConfig:
    model_configs: dict[str, ResolvedModelConfig]
    default_model: str

@dataclass
class ResolvedEmbeddingModelConfig:
    model_configs: dict[str, ResolvedModelConfig]
    default_model: str

@dataclass
class ResolvedCloudConfig:
    size: Optional[CloudExecutorSize] = None


@dataclass
class ResolvedSessionConfig:
    app_name: str
    db_path: Optional[Path]
    semantic: ResolvedSemanticConfig
    cloud: Optional[ResolvedCloudConfig] = None
