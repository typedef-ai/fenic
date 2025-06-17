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

from fenic._inference.model_catalog import (
    ANTHROPIC_AVAILABLE_LANGUAGE_MODELS,
    GOOGLE_GLA_AVAILABLE_MODELS,
    OPENAI_AVAILABLE_EMBEDDING_MODELS,
    OPENAI_AVAILABLE_LANGUAGE_MODELS,
)

ReasoningEffort = Literal["none", "low", "medium", "high"]

# --- Enums ---

class CloudExecutorSize(str, Enum):
    SMALL = "INSTANCE_SIZE_S"
    MEDIUM = "INSTANCE_SIZE_M"
    LARGE = "INSTANCE_SIZE_L"
    XLARGE = "INSTANCE_SIZE_XL"


# --- Model Configs ---

@dataclass
class ResolvedOpenAIModelConfig:
    model_name: Union[OPENAI_AVAILABLE_LANGUAGE_MODELS, OPENAI_AVAILABLE_EMBEDDING_MODELS]
    rpm: int
    tpm: int


@dataclass
class ResolvedAnthropicModelConfig:
    model_name: ANTHROPIC_AVAILABLE_LANGUAGE_MODELS
    rpm: int
    input_tpm: int
    output_tpm: int

@dataclass
class ResolvedGoogleGLAModelConfig:
    model_name: GOOGLE_GLA_AVAILABLE_MODELS
    rpm: int
    tpm: int
    reasoning_effort: Optional[ReasoningEffort]

ResolvedModelConfig = Union[ResolvedOpenAIModelConfig, ResolvedAnthropicModelConfig, ResolvedGoogleGLAModelConfig]


# --- Semantic / Cloud / Session Configs ---

@dataclass
class ResolvedSemanticConfig:
    language_models: dict[str, ResolvedModelConfig]
    default_language_model: str
    embedding_models: Optional[dict[str, ResolvedOpenAIModelConfig]] = None
    default_embedding_model: Optional[str] = None


@dataclass
class ResolvedCloudConfig:
    size: Optional[CloudExecutorSize] = None


@dataclass
class ResolvedSessionConfig:

    app_name: str
    db_path: Optional[Path]
    semantic: ResolvedSemanticConfig
    cloud: Optional[ResolvedCloudConfig] = None
