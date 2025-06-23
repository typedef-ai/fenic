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
class ResolvedAnthropicModelPreset:
    thinking_token_budget: Optional[int] = None

@dataclass
class ResolvedGoogleModelPreset:
    thinking_token_budget: Optional[int] = None


@dataclass
class ResolvedOpenAIModelPreset:
    reasoning_effort: Optional[ReasoningEffort] = None

@dataclass
class ResolvedOpenAIModelConfig:
    model_name: str
    rpm: int
    tpm: int
    presets: Optional[dict[str, ResolvedOpenAIModelPreset]] = None
    default_preset: Optional[str] = None


@dataclass
class ResolvedAnthropicModelConfig:
    model_name: str
    rpm: int
    input_tpm: int
    output_tpm: int
    presets: Optional[dict[str, ResolvedAnthropicModelPreset]] = None
    default_preset: Optional[str] = None

@dataclass
class ResolvedGoogleModelConfig:
    model_name: str
    model_provider: Literal[ModelProvider.GOOGLE_GLA, ModelProvider.GOOGLE_VERTEX]
    rpm: int
    tpm: int
    presets: Optional[dict[str, ResolvedGoogleModelPreset]] = None
    default_preset: Optional[str] = None

ResolvedModelConfig = Union[ResolvedOpenAIModelConfig, ResolvedAnthropicModelConfig, ResolvedGoogleModelConfig]


# --- Semantic / Cloud / Session Configs ---

@dataclass
class ResolvedSemanticConfig:
    language_models: dict[str, ResolvedModelConfig]
    all_language_model_aliases: set[str]
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
