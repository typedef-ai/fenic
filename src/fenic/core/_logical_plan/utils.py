from __future__ import annotations

from typing import Optional

from fenic._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._resolved_session_config import (
    ResolvedGoogleModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import ValidationError
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DataType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    JsonType,
    MarkdownType,
    StringType,
    StructType,
    TranscriptType,
    _HtmlType,
    _PrimitiveType,
)


def validate_completion_parameters(
    model_alias: Optional[str],
    resolved_session_config: ResolvedSessionConfig,
    temperature: float,
    max_tokens: Optional[int] = None,
):
    """Validates that the provided temperature and max_tokens are within the limits allowed by the specified language model.

    If no model alias is provided, the session's default language model is used.

    Parameters:
        model_alias (Optional[str]):
            Alias of the language model to validate. Defaults to the session's
            default if not provided.
        resolved_session_config (ResolvedSessionConfig):
            The resolved session config containing model definitions.
        temperature (float):
            Sampling temperature. Must be within the model's supported range.
        max_tokens (Optional[int]):
            Maximum number of tokens to generate. Must not exceed the model's limit.

    Raises:
        ValidationError: If temperature or max_tokens are out of bounds for the model.
    """
    if model_alias is None:
        model_alias = resolved_session_config.semantic.default_language_model
    if model_alias not in resolved_session_config.semantic.language_models:
        raise ValidationError(
            f"Language model alias '{model_alias}' not found in SessionConfig. "
            f"Available models: {', '.join(resolved_session_config.semantic.language_models.keys()) or 'none'}"
        )
    model_config = resolved_session_config.semantic.language_models[model_alias]
    if isinstance(model_config, ResolvedOpenAIModelConfig):
        model_provider = ModelProvider.OPENAI
    elif isinstance(model_config, ResolvedGoogleModelConfig):
        model_provider = ModelProvider.GOOGLE_GLA
    else:
        model_provider = ModelProvider.ANTHROPIC
    completion_parameters: CompletionModelParameters = model_catalog.get_completion_model_parameters(model_provider, model_config.model_name)
    if max_tokens is not None and max_tokens > completion_parameters.max_output_tokens:
        raise ValidationError(f"[{model_provider.value}:{model_config.model_name}] max_output_tokens must be a positive integer less than or equal to {completion_parameters.max_output_tokens}")
    if temperature is not None and (temperature < 0 or temperature > completion_parameters.max_temperature):
        raise ValidationError(f"[{model_provider.value}:{model_config.model_name}] temperature must be between 0 and {completion_parameters.max_temperature}")

UNIMPLEMENTED_TYPES = (_HtmlType, TranscriptType, DocumentPathType)
def can_cast(src: DataType, dst: DataType) -> bool:
    if type(src) in UNIMPLEMENTED_TYPES or type(dst) in UNIMPLEMENTED_TYPES:
        raise NotImplementedError(f"Unimplemented type: Cannot cast {src} → {dst}")

    if isinstance(src, EmbeddingType):
        return NotImplementedError(f"Unimplemented type: Cannot cast {src} → {dst}")

    if (src == ArrayType(element_type=FloatType) or src == ArrayType(element_type=DoubleType)) and isinstance(dst, EmbeddingType):
        return True

    if src == dst:
        return True

    if dst == MarkdownType:
        return can_cast(src, StringType)

    if src == MarkdownType:
        return can_cast(StringType, dst)

    if dst == JsonType or src == JsonType:
        return True

    if isinstance(src, _PrimitiveType) and isinstance(dst, _PrimitiveType):
        # Disallow string → bool
        if src == StringType and dst == BooleanType:
            return False
        return True

    if isinstance(src, ArrayType) and isinstance(dst, ArrayType):
        return can_cast(src.element_type, dst.element_type)

    if isinstance(src, StructType) and isinstance(dst, StructType):
        src_fields = {f.name: f.data_type for f in src.struct_fields}
        dst_fields = {f.name: f.data_type for f in dst.struct_fields}
        for name, dst_type in dst_fields.items():
            if name in src_fields and not can_cast(src_fields[name], dst_type):
                return False
        return True

    return False
