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
    # Check if any language models are configured
    if not resolved_session_config.semantic.language_models:
        raise ValidationError(
            "No language models configured. This operation requires language models. "
            "Please add language_models to your SemanticConfig."
        )
    language_model_config = resolved_session_config.semantic.language_models
    if model_alias is None:
        model_alias = language_model_config.default_model

    if model_alias not in language_model_config.model_configs:
        available_models = list(language_model_config.model_configs.keys())
        raise ValidationError(
            f"Language model alias '{model_alias}' not found in SessionConfig. "
            f"Available models: {', '.join(available_models)}"
        )

    model_config = language_model_config.model_configs[model_alias]
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
