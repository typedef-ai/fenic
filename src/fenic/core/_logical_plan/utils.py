from typing import Optional, Tuple

from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    ModelProvider,
    model_catalog,
)
from fenic.core._logical_plan.expressions import AggregateExpr, LogicalExpr, SortExpr
from fenic.core._logical_plan.resolved_types import ResolvedModelAlias
from fenic.core._resolved_session_config import (
    ResolvedGoogleModelConfig,
    ResolvedModelConfig,
    ResolvedOpenAIModelConfig,
    ResolvedOpenRouterModelConfig,
    ResolvedSessionConfig,
)
from fenic.core.error import PlanError, ValidationError


def parse_model_alias(model_alias: str) -> Tuple[str, Optional[str]]:
    """Parse a model alias to extract base model and profile name.

    Args:
        model_alias: Model alias in format 'model' or 'model.profile'

    Returns:
        Tuple of (base_model_alias, profile_name)
        If no dot present, profile_name will be None
    """
    if "." in model_alias:
        parts = model_alias.split(".", 1)  # Split on first dot only
        return parts[0], parts[1]
    else:
        return model_alias, None

def fetch_model_and_completion_parameters(
    model_alias: Optional[ResolvedModelAlias],
    resolved_session_config: ResolvedSessionConfig,
) -> Tuple[ResolvedModelConfig, ModelProvider, CompletionModelParameters]:
    """Fetches the model, provider, and completion parameters for a given model alias for parameter validation.

    If no model alias is provided, the session's default language model is used.
    """
   # Check if any language models are configured
    if not resolved_session_config.semantic.language_models:
        raise ValidationError(
            "No language models configured. This operation requires language models. "
            "Please add language_models to your SemanticConfig."
        )
    language_model_config = resolved_session_config.semantic.language_models
    model_alias_name = model_alias.name if model_alias else language_model_config.default_model

    if model_alias_name not in language_model_config.model_configs:
        available_models = list(language_model_config.model_configs.keys())
        raise ValidationError(
            f"Language model alias '{model_alias_name}' not found in SessionConfig. "
            f"Available models: {', '.join(available_models)}"
        )

    model_config = language_model_config.model_configs[model_alias_name]
    if isinstance(model_config, ResolvedOpenAIModelConfig):
        model_provider = ModelProvider.OPENAI
    elif isinstance(model_config, ResolvedGoogleModelConfig):
        model_provider = model_config.model_provider
    elif isinstance(model_config, ResolvedOpenRouterModelConfig):
        model_provider = ModelProvider.OPENROUTER
    else:
        model_provider = ModelProvider.ANTHROPIC
    return model_config, model_provider, model_catalog.get_completion_model_parameters(model_provider, model_config.model_name)

def validate_completion_parameters(
    model_alias: Optional[ResolvedModelAlias],
    resolved_session_config: ResolvedSessionConfig,
    temperature: float,
    max_tokens: Optional[int] = None,
):
    """Validates that the provided temperature and max_tokens are within the limits allowed by the specified language model.

    Parameters:
        model_alias (Optional[ResolvedModelAlias]):
            ResolvedModelAlias object containing model name and optional profile.
            Defaults to the session's default if not provided.
        resolved_session_config (ResolvedSessionConfig):
            The resolved session config containing model definitions.
        temperature (float):
            Sampling temperature. Must be within the model's supported range.
        max_tokens (Optional[int]):
            Maximum number of tokens to generate. Must not exceed the model's limit.

    Raises:
        ValidationError: If temperature or max_tokens are out of bounds for the model.
    """
    model_config, model_provider, completion_parameters = fetch_model_and_completion_parameters(model_alias, resolved_session_config)
    if max_tokens is not None and max_tokens > completion_parameters.max_output_tokens:
        raise ValidationError(
            f"[{model_provider.value}:{model_config.model_name}] max_output_tokens must be a positive integer less than or equal to {completion_parameters.max_output_tokens}"
        )
    if temperature is not None and (temperature < 0 or temperature > completion_parameters.max_temperature):
        raise ValidationError(
            f"[{model_provider.value}:{model_config.model_name}] temperature must be between 0 and {completion_parameters.max_temperature}"
        )


def validate_model_support_for_pdf_parsing(model_alias: Optional[ResolvedModelAlias], resolved_session_config: ResolvedSessionConfig):
    """Validates that the model supports PDF parsing.

    Parameters:
        model_alias (Optional[ResolvedModelAlias]):
            ResolvedModelAlias object containing model name and optional profile.
            Defaults to the session's default if not provided.
        resolved_session_config (ResolvedSessionConfig):
            The resolved session config containing model definitions.
    """
    model_config, model_provider, completion_parameters = fetch_model_and_completion_parameters(model_alias, resolved_session_config)

    if not completion_parameters.supports_pdf_parsing:
        raise ValidationError(f"fenic does not support parsing with [{model_provider.value}:{model_config.model_name}]. Supported models: {model_catalog.get_models_for_criteria(lambda params: params.supports_pdf_parsing)}")

def validate_scalar_expr(expr: LogicalExpr, function_name: str):
    if isinstance(expr, SortExpr):
        raise PlanError(
            f"Sort expressions are not allowed in `{function_name}`. "
            "Please use the sort() method instead."
        )
    if isinstance(expr, AggregateExpr):
        raise PlanError(
            f"Aggregate expressions are not allowed in {function_name}. "
            "Please use the agg() method instead."
        )
