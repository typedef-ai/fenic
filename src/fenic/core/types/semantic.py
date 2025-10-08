"""Types used to configure model selection for semantic functions."""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel

from fenic.core._logical_plan.resolved_types import ResolvedModelAlias

ParsingEngine = Literal["mistral-ocr", "pdf-text", "native"]

class ModelAlias(BaseModel):
    """A combination of a model name and a required profile for that model.

    Model aliases are used to select a specific model to use in a semantic operation.
    Both the model name and profile must be specified when creating a ModelAlias.

    Attributes:
        name: The name of the model.
        profile: The name of a profile configuration to use for the model.

    Example:
        ```python
        model_alias = ModelAlias(name="o4-mini", profile="low")
        ```
    """

    name: str
    profile: str

def _resolve_model_alias(model_alias: Optional[Union[str, ModelAlias]]) -> Optional[ResolvedModelAlias]:
    """Convert a model alias from the API layer to the expression layer format.

    Args:
        model_alias: Either a string, a ModelAlias, or None

    Returns:
        A ResolvedModelAlias with optional profile, or None
    """
    if model_alias is None:
        return None

    if isinstance(model_alias, str):
        return ResolvedModelAlias(name=model_alias)

    # It's a ModelAlias, convert to ResolvedModelAlias
    return ResolvedModelAlias(name=model_alias.name, profile=model_alias.profile)