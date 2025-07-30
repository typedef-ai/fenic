"""Types used to configure model selection for semantic functions."""
from __future__ import annotations

from pydantic import BaseModel


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