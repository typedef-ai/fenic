"""Types used to configure model selection for semantic functions."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ModelAlias(BaseModel):
    """A combination of a model name and an optional profile for that model.

    Model aliases are used to select a specific model to use in a semantic operation.

    Attributes:
        name: The name of the model.
        profile: The optional name of a profile configuration to use for the model.

    Example:
        ```python
        model_alias = ModelAlias(name="o4-mini", profile="low")
        ```
    """

    name: str
    profile: Optional[str] = None


    @classmethod
    def from_str(cls, model_alias: str) -> ModelAlias:
        """Create a ModelAlias from a string.

        Args:
            model_alias: The string to create a ModelAlias from.

        Returns:
            ModelAlias: The created ModelAlias.
        """
        return cls(name=model_alias)

    def __str__(self) -> str:
        """Return the string representation of the ModelAlias.

        Returns:
            str: The string representation of the ModelAlias.
        """
        return f"{self.name}{f':{self.profile}' if self.profile else ''}"