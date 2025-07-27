"""Types used to configure model selection for semantic functions."""
from typing import Optional

from pydantic import BaseModel


class ModelAlias(BaseModel):
    """A combination of a model name and an optional preset for that model.

    Model aliases are used to select a specific model to use in a semantic operation.

    Args:
        name: The name of the model.
        preset: The optional name of a preset configuration to use for the model.

    Example:
        ```python
        model_alias = ModelAlias(name="o4-mini", preset="low")
        ```
    """

    name: str
    preset: Optional[str] = None
