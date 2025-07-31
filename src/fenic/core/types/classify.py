"""Definition of a ClassDefinition class with optional description for semantic.classify."""

from pydantic import BaseModel


class ClassDefinition(BaseModel):
    """Definition of a classification class with optional description.

    Used to define the available classes for semantic classification operations.
    The description helps the LLM understand what each class represents.
    """

    label: str
    description: str
