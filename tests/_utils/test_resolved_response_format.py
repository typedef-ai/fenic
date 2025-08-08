

from typing import Literal

from pydantic import BaseModel, Field

from fenic.core._logical_plan.resolved_types import ResolvedResponseFormat
from fenic.core._utils.structured_outputs import (
    convert_pydantic_model_to_key_descriptions,
)


class NestedModel(BaseModel):
    nested_str: str = Field(..., description="A nested string.")
    nested_literal_str: Literal["a", "b", "c"] = Field(..., description="A nested string literal.")

class PydanticModelWithDescriptions(BaseModel):
    """A simple model for boolean answers."""

    output: bool = Field(..., description="The boolean answer to the question posed by the user.")
    test_str: str = Field(..., description="A test string.")
    test_nested: NestedModel = Field(..., description="A nested model.")
    test_array_nested: list[NestedModel] = Field(..., description="A list of nested models.")
    test_array_str: list[str] = Field(..., description="A list of strings.")
    test_str_literal: Literal["a", "b", "c"] = Field(..., description="A string literal.")

def test_resolved_response_format_eq():

    resolved_format_boolean =  ResolvedResponseFormat.from_pydantic_model(PydanticModelWithDescriptions)
    resolved_format_boolean_2 =  ResolvedResponseFormat.from_pydantic_model(PydanticModelWithDescriptions)
    assert resolved_format_boolean == resolved_format_boolean_2

    original_field_descriptions = convert_pydantic_model_to_key_descriptions(PydanticModelWithDescriptions)
    resolved_field_descriptions = resolved_format_boolean.prompt_schema_definition
    assert original_field_descriptions == resolved_field_descriptions