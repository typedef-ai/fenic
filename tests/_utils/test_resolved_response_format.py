

from typing import Literal

from json_schema_to_pydantic import create_model as create_pydantic_model
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
    test_int: int = Field(..., description="A test integer.", multiple_of=10, gt=0, lt=100)
    test_str: str = Field(..., description="A test string.", pattern=r"^[a-z]+$")
    test_nested: NestedModel = Field(..., description="A nested model.")
    test_array_nested: list[NestedModel] = Field(..., description="A list of nested models.", max_length=10)
    test_array_str: list[str] = Field(..., description="A list of strings.", max_length=10)
    test_str_literal: Literal["a", "b", "c"] = Field(..., description="A string literal.")

def test_resolved_response_format_eq():

    resolved_format_boolean =  ResolvedResponseFormat.from_pydantic_model(PydanticModelWithDescriptions)
    resolved_format_boolean_2 =  ResolvedResponseFormat.from_pydantic_model(PydanticModelWithDescriptions)
    assert resolved_format_boolean == resolved_format_boolean_2

    original_field_descriptions = convert_pydantic_model_to_key_descriptions(PydanticModelWithDescriptions)
    resolved_field_descriptions = resolved_format_boolean.prompt_schema_definition
    assert original_field_descriptions == resolved_field_descriptions

def test_resolved_response_format_from_json_schema():
    json_schema = PydanticModelWithDescriptions.model_json_schema()
    generated_model = create_pydantic_model(json_schema)
    assert generated_model.model_json_schema() == json_schema

    data_json = {
        "output": True,
        "test_int": 10,
        "test_str": "test",
        "test_nested": {
            "nested_str": "test",
            "nested_literal_str": "a"
        },
        "test_array_nested": [
            {
                "nested_str": "test",
                "nested_literal_str": "a"
            }
        ],
        "test_array_str": ["test"],
        "test_str_literal": "a",
    }
    test_original = PydanticModelWithDescriptions.model_validate(data_json)

    test_generated = generated_model.model_validate(data_json)

    assert test_original.output == test_generated.output
    assert test_original.test_int == test_generated.test_int
    assert test_original.test_str == test_generated.test_str
    assert test_original.test_nested.nested_str == test_generated.test_nested.nested_str
    assert test_original.test_nested.nested_literal_str == test_generated.test_nested.nested_literal_str
    assert test_original.test_array_str == test_generated.test_array_str
    assert test_original.test_str_literal == test_generated.test_str_literal