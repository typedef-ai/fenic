from __future__ import annotations

from typing import Literal, Optional

import pytest
from json_schema_to_pydantic import create_model as create_pydantic_model
from pydantic import BaseModel, Field, ValidationError

from fenic.core._utils.structured_outputs import validate_output_format


class Nested(BaseModel):
    inner: str = Field(..., description="Inner string")
    tag: Literal["a", "b"] = Field(..., description="Inner literal")


class PrimitivesAndOptional(BaseModel):
    a_bool: bool = Field(..., description="A boolean")
    a_int: int = Field(..., description="An integer")
    a_float: float = Field(..., description="A float")
    a_str: str = Field(..., description="A string")
    maybe_str: Optional[str] = Field(None, description="Optional string")
    a_lit: Literal["x", "y"] = Field(..., description="A literal")


class StringConstraints(BaseModel):
    s: str = Field(
        ...,
        description="Constrained string",
        min_length=3,
        max_length=5,
        pattern=r"^ab",
    )


class NumberConstraints(BaseModel):
    i: int = Field(..., description="Constrained int", ge=0, lt=10, multiple_of=2)
    f: float = Field(..., description="Constrained float", gt=0.5, le=1.5)


class ListConstraints(BaseModel):
    items: list[str] = Field(
        ...,
        description="Constrained list of strings",
        min_length=1,
        max_length=2,
    )
    nested: list[Nested] = Field(
        ...,
        description="Constrained list of nested",
        max_length=2,
    )


def _roundtrip(model: type[BaseModel]) -> type[BaseModel]:
    # Ensure the model is within our supported subset
    validate_output_format(model)
    # Round-trip via JSON Schema
    json_schema = model.model_json_schema()
    generated = create_pydantic_model(json_schema)
    assert generated.model_json_schema() == json_schema
    # Validate the generated model is also compliant with our subset
    validate_output_format(generated)
    return generated


def test_roundtrip_primitives_optional_literal():
    deserialized = _roundtrip(PrimitivesAndOptional)

    ok = deserialized.model_validate({
        "a_bool": True,
        "a_int": 1,
        "a_float": 1.0,
        "a_str": "hi",
        "maybe_str": None,
        "a_lit": "x",
    })
    assert ok.a_bool is True
    assert ok.maybe_str is None
    with pytest.raises(ValidationError):
        deserialized.model_validate({
            "a_bool": True,
            "a_int": 1,
            "a_float": 1.0,
            "a_str": "hi",
            "maybe_str": None,
            "a_lit": "z",
        })


def test_roundtrip_string_constraints():
    deserialized = _roundtrip(StringConstraints)
    assert deserialized.model_validate({"s": "abc"}).s == "abc"
    with pytest.raises(ValidationError):
        deserialized.model_validate({"s": "ab"})  # too short
    with pytest.raises(ValidationError):
        deserialized.model_validate({"s": "abcdef"})  # too long
    with pytest.raises(ValidationError):
        deserialized.model_validate({"s": "zzzzz"})  # pattern mismatch


def test_roundtrip_number_constraints():
    deserialized = _roundtrip(NumberConstraints)
    assert deserialized.model_validate({"i": 2, "f": 1.0}).i == 2
    with pytest.raises(ValidationError):
        deserialized.model_validate({"i": -2, "f": 1.0})  # ge
    with pytest.raises(ValidationError):
        deserialized.model_validate({"i": 10, "f": 1.0})  # lt
    with pytest.raises(ValidationError):
        deserialized.model_validate({"i": 3, "f": 1.0})  # multiple_of
    with pytest.raises(ValidationError):
        deserialized.model_validate({"i": 2, "f": 0.5})  # gt
    with pytest.raises(ValidationError):
        deserialized.model_validate({"i": 2, "f": 1.6})  # le


def test_roundtrip_list_constraints_and_nested():
    deserialized = _roundtrip(ListConstraints)

    ok = deserialized.model_validate({
        "items": ["a"],
        "nested": [{"inner": "v", "tag": "a"}],
    })
    assert ok.items == ["a"]
    assert ok.nested[0].inner == "v"

    with pytest.raises(ValidationError):
        deserialized.model_validate({"items": [], "nested": [{"inner": "v", "tag": "a"}]})  # min_items
    with pytest.raises(ValidationError):
        deserialized.model_validate({
            "items": ["a", "b", "c"],  # max_items
            "nested": [{"inner": "v", "tag": "a"}],
        })
    with pytest.raises(ValidationError):
        deserialized.model_validate({
            "items": ["a"],
            "nested": [
                {"inner": "v", "tag": "a"},
                {"inner": "w", "tag": "b"},
                {"inner": "u", "tag": "a"},
            ],  # max_items
        })


