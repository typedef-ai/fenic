import json
from typing import Any, Dict

import pytest

from fenic._backends.schema_serde import (
    _deserialize_data_type,
    deserialize_schema,
    serialize_data_type,
    serialize_schema,
)
from fenic.core.error import InternalError
from fenic.core.types import (
    ArrayType,
    BooleanType,
    ColumnField,
    DataType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    HtmlType,
    IntegerType,
    JsonType,
    MarkdownType,
    Schema,
    StringType,
    StructField,
    StructType,
    TranscriptType,
)


def create_sample_schema() -> Schema:
    """Helper function to create a more complex sample schema for testing."""
    return Schema(
        column_fields=[
            ColumnField(name="name", data_type=StringType),
            ColumnField(name="age", data_type=IntegerType),
            ColumnField(name="height", data_type=FloatType),
            ColumnField(name="is_active", data_type=BooleanType),
            ColumnField(name="tags", data_type=ArrayType(element_type=StringType)),
            ColumnField(
                name="address",
                data_type=StructType(
                    struct_fields=[
                        StructField(name="street", data_type=StringType),
                        StructField(name="city", data_type=StringType),
                        StructField(
                            name="zip_code",
                            data_type=StructType(  # Nested struct
                                struct_fields=[
                                    StructField(name="main", data_type=IntegerType),
                                    StructField(
                                        name="extension", data_type=IntegerType
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ColumnField(
                name="embedding",
                data_type=EmbeddingType(
                    dimensions=384, embedding_model="text-embedding-3-small"
                ),
            ),
            ColumnField(name="markdown_text", data_type=MarkdownType),
            ColumnField(name="html_content", data_type=HtmlType),
            ColumnField(name="json_data", data_type=JsonType),
            ColumnField(name="transcript", data_type=TranscriptType(format="srt")),
            ColumnField(name="document_path", data_type=DocumentPathType(format="pdf")),
            ColumnField(
                name="related_items",
                data_type=ArrayType(  # Array of structs
                    element_type=StructType(
                        struct_fields=[
                            StructField(name="id", data_type=IntegerType),
                            StructField(name="title", data_type=StringType),
                        ]
                    )
                ),
            ),
            ColumnField(
                name="nested_array",
                data_type=ArrayType(
                    element_type=ArrayType(element_type=IntegerType)  # Nested array
                ),
            ),
        ]
    )


# --- Unit Tests for _serialize_data_type and _deserialize_data_type ---


@pytest.mark.parametrize(
    "data_type, expected_serialization",
    [
        (StringType, {"type": "StringType"}),
        (IntegerType, {"type": "IntegerType"}),
        (FloatType, {"type": "FloatType"}),
        (DoubleType, {"type": "DoubleType"}),
        (BooleanType, {"type": "BooleanType"}),
        (
            ArrayType(element_type=StringType),
            {"type": "ArrayType", "element_type": {"type": "StringType"}},
        ),
        (
            StructType(
                struct_fields=[
                    StructField(name="name", data_type=StringType),
                    StructField(name="age", data_type=IntegerType),
                ]
            ),
            {
                "type": "StructType",
                "struct_fields": [
                    {"name": "name", "data_type": {"type": "StringType"}},
                    {"name": "age", "data_type": {"type": "IntegerType"}},
                ],
            },
        ),
        (
            EmbeddingType(dimensions=128, embedding_model="my-model"),
            {"type": "EmbeddingType", "dimensions": 128, "embedding_model": "my-model"},
        ),
        (MarkdownType, {"type": "MarkdownType"}),
        (HtmlType, {"type": "HtmlType"}),
        (JsonType, {"type": "JsonType"}),
        (TranscriptType(format="srt"), {"type": "TranscriptType", "format": "srt"}),
        (DocumentPathType(format="pdf"), {"type": "DocumentPathType", "format": "pdf"}),
    ],
)
def test_serialize_deserialize_data_type(
    data_type: DataType, expected_serialization: Dict[str, Any]
) -> None:
    """Tests serialization and deserialization of DataType instances."""
    serialized = serialize_data_type(data_type)
    assert serialized == expected_serialization
    deserialized = _deserialize_data_type(serialized)
    assert deserialized == data_type


# --- Unit Tests for serialize_schema and deserialize_schema ---


def test_serialize_deserialize_schema() -> None:
    """Tests serialization and deserialization of Schema instances."""
    sample_schema = create_sample_schema()
    serialized_schema = serialize_schema(sample_schema)
    deserialized_schema = deserialize_schema(serialized_schema)
    assert deserialized_schema == sample_schema


def test_deserialize_schema_invalid_json() -> None:
    """Tests that deserialize_schema raises an error for invalid JSON."""
    with pytest.raises(json.JSONDecodeError):
        deserialize_schema("invalid json string")


def test_deserialize_schema_missing_column_fields() -> None:
    """Tests deserialization with missing 'column_fields' key."""
    # This should not raise an error, but return a Schema with empty column_fields.
    deserialized_schema = deserialize_schema("{}")
    assert deserialized_schema == Schema(column_fields=[])

    deserialized_schema_2 = deserialize_schema('{"other_key": "value"}')
    assert deserialized_schema_2 == Schema(column_fields=[])


# --- Additional Tests for Robustness ---


def test_deserialize_data_type_unknown_type() -> None:
    """Tests that _deserialize_data_type raises an error for an unknown type."""
    with pytest.raises(ValueError, match="Unknown data type: UnknownType"):
        _deserialize_data_type({"type": "UnknownType"})


def test_deserialize_data_type_missing_type() -> None:
    """Tests that _deserialize_data_type raises an error for missing 'type'."""
    with pytest.raises(ValueError, match="Missing 'type' key"):
        _deserialize_data_type({})


def test_serialize_unsupported_data_type() -> None:
    """Tests that _serialize_data_type raises an error for an unsupported type."""

    class UnsupportedType(DataType):  # Create a dummy unsupported type for the test
        def __str__(self) -> str:
            return "UnsupportedType"

        def __eq__(self, other: object) -> bool:
            return isinstance(other, UnsupportedType)

    unsupported_instance = UnsupportedType()
    with pytest.raises(InternalError, match="Unsupported data type:"):
        serialize_data_type(unsupported_instance)
