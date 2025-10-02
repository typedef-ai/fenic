import json
from typing import Any, Dict, Type

from fenic.core.error import InternalError
from fenic.core.types import (
    ArrayType,
    ColumnField,
    DataType,
    DocumentPathType,
    EmbeddingType,
    Schema,
    StructField,
    StructType,
    TranscriptType,
)
from fenic.core.types.datatypes import (
    _BooleanType,
    _DateType,
    _DoubleType,
    _FloatType,
    _HtmlType,
    _IntegerType,
    _JsonType,
    _MarkdownType,
    _StringType,
    _TimestampType,
)

_TYPE_MAPPING: Dict[str, Type[DataType]] = {
    "StringType": _StringType,
    "IntegerType": _IntegerType,
    "FloatType": _FloatType,
    "DoubleType": _DoubleType,
    "BooleanType": _BooleanType,
    "DateType": _DateType,
    "TimestampType": _TimestampType,
    "ArrayType": ArrayType,
    "StructType": StructType,
    "StructField": StructField,
    "EmbeddingType": EmbeddingType,
    "MarkdownType": _MarkdownType,
    "HtmlType": _HtmlType,
    "JsonType": _JsonType,
    "TranscriptType": TranscriptType,
    "DocumentPathType": DocumentPathType,
}

_REVERSE_TYPE_MAPPING: Dict[Type[DataType], str] = {
    v: k for k, v in _TYPE_MAPPING.items()
}


def serialize_schema(schema: Schema) -> str:
    """Serializes a Schema instance to a JSON string.

    This function converts a Schema object into a JSON string representation,
    allowing it to be stored or transmitted as text. The serialization preserves
    all column fields and their data types.

    Args:
        schema: The Schema instance to serialize.

    Returns:
        A JSON string representation of the schema.

    Raises:
        ValueError: If the schema contains unsupported data types.
    """
    serialized_fields = [
        _serialize_column_field(field) for field in schema.column_fields
    ]
    return json.dumps({"column_fields": serialized_fields})


def deserialize_schema(json_string: str) -> Schema:
    """Deserializes a JSON string into a Schema instance.

    This function reconstructs a Schema object from its JSON string representation,
    restoring all column fields with their original data types.

    Args:
        json_string: A JSON string previously created by serialize_schema().

    Returns:
        A Schema instance reconstructed from the JSON string.

    Raises:
        ValueError: If the JSON string contains unknown data types or is malformed.
        json.JSONDecodeError: If the input string is not valid JSON.
    """
    data = json.loads(json_string)
    column_fields_data = data.get("column_fields", [])
    column_fields = [
        _deserialize_column_field(field_data) for field_data in column_fields_data
    ]
    return Schema(column_fields=column_fields)


def serialize_data_type(data_type: DataType) -> Dict[str, Any]:
    """Convert a DataType instance into a JSON-serializable dictionary.

    This function recursively serializes complex data types (such as arrays,
    structs, and custom types) into a dictionary format suitable for JSON encoding.
    It is used for persisting or transmitting schema information across systems.

    Supported types:
    - Primitive types (e.g., IntegerType, StringType)
    - ArrayType: Includes nested element type
    - StructType: Includes a list of named fields with associated types
    - StructField: Includes field name and type
    - EmbeddingType: Includes model name and vector dimensions
    - TranscriptType, DocumentPathType: Includes format information

    Args:
        data_type (DataType): The type to serialize.

    Returns:
        Dict[str, Any]: A JSON-compatible dictionary representation of the data type.

    Raises:
        InternalError: If the given type is not supported or unrecognized.
    """
    type_name = _REVERSE_TYPE_MAPPING.get(type(data_type))
    if not type_name:
        raise InternalError(f"Unsupported data type: {data_type}")

    serialized: Dict[str, Any] = {"type": type_name}
    if isinstance(data_type, ArrayType):
        serialized["element_type"] = serialize_data_type(data_type.element_type)
    elif isinstance(data_type, StructType):
        serialized["struct_fields"] = [
            _serialize_struct_field(field) for field in data_type.struct_fields
        ]
    elif isinstance(data_type, StructField):
        serialized["name"] = data_type.name
        serialized["data_type"] = serialize_data_type(data_type.data_type)
    elif isinstance(data_type, EmbeddingType):
        serialized["dimensions"] = data_type.dimensions
        serialized["embedding_model"] = data_type.embedding_model
    elif isinstance(data_type, TranscriptType):
        serialized["format"] = data_type.format
    elif isinstance(data_type, DocumentPathType):
        serialized["format"] = data_type.format
    return serialized


def _deserialize_data_type(data: Dict[str, Any]) -> DataType:
    """Deserializes a dictionary into a DataType instance."""
    type_name = data.get("type")
    if not type_name:
        raise ValueError("Missing 'type' key in data type serialization.")

    data_type_cls = _TYPE_MAPPING.get(type_name)
    if not data_type_cls:
        raise ValueError(f"Unknown data type: {type_name}")

    if type_name == "ArrayType":
        return ArrayType(element_type=_deserialize_data_type(data["element_type"]))
    elif type_name == "StructType":
        return StructType(
            struct_fields=[_deserialize_struct_field(f) for f in data["struct_fields"]]
        )
    elif type_name == "StructField":
        return StructField(
            name=data["name"], data_type=_deserialize_data_type(data["data_type"])
        )
    elif type_name == "EmbeddingType":
        return EmbeddingType(
            dimensions=data["dimensions"], embedding_model=data["embedding_model"]
        )
    elif type_name == "TranscriptType":
        return TranscriptType(format=data["format"])
    elif type_name == "DocumentPathType":
        return DocumentPathType(format=data["format"])
    else:
        return data_type_cls()


def _serialize_struct_field(field: StructField) -> Dict[str, Any]:
    """Serializes a StructField instance to a JSON-compatible dictionary."""
    return {"name": field.name, "data_type": serialize_data_type(field.data_type)}


def _deserialize_struct_field(data: Dict[str, Any]) -> StructField:
    """Deserializes a dictionary into a StructField instance."""
    return StructField(
        name=data["name"], data_type=_deserialize_data_type(data["data_type"])
    )


def _serialize_column_field(column_field: ColumnField) -> Dict[str, Any]:
    """Serializes a ColumnField instance to a JSON-compatible dictionary."""
    return {
        "name": column_field.name,
        "data_type": serialize_data_type(column_field.data_type),
    }


def _deserialize_column_field(data: Dict[str, Any]) -> ColumnField:
    """Deserializes a dictionary into a ColumnField instance."""
    return ColumnField(
        name=data["name"], data_type=_deserialize_data_type(data["data_type"])
    )
