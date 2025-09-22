"""Data type serialization/deserialization using singledispatch."""

from functools import singledispatch

from google.protobuf.message import Message

from fenic.core._serde.proto.errors import (
    DeserializationError,
    SerializationError,
)
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    ArrayTypeProto,
    BooleanTypeProto,
    DataTypeProto,
    DateTypeProto,
    DocumentPathTypeProto,
    DoubleTypeProto,
    EmbeddingTypeProto,
    FloatTypeProto,
    HTMLTypeProto,
    IntegerTypeProto,
    JSONTypeProto,
    MarkdownTypeProto,
    StringTypeProto,
    StructFieldProto,
    StructTypeProto,
    TimestampTypeProto,
    TranscriptTypeProto,
)
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    DataType,
    DateType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    HtmlType,
    IntegerType,
    JsonType,
    MarkdownType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    TranscriptType,
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

# =============================================================================
# Top-level functions
# =============================================================================


@singledispatch
def serialize_data_type(data_type: DataType, context: SerdeContext) -> DataTypeProto:
    """Serialize a data type to its protobuf representation.

    This function uses singledispatch to handle different data type classes.
    Each data type class should have a corresponding register function that
    implements the specific serialization logic.

    Args:
        data_type: The data type to serialize. Must be a registered type.
        context: The serde context for error reporting and path tracking.

    Returns:
        DataTypeProto: The serialized protobuf representation of the data type.

    Raises:
        SerializationError: If the data type is not registered or serialization fails.
    """
    raise context.create_serde_error(
        SerializationError,
        f"Serialization not implemented for DataType: {type(data_type)}",
        type(data_type),
    )


def deserialize_data_type(
    data_type_proto: DataTypeProto,
    context: SerdeContext,
) -> DataType:
    """Deserialize a data type from its protobuf representation.

    This function determines which oneof field is set in the DataTypeProto
    and delegates to the appropriate deserialization helper function.

    Args:
        data_type_proto: The protobuf representation to deserialize.
        context: The serde context for error reporting and path tracking.

    Returns:
        DataType: The deserialized data type.

    Raises:
        DeserializationError: If the protobuf is empty or deserialization fails.
    """
    which_oneof = data_type_proto.WhichOneof("data_type")
    if which_oneof is None:
        return None # Indicates a None DataType was passed in for serialization, which is allowed in certain cases.
    underlying_proto = getattr(data_type_proto, which_oneof)
    return _deserialize_data_type_helper(underlying_proto, context)


@singledispatch
def _deserialize_data_type_helper(
    underlying_proto: Message, context: SerdeContext
) -> DataType:
    """Deserialize a data type from its underlying protobuf message.

    This function uses singledispatch to handle different protobuf message types.
    Each protobuf message type should have a corresponding register function that
    implements the specific deserialization logic.

    Args:
        underlying_proto: The underlying protobuf message to deserialize.
        context: The serde context for error reporting and path tracking.

    Returns:
        DataType: The deserialized data type.

    Raises:
        DeserializationError: If the protobuf type is not registered or deserialization fails.
    """
    raise context.create_serde_error(
        DeserializationError,
        f"Deserialization not implemented for Data Type: {type(underlying_proto)}",
        type(underlying_proto),
    )


# =============================================================================
# StringType
# =============================================================================


@serialize_data_type.register
def _serialize_string_type(_: _StringType, context: SerdeContext) -> DataTypeProto:
    """Serialize a string type."""
    return DataTypeProto(string=StringTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_string_type(_: StringTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a string type."""
    return StringType


# =============================================================================
# IntegerType
# =============================================================================


@serialize_data_type.register
def _serialize_integer_type(_: _IntegerType, context: SerdeContext) -> DataTypeProto:
    """Serialize an integer type."""
    return DataTypeProto(integer=IntegerTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_integer_type(_: IntegerTypeProto, context: SerdeContext) -> DataType:
    """Deserialize an integer type."""
    return IntegerType


# =============================================================================
# FloatType
# =============================================================================


@serialize_data_type.register
def _serialize_float_type(_: _FloatType, context: SerdeContext) -> DataTypeProto:
    """Serialize a float type."""
    return DataTypeProto(float=FloatTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_float_type(_: FloatTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a float type."""
    return FloatType


# =============================================================================
# DoubleType
# =============================================================================


@serialize_data_type.register
def _serialize_double_type(_: _DoubleType, context: SerdeContext) -> DataTypeProto:
    """Serialize a double type."""
    return DataTypeProto(double=DoubleTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_double_type(_: DoubleTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a double type."""
    return DoubleType


# =============================================================================
# BooleanType
# =============================================================================


@serialize_data_type.register
def _serialize_boolean_type(_: _BooleanType, context: SerdeContext) -> DataTypeProto:
    """Serialize a boolean type."""
    return DataTypeProto(boolean=BooleanTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_boolean_type(_: BooleanTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a boolean type."""
    return BooleanType


# =============================================================================
# ArrayType
# =============================================================================


@serialize_data_type.register
def _serialize_array_type(
    array_type: ArrayType, context: SerdeContext
) -> DataTypeProto:
    """Serialize an array type."""
    element_type_proto = context.serialize_data_type(
        "element_type", array_type.element_type
    )
    return DataTypeProto(array=ArrayTypeProto(element_type=element_type_proto))


@_deserialize_data_type_helper.register
def _deserialize_array_type(
    array_proto: ArrayTypeProto, context: SerdeContext
) -> DataType:
    """Deserialize an array type."""
    element_type = context.deserialize_data_type(
        "element_type", array_proto.element_type
    )
    return ArrayType(element_type=element_type)


# =============================================================================
# StructType
# =============================================================================


@serialize_data_type.register
def _serialize_struct_type(
    struct_type: StructType, context: SerdeContext
) -> DataTypeProto:
    """Serialize a struct type."""
    struct_fields = []
    for i, field in enumerate(struct_type.struct_fields):
        with context.path_context(f"struct_fields[{i}]"):
            field_proto = StructFieldProto(
                name=field.name,
                data_type=context.serialize_data_type("data_type", field.data_type),
            )
            struct_fields.append(field_proto)
    return DataTypeProto(struct=StructTypeProto(fields=struct_fields))


@_deserialize_data_type_helper.register
def _deserialize_struct_type(
    struct_proto: StructTypeProto, context: SerdeContext
) -> DataType:
    """Deserialize a struct type."""
    struct_fields = []
    for i, field_proto in enumerate(struct_proto.fields):
        with context.path_context(f"fields[{i}]"):
            field_data_type = context.deserialize_data_type(
                "data_type", field_proto.data_type
            )
            struct_field = StructField(name=field_proto.name, data_type=field_data_type)
            struct_fields.append(struct_field)
    return StructType(struct_fields=struct_fields)


# =============================================================================
# EmbeddingType
# =============================================================================


@serialize_data_type.register
def _serialize_embedding_type(
    embedding_type: EmbeddingType, context: SerdeContext
) -> DataTypeProto:
    """Serialize an embedding type."""
    return DataTypeProto(
        embedding=EmbeddingTypeProto(
            dimensions=embedding_type.dimensions,
            embedding_model=embedding_type.embedding_model,
        )
    )


@_deserialize_data_type_helper.register
def _deserialize_embedding_type(
    embedding_proto: EmbeddingTypeProto, context: SerdeContext
) -> DataType:
    """Deserialize an embedding type."""
    return EmbeddingType(
        dimensions=embedding_proto.dimensions,
        embedding_model=embedding_proto.embedding_model,
    )


# =============================================================================
# TranscriptType
# =============================================================================


@serialize_data_type.register
def _serialize_transcript_type(
    transcript_type: TranscriptType, context: SerdeContext
) -> DataTypeProto:
    """Serialize a transcript type."""
    return DataTypeProto(transcript=TranscriptTypeProto(format=transcript_type.format))


@_deserialize_data_type_helper.register
def _deserialize_transcript_type(
    transcript_proto: TranscriptTypeProto, context: SerdeContext
) -> DataType:
    """Deserialize a transcript type."""
    return TranscriptType(format=transcript_proto.format)


# =============================================================================
# DocumentPathType
# =============================================================================


@serialize_data_type.register
def _serialize_document_path_type(
    document_path_type: DocumentPathType, context: SerdeContext
) -> DataTypeProto:
    """Serialize a document path type."""
    return DataTypeProto(
        document_path=DocumentPathTypeProto(format=document_path_type.format)
    )


@_deserialize_data_type_helper.register
def _deserialize_document_path_type(
    document_path_proto: DocumentPathTypeProto, context: SerdeContext
) -> DataType:
    """Deserialize a document path type."""
    return DocumentPathType(format=document_path_proto.format)


# =============================================================================
# MarkdownType
# =============================================================================


@serialize_data_type.register
def _serialize_markdown_type(_: _MarkdownType, context: SerdeContext) -> DataTypeProto:
    """Serialize a markdown type."""
    return DataTypeProto(markdown=MarkdownTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_markdown_type(_: MarkdownTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a markdown type."""
    return MarkdownType


# =============================================================================
# HtmlType
# =============================================================================


@serialize_data_type.register
def _serialize_html_type(_: _HtmlType, context: SerdeContext) -> DataTypeProto:
    """Serialize an HTML type."""
    return DataTypeProto(html=HTMLTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_html_type(_: HTMLTypeProto, context: SerdeContext) -> DataType:
    """Deserialize an HTML type."""
    return HtmlType


# =============================================================================
# JsonType
# =============================================================================


@serialize_data_type.register
def _serialize_json_type(_: _JsonType, context: SerdeContext) -> DataTypeProto:
    """Serialize a JSON type."""
    return DataTypeProto(json=JSONTypeProto())


@_deserialize_data_type_helper.register
def _deserialize_json_type(_: JSONTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a JSON type."""
    return JsonType

# =============================================================================
# DateType
# =============================================================================

@serialize_data_type.register
def _serialize_date_type(_: _DateType, context: SerdeContext) -> DataTypeProto:
    """Serialize a date type."""
    return DataTypeProto(date=DateTypeProto())

@_deserialize_data_type_helper.register
def _deserialize_date_type(_: DateTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a date type."""
    return DateType

# =============================================================================
# TimestampType
# =============================================================================

@serialize_data_type.register
def _serialize_timestamp_type(ts: _TimestampType, context: SerdeContext) -> DataTypeProto:
    """Serialize a timestamp type."""
    return DataTypeProto(timestamp=TimestampTypeProto())

@_deserialize_data_type_helper.register
def _deserialize_timestamp_type(ts_proto: TimestampTypeProto, context: SerdeContext) -> DataType:
    """Deserialize a timestamp type."""
    return TimestampType