"""Tests for datatype serialization/deserialization."""

import pytest
from google.protobuf.message import Message

from fenic.core._serde.proto.datatype_serde import (
    _deserialize_data_type_helper,
    deserialize_data_type,
    serialize_data_type,
)
from fenic.core._serde.proto.errors import DeserializationError, SerializationError
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    ArrayTypeProto,
    BooleanTypeProto,
    DataTypeProto,
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
    TranscriptTypeProto,
)
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
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
    TranscriptType,
)


class TestDataTypeSerde:
    """Test cases for data type serialization and deserialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = SerdeContext()

    def test_serialize_string_type(self):
        """Test serialization of StringType."""
        result = serialize_data_type(StringType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("string")
        assert isinstance(result.string, StringTypeProto)

    def test_deserialize_string_type(self):
        """Test deserialization of StringType."""
        proto = DataTypeProto(string=StringTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == StringType

    def test_serialize_integer_type(self):
        """Test serialization of IntegerType."""
        result = serialize_data_type(IntegerType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("integer")
        assert isinstance(result.integer, IntegerTypeProto)

    def test_deserialize_integer_type(self):
        """Test deserialization of IntegerType."""
        proto = DataTypeProto(integer=IntegerTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == IntegerType

    def test_serialize_float_type(self):
        """Test serialization of FloatType."""
        result = serialize_data_type(FloatType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("float")
        assert isinstance(result.float, FloatTypeProto)

    def test_deserialize_float_type(self):
        """Test deserialization of FloatType."""
        proto = DataTypeProto(float=FloatTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == FloatType

    def test_serialize_double_type(self):
        """Test serialization of DoubleType."""
        result = serialize_data_type(DoubleType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("double")
        assert isinstance(result.double, DoubleTypeProto)

    def test_deserialize_double_type(self):
        """Test deserialization of DoubleType."""
        proto = DataTypeProto(double=DoubleTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == DoubleType

    def test_serialize_boolean_type(self):
        """Test serialization of BooleanType."""
        result = serialize_data_type(BooleanType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("boolean")
        assert isinstance(result.boolean, BooleanTypeProto)

    def test_deserialize_boolean_type(self):
        """Test deserialization of BooleanType."""
        proto = DataTypeProto(boolean=BooleanTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == BooleanType

    def test_serialize_array_type(self):
        """Test serialization of ArrayType."""
        array_type = ArrayType(element_type=StringType)
        result = serialize_data_type(array_type, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("array")
        assert isinstance(result.array, ArrayTypeProto)
        assert result.array.HasField("element_type")
        assert result.array.element_type.HasField("string")

    def test_deserialize_array_type(self):
        """Test deserialization of ArrayType."""
        element_proto = DataTypeProto(string=StringTypeProto())
        array_proto = DataTypeProto(array=ArrayTypeProto(element_type=element_proto))
        result = deserialize_data_type(array_proto, self.context)
        assert isinstance(result, ArrayType)
        assert result.element_type == StringType

    def test_serialize_struct_type(self):
        """Test serialization of StructType."""
        struct_type = StructType(
            struct_fields=[
                StructField(name="field1", data_type=StringType),
                StructField(name="field2", data_type=IntegerType),
            ]
        )
        result = serialize_data_type(struct_type, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("struct")
        assert isinstance(result.struct, StructTypeProto)
        assert len(result.struct.fields) == 2
        assert result.struct.fields[0].name == "field1"
        assert result.struct.fields[0].data_type.HasField("string")
        assert result.struct.fields[1].name == "field2"
        assert result.struct.fields[1].data_type.HasField("integer")

    def test_deserialize_struct_type(self):
        """Test deserialization of StructType."""
        field1_proto = StructFieldProto(
            name="field1", data_type=DataTypeProto(string=StringTypeProto())
        )
        field2_proto = StructFieldProto(
            name="field2", data_type=DataTypeProto(integer=IntegerTypeProto())
        )
        struct_proto = DataTypeProto(
            struct=StructTypeProto(fields=[field1_proto, field2_proto])
        )
        result = deserialize_data_type(struct_proto, self.context)
        assert isinstance(result, StructType)
        assert len(result.struct_fields) == 2
        assert result.struct_fields[0].name == "field1"
        assert result.struct_fields[0].data_type == StringType
        assert result.struct_fields[1].name == "field2"
        assert result.struct_fields[1].data_type == IntegerType

    def test_serialize_embedding_type(self):
        """Test serialization of EmbeddingType."""
        embedding_type = EmbeddingType(
            dimensions=768, embedding_model="text-embedding-ada-002"
        )
        result = serialize_data_type(embedding_type, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("embedding")
        assert isinstance(result.embedding, EmbeddingTypeProto)
        assert result.embedding.dimensions == 768
        assert result.embedding.embedding_model == "text-embedding-ada-002"

    def test_deserialize_embedding_type(self):
        """Test deserialization of EmbeddingType."""
        embedding_proto = DataTypeProto(
            embedding=EmbeddingTypeProto(
                dimensions=768, embedding_model="text-embedding-ada-002"
            )
        )
        result = deserialize_data_type(embedding_proto, self.context)
        assert isinstance(result, EmbeddingType)
        assert result.dimensions == 768
        assert result.embedding_model == "text-embedding-ada-002"

    def test_serialize_transcript_type(self):
        """Test serialization of TranscriptType."""
        transcript_type = TranscriptType(format="srt")
        result = serialize_data_type(transcript_type, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("transcript")
        assert isinstance(result.transcript, TranscriptTypeProto)
        assert result.transcript.format == "srt"

    def test_deserialize_transcript_type(self):
        """Test deserialization of TranscriptType."""
        transcript_proto = DataTypeProto(transcript=TranscriptTypeProto(format="srt"))
        result = deserialize_data_type(transcript_proto, self.context)
        assert isinstance(result, TranscriptType)
        assert result.format == "srt"

    def test_serialize_document_path_type(self):
        """Test serialization of DocumentPathType."""
        document_path_type = DocumentPathType(format="pdf")
        result = serialize_data_type(document_path_type, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("document_path")
        assert isinstance(result.document_path, DocumentPathTypeProto)
        assert result.document_path.format == "pdf"

    def test_deserialize_document_path_type(self):
        """Test deserialization of DocumentPathType."""
        document_path_proto = DataTypeProto(
            document_path=DocumentPathTypeProto(format="pdf")
        )
        result = deserialize_data_type(document_path_proto, self.context)
        assert isinstance(result, DocumentPathType)
        assert result.format == "pdf"

    def test_serialize_markdown_type(self):
        """Test serialization of MarkdownType."""
        result = serialize_data_type(MarkdownType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("markdown")
        assert isinstance(result.markdown, MarkdownTypeProto)

    def test_deserialize_markdown_type(self):
        """Test deserialization of MarkdownType."""
        proto = DataTypeProto(markdown=MarkdownTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == MarkdownType

    def test_serialize_html_type(self):
        """Test serialization of HtmlType."""
        result = serialize_data_type(HtmlType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("html")
        assert isinstance(result.html, HTMLTypeProto)

    def test_deserialize_html_type(self):
        """Test deserialization of HtmlType."""
        proto = DataTypeProto(html=HTMLTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == HtmlType

    def test_serialize_json_type(self):
        """Test serialization of JsonType."""
        result = serialize_data_type(JsonType, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("json")
        assert isinstance(result.json, JSONTypeProto)

    def test_deserialize_json_type(self):
        """Test deserialization of JsonType."""
        proto = DataTypeProto(json=JSONTypeProto())
        result = deserialize_data_type(proto, self.context)
        assert result == JsonType

    def test_serialize_unregistered_type(self):
        """Test serialization of an unregistered type raises error."""

        class UnregisteredType:
            pass

        with pytest.raises(SerializationError) as exc_info:
            serialize_data_type(UnregisteredType(), self.context)
        assert "Serialization not implemented for" in str(exc_info.value)

    def test_deserialize_unknown_proto(self):
        """Test deserialization of an unknown proto type raises error."""

        class UnknownProto(Message):
            pass

        with pytest.raises(DeserializationError) as exc_info:
            _deserialize_data_type_helper(UnknownProto(), self.context)
        assert "Deserialization not implemented for" in str(exc_info.value)

    def test_deserialize_empty_proto(self):
        """Test deserialization of an empty DataTypeProto returns None."""
        empty_proto = DataTypeProto()
        result = deserialize_data_type(empty_proto, self.context)
        assert result is None

    def test_nested_array_serialization(self):
        """Test serialization of nested array types."""
        inner_array = ArrayType(element_type=StringType)
        outer_array = ArrayType(element_type=inner_array)
        result = serialize_data_type(outer_array, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("array")
        assert result.array.HasField("element_type")
        assert result.array.element_type.HasField("array")
        assert result.array.element_type.array.HasField("element_type")
        assert result.array.element_type.array.element_type.HasField("string")

    def test_nested_array_deserialization(self):
        """Test deserialization of nested array types."""
        inner_element_proto = DataTypeProto(string=StringTypeProto())
        inner_array_proto = DataTypeProto(
            array=ArrayTypeProto(element_type=inner_element_proto)
        )
        outer_array_proto = DataTypeProto(
            array=ArrayTypeProto(element_type=inner_array_proto)
        )
        result = deserialize_data_type(outer_array_proto, self.context)
        assert isinstance(result, ArrayType)
        assert isinstance(result.element_type, ArrayType)
        assert result.element_type.element_type == StringType

    def test_complex_struct_serialization(self):
        """Test serialization of complex struct with nested types."""
        struct_type = StructType(
            struct_fields=[
                StructField(
                    name="strings", data_type=ArrayType(element_type=StringType)
                ),
                StructField(
                    name="numbers", data_type=ArrayType(element_type=IntegerType)
                ),
                StructField(
                    name="nested",
                    data_type=StructType(
                        struct_fields=[StructField(name="inner", data_type=BooleanType)]
                    ),
                ),
            ]
        )
        result = serialize_data_type(struct_type, self.context)
        assert isinstance(result, DataTypeProto)
        assert result.HasField("struct")
        assert len(result.struct.fields) == 3
        # Check first field (array of strings)
        assert result.struct.fields[0].name == "strings"
        assert result.struct.fields[0].data_type.HasField("array")
        assert result.struct.fields[0].data_type.array.element_type.HasField("string")
        # Check second field (array of integers)
        assert result.struct.fields[1].name == "numbers"
        assert result.struct.fields[1].data_type.HasField("array")
        assert result.struct.fields[1].data_type.array.element_type.HasField("integer")
        # Check third field (nested struct)
        assert result.struct.fields[2].name == "nested"
        assert result.struct.fields[2].data_type.HasField("struct")
        assert len(result.struct.fields[2].data_type.struct.fields) == 1
        assert result.struct.fields[2].data_type.struct.fields[0].name == "inner"
        assert (
            result.struct.fields[2]
            .data_type.struct.fields[0]
            .data_type.HasField("boolean")
        )

    def test_complex_struct_deserialization(self):
        """Test deserialization of complex struct with nested types."""
        # Build the complex proto structure
        inner_element_proto = DataTypeProto(string=StringTypeProto())
        strings_array_proto = DataTypeProto(
            array=ArrayTypeProto(element_type=inner_element_proto)
        )

        numbers_element_proto = DataTypeProto(integer=IntegerTypeProto())
        numbers_array_proto = DataTypeProto(
            array=ArrayTypeProto(element_type=numbers_element_proto)
        )

        inner_field_proto = StructFieldProto(
            name="inner", data_type=DataTypeProto(boolean=BooleanTypeProto())
        )
        nested_struct_proto = DataTypeProto(
            struct=StructTypeProto(fields=[inner_field_proto])
        )

        field1_proto = StructFieldProto(name="strings", data_type=strings_array_proto)
        field2_proto = StructFieldProto(name="numbers", data_type=numbers_array_proto)
        field3_proto = StructFieldProto(name="nested", data_type=nested_struct_proto)

        struct_proto = DataTypeProto(
            struct=StructTypeProto(fields=[field1_proto, field2_proto, field3_proto])
        )

        result = deserialize_data_type(struct_proto, self.context)
        assert isinstance(result, StructType)
        assert len(result.struct_fields) == 3

        # Check first field
        assert result.struct_fields[0].name == "strings"
        assert isinstance(result.struct_fields[0].data_type, ArrayType)
        assert result.struct_fields[0].data_type.element_type == StringType

        # Check second field
        assert result.struct_fields[1].name == "numbers"
        assert isinstance(result.struct_fields[1].data_type, ArrayType)
        assert result.struct_fields[1].data_type.element_type == IntegerType

        # Check third field
        assert result.struct_fields[2].name == "nested"
        assert isinstance(result.struct_fields[2].data_type, StructType)
        assert len(result.struct_fields[2].data_type.struct_fields) == 1
        assert result.struct_fields[2].data_type.struct_fields[0].name == "inner"
        assert (
            result.struct_fields[2].data_type.struct_fields[0].data_type == BooleanType
        )
