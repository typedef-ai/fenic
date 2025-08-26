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
from fenic.core._serde.proto.types import DataTypeProto
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
    _BooleanType,
    _DoubleType,
    _FloatType,
    _HtmlType,
    _IntegerType,
    _JsonType,
    _MarkdownType,
    _StringType,
)

# Define examples for each datatype
# Each type has a list of examples to test different scenarios
datatype_examples = {
    # Basic types
    _StringType: [StringType],
    _IntegerType: [IntegerType],
    _FloatType: [FloatType],
    _DoubleType: [DoubleType],
    _BooleanType: [BooleanType],
    _JsonType: [JsonType],
    _MarkdownType: [MarkdownType],
    _HtmlType: [HtmlType],

    # Complex types with parameters
    ArrayType: [
        ArrayType(element_type=StringType),
        ArrayType(element_type=IntegerType),
        ArrayType(element_type=ArrayType(element_type=StringType)),
    ],
    StructType: [
        StructType(
            struct_fields=[
                StructField(name="field1", data_type=StringType),
                StructField(name="field2", data_type=IntegerType),
            ]
        ),
        StructType(
            struct_fields=[
                StructField(name="nested", data_type=StructType(
                    struct_fields=[StructField(name="inner", data_type=BooleanType)]
                )),
            ]
        ),
    ],
    EmbeddingType: [
        EmbeddingType(dimensions=768, embedding_model="text-embedding-ada-002"),
        EmbeddingType(dimensions=1536, embedding_model="text-embedding-3-small"),
    ],
    TranscriptType: [
        TranscriptType(format="srt"),
        TranscriptType(format="webvtt"),
    ],
    DocumentPathType: [
        DocumentPathType(format="pdf"),
    ],
}


class TestDataTypeSerde:
    """Test cases for data type serialization and deserialization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = SerdeContext()

    def _compare_datatypes(self, original, deserialized, datatype_class_name: str, example_index: int):
        """Compare key attributes of original and deserialized datatypes."""
        if not original == deserialized:
            raise ValueError(f"Original {original} does not match deserialized {deserialized}. Class Name: {datatype_class_name}, Example Index: {example_index}")

    @pytest.mark.parametrize("datatype_class", datatype_examples.keys())
    def test_all_datatype_types_with_examples(self, datatype_class):
        """Test all registered datatype types with comprehensive examples."""

        # Get the class name for error messages
        class_name = type(datatype_class).__name__ if hasattr(datatype_class, '__name__') else type(datatype_class).__name__

        # Test each datatype type with its examples
        for i, example in enumerate(datatype_examples[datatype_class]):
            try:
                # Serialize the datatype
                serialized = serialize_data_type(example, self.context)
                assert serialized is not None, (
                    f"Serialization failed for {class_name} example {i}"
                )

                # Deserialize the datatype
                deserialized = deserialize_data_type(serialized, self.context)
                assert deserialized is not None, (
                    f"Deserialization failed for {class_name} example {i}"
                )

                # Basic type check
                assert isinstance(deserialized, type(example)), (
                    f"Deserialized type mismatch for {class_name} example {i}"
                )

                # Compare datatypes using the helper method
                self._compare_datatypes(example, deserialized, class_name, i)

            except Exception as e:
                pytest.fail(
                    f"Serde failed for {class_name} example {i}: {e}"
                )

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

    def test_complex_nested_structures(self):
        """Test complex nested datatype structures."""
        # Create a complex nested structure
        complex_struct = StructType(
            struct_fields=[
                StructField(
                    name="strings",
                    data_type=ArrayType(element_type=StringType)
                ),
                StructField(
                    name="numbers",
                    data_type=ArrayType(element_type=IntegerType)
                ),
                StructField(
                    name="nested",
                    data_type=StructType(
                        struct_fields=[
                            StructField(name="inner", data_type=BooleanType),
                            StructField(
                                name="deep_array",
                                data_type=ArrayType(element_type=ArrayType(element_type=StringType))
                            ),
                        ]
                    ),
                ),
            ]
        )

        # Serialize and deserialize
        serialized = serialize_data_type(complex_struct, self.context)
        deserialized = deserialize_data_type(serialized, self.context)

        # Should maintain structure
        assert isinstance(deserialized, StructType)
        assert len(deserialized.struct_fields) == 3

        # Check first field (array of strings)
        assert deserialized.struct_fields[0].name == "strings"
        assert isinstance(deserialized.struct_fields[0].data_type, ArrayType)
        assert deserialized.struct_fields[0].data_type.element_type == StringType

        # Check second field (array of integers)
        assert deserialized.struct_fields[1].name == "numbers"
        assert isinstance(deserialized.struct_fields[1].data_type, ArrayType)
        assert deserialized.struct_fields[1].data_type.element_type == IntegerType

        # Check third field (nested struct)
        assert deserialized.struct_fields[2].name == "nested"
        assert isinstance(deserialized.struct_fields[2].data_type, StructType)
        assert len(deserialized.struct_fields[2].data_type.struct_fields) == 2
        assert deserialized.struct_fields[2].data_type.struct_fields[0].name == "inner"
        assert deserialized.struct_fields[2].data_type.struct_fields[0].data_type == BooleanType
        assert deserialized.struct_fields[2].data_type.struct_fields[1].name == "deep_array"
        assert isinstance(deserialized.struct_fields[2].data_type.struct_fields[1].data_type, ArrayType)
        assert isinstance(deserialized.struct_fields[2].data_type.struct_fields[1].data_type.element_type, ArrayType)
        assert deserialized.struct_fields[2].data_type.struct_fields[1].data_type.element_type.element_type == StringType

        # Verify equality
        assert complex_struct == deserialized

    def test_all_datatype_subclasses_covered(self):
        """Test that all concrete datatype classes are covered in the test file."""
        import importlib
        import inspect

        from fenic.core.types.datatypes import DataType

        # Find all concrete DataType subclasses
        concrete_subclasses = set()
        try:
            module = importlib.import_module("fenic.core.types.datatypes")
            for _name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, DataType) and
                    obj != DataType and
                    not inspect.isabstract(obj)):
                    concrete_subclasses.add(obj.__name__)
        except ImportError:
            pass

        # Get all tested datatype classes from the datatype_examples dictionary
        tested_classes = set(cls.__name__ for cls in datatype_examples.keys())

        # Find missing classes
        missing = concrete_subclasses - tested_classes

        if missing:
            pytest.fail(
                f"Missing {len(missing)} concrete DataType subclasses from tests: {sorted(missing)}. "
                f"Add them to the datatype_examples dictionary in this test file."
            )

        # Optional: Check for extra classes (not DataType subclasses)
        extra = tested_classes - concrete_subclasses
        if extra:
            print(f"Warning: {len(extra)} tested classes are not concrete DataType subclasses: {sorted(extra)}")

        # Verify coverage
        coverage = len(concrete_subclasses - missing) / len(concrete_subclasses) * 100
        assert coverage == 100.0, f"Datatype coverage is {coverage:.1f}%, expected 100%"
