"""Tests for enum serialization/deserialization using real enums from the codebase."""

from enum import Enum

import pytest

from fenic.core._logical_plan.expressions.base import Operator
from fenic.core._logical_plan.expressions.text import (
    ChunkCharacterSet,
    ChunkLengthFunction,
)
from fenic.core._serde.proto.enum_serde import (
    deserialize_enum_value,
    serialize_enum_value,
)
from fenic.core._serde.proto.errors import DeserializationError, SerializationError
from fenic.core._serde.proto.serde_context import SerdeContext
from fenic.core._serde.proto.types import (
    ChunkCharacterSetProto,
    ChunkLengthFunctionProto,
    OperatorProto,
)


class TestEnumSerde:
    """Test cases for enum serialization and deserialization using real enums."""

    def setup_method(self):
        """Set up test fixtures."""
        self.context = SerdeContext()

    def test_serialize_enum_value_not_found(self):
        """Test serialization of enum value not found in proto raises error."""

        # Create a test enum that doesn't match the proto
        class TestEnum(Enum):
            UNKNOWN_VALUE = "unknown"

        # Get the actual proto enum wrapper
        proto_enum = ChunkLengthFunctionProto

        with pytest.raises(SerializationError) as exc_info:
            serialize_enum_value(TestEnum.UNKNOWN_VALUE, proto_enum, self.context)
        assert "Enum value" in str(exc_info.value)
        assert "does not have a corresponding protobuf value" in str(exc_info.value)

    def test_deserialize_enum_value_not_found(self):
        """Test deserialization of enum value not found in target enum raises error."""
        # Get the actual proto enum wrapper
        proto_enum = ChunkLengthFunctionProto

        # Create a test enum with different values than the proto
        class TestEnum(Enum):
            DIFFERENT_VALUE = "different"

        with pytest.raises(DeserializationError) as exc_info:
            deserialize_enum_value(TestEnum, proto_enum, 0, self.context)
        assert "Protobuf enum name" in str(exc_info.value)
        assert "is not present in" in str(exc_info.value)


    def test_deserialize_enum_value_invalid_proto_value(self):
        """Test deserialization with invalid proto value raises error."""
        # Get the actual proto enum wrapper
        proto_enum = ChunkLengthFunctionProto

        # Try to deserialize a value that doesn't exist in the proto
        with pytest.raises(DeserializationError) as exc_info:
            deserialize_enum_value(ChunkLengthFunction, proto_enum, 999, self.context)
        assert "is not present in" in str(exc_info.value)

    def test_round_trip_serialization(self):
        """Test round-trip serialization and deserialization."""
        # Get the actual proto enum wrapper
        proto_enum = ChunkLengthFunctionProto

        # Test round-trip for all values
        for enum_value in ChunkLengthFunction:
            serialized = serialize_enum_value(enum_value, proto_enum, self.context)
            deserialized = deserialize_enum_value(
                ChunkLengthFunction, proto_enum, serialized, self.context
            )
            assert deserialized == enum_value

    def test_round_trip_serialization_character_set(self):
        """Test round-trip serialization and deserialization for ChunkCharacterSet."""
        # Get the actual proto enum wrapper
        proto_enum = ChunkCharacterSetProto

        # Test round-trip for all values
        for enum_value in ChunkCharacterSet:
            serialized = serialize_enum_value(enum_value, proto_enum, self.context)
            deserialized = deserialize_enum_value(
                ChunkCharacterSet, proto_enum, serialized, self.context
            )
            assert deserialized == enum_value

    def test_round_trip_serialization_operator(self):
        """Test round-trip serialization and deserialization for Operator."""
        # Get the actual proto enum wrapper
        proto_enum = OperatorProto

        # Test round-trip for all values
        for enum_value in Operator:
            serialized = serialize_enum_value(enum_value, proto_enum, self.context)
            deserialized = deserialize_enum_value(
                Operator, proto_enum, serialized, self.context
            )
            assert deserialized == enum_value

    def test_serialize_enum_value_case_sensitive(self):
        """Test that enum serialization is case sensitive."""
        # Get the actual proto enum wrapper
        proto_enum = ChunkLengthFunctionProto

        # Create an enum with different case
        class CaseSensitiveEnum(Enum):
            character = "character"  # Different case

        with pytest.raises(SerializationError) as exc_info:
            serialize_enum_value(CaseSensitiveEnum.character, proto_enum, self.context)
        assert "does not have a corresponding protobuf value" in str(exc_info.value)

    def test_deserialize_enum_value_case_sensitive(self):
        """Test that enum deserialization is case sensitive."""
        # Get the actual proto enum wrapper
        proto_enum = ChunkLengthFunctionProto

        # Create an enum with different case
        class CaseSensitiveEnum(Enum):
            character = "character"  # Different case

        with pytest.raises(DeserializationError) as exc_info:
            deserialize_enum_value(CaseSensitiveEnum, proto_enum, ChunkLengthFunctionProto.CHARACTER, self.context)
        assert "is not present in" in str(exc_info.value)
