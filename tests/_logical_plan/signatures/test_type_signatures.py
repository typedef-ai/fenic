"""
Test TypeSignature implementations.

This tests all the TypeSignature classes that validate argument types.
"""

import pytest

from fenic.core._logical_plan.signatures.type_signature import (
    ArrayOfAny,
    ArrayWithMatchingElement,
    EqualTypes,
    Exact,
    InstanceOf,
    Numeric,
    OneOf,
    Uniform,
    VariadicAny,
    VariadicUniform,
)
from fenic.core.error import InternalError, TypeMismatchError, ValidationError
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    EmbeddingType,
    FloatType,
    IntegerType,
    StringType,
)


class TestExact:
    """Test Exact signature type."""

    def test_validates_exact_argument_count_and_types(self):
        sig = Exact([StringType, IntegerType])

        # Should accept correct types
        sig.validate([StringType, IntegerType], "test_func")

        # Should reject wrong argument count
        with pytest.raises(InternalError, match="test_func expects 2 arguments, got 1"):
            sig.validate([StringType], "test_func")

        with pytest.raises(InternalError, match="test_func expects 2 arguments, got 3"):
            sig.validate([StringType, IntegerType, BooleanType], "test_func")

        # Should reject wrong types
        with pytest.raises(TypeMismatchError, match="test_func Argument 0: expected StringType, got IntegerType"):
            sig.validate([IntegerType, IntegerType], "test_func")

        with pytest.raises(TypeMismatchError, match="test_func Argument 1: expected IntegerType, got StringType"):
            sig.validate([StringType, StringType], "test_func")


class TestVariadicUniform:
    """Test VariadicUniform signature type."""

    def test_requires_minimum_arguments(self):
        sig = VariadicUniform(expected_min_args=2)

        # Should accept minimum or more
        sig.validate([StringType, StringType], "test_func")
        sig.validate([StringType, StringType, StringType], "test_func")

        # Should reject fewer than minimum
        with pytest.raises(ValidationError, match="test_func expects at least 2 arguments"):
            sig.validate([StringType], "test_func")

    def test_uniform_type_validation_with_required_type(self):
        sig = VariadicUniform(expected_min_args=1, required_type=StringType)

        # Should accept all strings
        sig.validate([StringType], "test_func")
        sig.validate([StringType, StringType, StringType], "test_func")

        # Should reject wrong types
        with pytest.raises(TypeMismatchError, match="test_func Argument 0: expected StringType, got IntegerType"):
            sig.validate([IntegerType], "test_func")

        with pytest.raises(TypeMismatchError, match="test_func expects all arguments to have the same type"):
            sig.validate([StringType, IntegerType], "test_func")

    def test_uniform_type_validation_same_as_first(self):
        sig = VariadicUniform(expected_min_args=1)  # No required_type = same as first

        # Should accept all same type
        sig.validate([StringType, StringType], "test_func")
        sig.validate([IntegerType, IntegerType, IntegerType], "test_func")

        # Should reject mixed types
        with pytest.raises(TypeMismatchError, match="test_func expects all arguments to have the same type"):
            sig.validate([StringType, IntegerType], "test_func")


class TestVariadicAny:
    """Test VariadicAny signature type."""

    def test_accepts_any_types_and_counts(self):
        sig = VariadicAny(expected_min_args=1)

        # Should accept any types
        sig.validate([StringType], "test_func")
        sig.validate([StringType, IntegerType], "test_func")
        sig.validate([StringType, IntegerType, BooleanType], "test_func")

        # Should still enforce minimum count
        with pytest.raises(ValidationError, match="test_func expects at least 1 arguments, got 0"):
            sig.validate([], "test_func")


class TestArrayOfAny:
    """Test ArrayOfAny signature type."""

    def test_accepts_only_array_types(self):
        sig = ArrayOfAny()

        # Should accept any array type
        sig.validate([ArrayType(StringType)], "test_func")
        sig.validate([ArrayType(IntegerType)], "test_func")
        sig.validate([ArrayType(BooleanType)], "test_func")

        # Should reject non-array types
        with pytest.raises(TypeMismatchError, match="test_func expects argument 0 to be an array type"):
            sig.validate([StringType], "test_func")

        with pytest.raises(TypeMismatchError, match="test_func expects argument 0 to be an array type"):
            sig.validate([IntegerType], "test_func")

    def test_validates_argument_count(self):
        sig = ArrayOfAny(expected_num_args=2)

        # Should require exactly 2 arguments
        sig.validate([ArrayType(StringType), ArrayType(IntegerType)], "test_func")

        with pytest.raises(ValidationError, match="test_func expects 2 arguments, got 1"):
            sig.validate([ArrayType(StringType)], "test_func")


class TestArrayWithMatchingElement:
    """Test ArrayWithMatchingElement signature type."""

    def test_validates_array_and_matching_element(self):
        sig = ArrayWithMatchingElement()

        # Should accept array + matching element
        sig.validate([ArrayType(StringType), StringType], "test_func")
        sig.validate([ArrayType(IntegerType), IntegerType], "test_func")
        sig.validate([ArrayType(BooleanType), BooleanType], "test_func")

        # Should reject non-array first argument
        with pytest.raises(TypeMismatchError, match="test_func expects argument 0 to be an array type"):
            sig.validate([StringType, StringType], "test_func")

        # Should reject mismatched element type
        with pytest.raises(TypeMismatchError, match="test_func Argument 1: expected StringType, got IntegerType"):
            sig.validate([ArrayType(StringType), IntegerType], "test_func")

    def test_requires_exactly_two_arguments(self):
        sig = ArrayWithMatchingElement()

        with pytest.raises(ValidationError, match=r"test_func expects 2 arguments \(array, element\), got 1"):
            sig.validate([ArrayType(StringType)], "test_func")

        with pytest.raises(ValidationError, match=r"test_func expects 2 arguments \(array, element\), got 3"):
            sig.validate([ArrayType(StringType), StringType, StringType], "test_func")


class TestNumeric:
    """Test Numeric signature type."""

    def test_accepts_numeric_types(self):
        sig = Numeric(1)

        # Should accept numeric types
        sig.validate([IntegerType], "test_func")
        sig.validate([FloatType], "test_func")

        # Should reject non-numeric types
        with pytest.raises(TypeMismatchError, match="test_func expects numeric type for argument 0"):
            sig.validate([StringType], "test_func")

        with pytest.raises(TypeMismatchError, match="test_func expects numeric type for argument 0"):
            sig.validate([BooleanType], "test_func")

    def test_validates_argument_count(self):
        sig = Numeric(2)

        # Should require exactly 2 arguments
        sig.validate([IntegerType, FloatType], "test_func")

        with pytest.raises(InternalError, match="test_func expects 2 arguments, got 1"):
            sig.validate([IntegerType], "test_func")

class TestUniform:
    """Test Uniform signature type."""

    def test_validates_exact_count_and_uniform_types(self):
        sig = Uniform(3)

        # Should accept same types
        sig.validate([StringType, StringType, StringType], "test_func")
        sig.validate([IntegerType, IntegerType, IntegerType], "test_func")

        # Should reject wrong count
        with pytest.raises(InternalError, match="test_func expects 3 arguments, got 2"):
            sig.validate([StringType, StringType], "test_func")

        # Should reject mixed types
        with pytest.raises(TypeMismatchError, match="test_func expects all arguments to have the same type"):
            sig.validate([StringType, IntegerType, StringType], "test_func")

    def test_uniform_with_required_type(self):
        sig = Uniform(2, required_type=StringType)

        # Should accept required type
        sig.validate([StringType, StringType], "test_func")

        # Should reject wrong type
        with pytest.raises(TypeMismatchError, match="test_func Argument 0: expected StringType, got IntegerType"):
            sig.validate([IntegerType, IntegerType], "test_func")


class TestOneOf:
    """Test OneOf signature type."""

    def test_matches_any_alternative_signature(self):
        sig = OneOf([
            Exact([StringType]),
            Exact([IntegerType, IntegerType])
        ])

        # Should accept first alternative
        sig.validate([StringType], "test_func")

        # Should accept second alternative
        sig.validate([IntegerType, IntegerType], "test_func")

        # Should reject if no alternatives match
        with pytest.raises(TypeMismatchError, match="test_func does not match any valid signature"):
            sig.validate([BooleanType], "test_func")

        with pytest.raises(TypeMismatchError, match="test_func does not match any valid signature"):
            sig.validate([StringType, StringType], "test_func")


class TestEqualTypes:
    """Test EqualTypes signature type."""

    def test_validates_equal_types(self):
        sig = EqualTypes(ArrayType)

        # Should accept two arrays with same element type
        array_int = ArrayType(IntegerType)
        sig.validate([array_int, array_int], "test_func")

        # Should accept two arrays with same structure but different instances
        array_int2 = ArrayType(IntegerType)
        sig.validate([array_int, array_int2], "test_func")

        # Should reject arrays with different element types
        array_str = ArrayType(StringType)
        with pytest.raises(TypeMismatchError, match="test_func expects both arguments to be equal"):
            sig.validate([array_int, array_str], "test_func")

    def test_validates_instance_type(self):
        sig = EqualTypes(ArrayType)

        # Should reject non-array types
        with pytest.raises(TypeMismatchError, match="test_func expects argument 0 to be an instance of ArrayType"):
            sig.validate([StringType, ArrayType(IntegerType)], "test_func")

        with pytest.raises(TypeMismatchError, match="test_func expects argument 1 to be an instance of ArrayType"):
            sig.validate([ArrayType(IntegerType), StringType], "test_func")

    def test_requires_exactly_two_arguments(self):
        sig = EqualTypes(ArrayType)

        with pytest.raises(InternalError, match="test_func expects 2 arguments, got 1"):
            sig.validate([ArrayType(IntegerType)], "test_func")

        with pytest.raises(InternalError, match="test_func expects 2 arguments, got 3"):
            sig.validate([ArrayType(IntegerType), ArrayType(IntegerType), ArrayType(IntegerType)], "test_func")

    def test_works_with_embedding_type(self):
        sig = EqualTypes(EmbeddingType)

        # Should accept embeddings with same dimensions
        emb1 = EmbeddingType(dimensions=128, embedding_model="test")
        emb2 = EmbeddingType(dimensions=128, embedding_model="test")
        sig.validate([emb1, emb2], "test_func")

        # Should reject embeddings with different dimensions
        emb3 = EmbeddingType(dimensions=256, embedding_model="test")
        with pytest.raises(TypeMismatchError, match="test_func expects both arguments to be equal"):
            sig.validate([emb1, emb3], "test_func")


class TestInstanceOf:
    """Test InstanceOf signature type."""

    def test_validates_instance_types(self):
        sig = InstanceOf([ArrayType, type(StringType)])

        # Should accept correct types
        sig.validate([ArrayType(IntegerType), StringType], "test_func")

        # Should reject wrong types
        with pytest.raises(TypeMismatchError, match="test_func expects argument 0 to be an instance of ArrayType"):
            sig.validate([StringType, StringType], "test_func")

        with pytest.raises(TypeMismatchError, match="test_func expects argument 1 to be an instance of _StringType"):
            sig.validate([ArrayType(IntegerType), IntegerType], "test_func")

    def test_validates_argument_count(self):
        sig = InstanceOf([ArrayType, type(StringType), type(IntegerType)])

        # Should accept exact count
        sig.validate([ArrayType(StringType), StringType, IntegerType], "test_func")

        # Should reject wrong count
        with pytest.raises(InternalError, match="test_func expects 3 arguments, got 2"):
            sig.validate([ArrayType(StringType), StringType], "test_func")

        with pytest.raises(InternalError, match="test_func expects 3 arguments, got 4"):
            sig.validate([ArrayType(StringType), StringType, IntegerType, BooleanType], "test_func")

    def test_works_with_single_type(self):
        sig = InstanceOf([EmbeddingType])

        # Should accept embedding type
        emb = EmbeddingType(dimensions=128, embedding_model="test")
        sig.validate([emb], "test_func")

        # Should reject non-embedding type
        with pytest.raises(TypeMismatchError, match="test_func expects argument 0 to be an instance of EmbeddingType"):
            sig.validate([StringType], "test_func")
