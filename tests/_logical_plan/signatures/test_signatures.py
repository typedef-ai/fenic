"""
Test TypeSignature implementations.

This tests all the TypeSignature classes that validate argument types.
"""

import pytest

from fenic.core._logical_plan.signatures.types import (
    ArrayOfAny,
    ArrayWithMatchingElement,
    Exact,
    Numeric,
    OneOf,
    StructWithStringKey,
    Uniform,
    VariadicAny,
    VariadicUniform,
)
from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types.datatypes import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


class TestExact:
    """Test Exact signature type."""
    
    def test_validates_exact_argument_count_and_types(self):
        sig = Exact([StringType, IntegerType])
        
        # Should accept correct types
        sig.validate([StringType, IntegerType], "test_func")
        
        # Should reject wrong argument count
        with pytest.raises(ValidationError, match="test_func expects 2 arguments, got 1"):
            sig.validate([StringType], "test_func")
        
        with pytest.raises(ValidationError, match="test_func expects 2 arguments, got 3"):
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
        
        with pytest.raises(ValidationError, match="test_func expects 2 arguments, got 1"):
            sig.validate([IntegerType], "test_func")

class TestUniform:
    """Test Uniform signature type."""
    
    def test_validates_exact_count_and_uniform_types(self):
        sig = Uniform(3)
        
        # Should accept same types
        sig.validate([StringType, StringType, StringType], "test_func")
        sig.validate([IntegerType, IntegerType, IntegerType], "test_func")
        
        # Should reject wrong count
        with pytest.raises(ValidationError, match="test_func expects 3 arguments, got 2"):
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


class TestStructWithStringKey:
    """Test StructWithStringKey signature type."""
    
    def test_validates_struct_and_string_key(self):
        sig = StructWithStringKey()
        struct_type = StructType([
            StructField("field1", StringType), 
            StructField("field2", IntegerType)
        ])
        
        # Should accept struct + string
        sig.validate([struct_type, StringType], "test_func")
        
        # Should reject non-struct first argument
        with pytest.raises(TypeMismatchError, match="test_func expects argument 0 to be a struct type"):
            sig.validate([StringType, StringType], "test_func")
        
        # Should reject non-string second argument
        with pytest.raises(TypeMismatchError, match="test_func Argument 1: expected StringType, got IntegerType"):
            sig.validate([struct_type, IntegerType], "test_func")
    
    def test_requires_exactly_two_arguments(self):
        sig = StructWithStringKey()
        struct_type = StructType([StructField("field1", StringType)])
        
        with pytest.raises(ValidationError, match=r"test_func expects 2 arguments \(struct, field_name\), got 1"):
            sig.validate([struct_type], "test_func")
        
        with pytest.raises(ValidationError, match=r"test_func expects 2 arguments \(struct, field_name\), got 3"):
            sig.validate([struct_type, StringType, StringType], "test_func")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])