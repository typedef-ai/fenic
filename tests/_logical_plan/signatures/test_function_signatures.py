"""
Test FunctionSignature class and return type inference.

This tests the complete function signature validation and return type strategies.
"""

import pytest

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.expressions.basic import ArrayLengthExpr, ColumnExpr
from fenic.core._logical_plan.signatures.function_signature import (
    FunctionSignature,
    ReturnTypeStrategy,
)
from fenic.core._logical_plan.signatures.type_signature import (
    Exact,
    Numeric,
    VariadicUniform,
)
from fenic.core.error import InternalError, TypeMismatchError
from fenic.core.types.datatypes import ArrayType, FloatType, IntegerType, StringType
from fenic.core.types.schema import ColumnField, Schema


# Mock classes for testing
class MockPlan:
    def __init__(self, column_fields=None):
        if column_fields is None:
            column_fields = []
        self._schema = Schema(column_fields=column_fields)

    def schema(self):
        return self._schema


class MockColumn(LogicalExpr):
    def __init__(self, name: str, data_type):
        self.name = name
        self.data_type = data_type

    def to_column_field(self, plan):
        return ColumnField(self.name, self.data_type)

    def children(self):
        return []

    def __str__(self):
        return f"col({self.name})"


class TestFunctionSignature:
    """Test FunctionSignature class."""

    def test_fixed_return_type(self):
        """Test function with fixed return type."""
        sig = FunctionSignature(function_name="array_size", type_signature=Exact([ArrayType(StringType)]),
                                return_type=IntegerType)

        # Should infer fixed return type
        return_type = sig.infer_return_type([ArrayType(StringType)])
        assert return_type == IntegerType

    def test_same_as_input_return_type(self):
        """Test SAME_AS_INPUT return type strategy."""
        sig = FunctionSignature(function_name="upper", type_signature=Exact([StringType]),
                                return_type=ReturnTypeStrategy.SAME_AS_INPUT)

        # Should return same type as first argument
        return_type = sig.infer_return_type([StringType])
        assert return_type == StringType

        return_type = sig.infer_return_type([IntegerType])
        assert return_type == IntegerType

    def test_promoted_return_type(self):
        """Test PROMOTED return type strategy."""
        sig = FunctionSignature(function_name="add", type_signature=Numeric(2), return_type=ReturnTypeStrategy.PROMOTED)

        # Should promote to most general type
        return_type = sig.infer_return_type([IntegerType, IntegerType])
        assert return_type == IntegerType

        return_type = sig.infer_return_type([IntegerType, FloatType])
        assert return_type == FloatType

    def test_dynamic_return_type_requires_function(self):
        """Test DYNAMIC return type strategy."""
        sig = FunctionSignature(function_name="extract", type_signature=Exact([StringType]),
                                return_type=ReturnTypeStrategy.DYNAMIC)

        # Should raise error without dynamic function
        with pytest.raises(InternalError, match="DYNAMIC return type requires dynamic_return_type_func"):
            sig.infer_return_type([StringType])

    def test_validate_and_infer_type_integration(self):
        """Test complete validation and type inference."""
        sig = FunctionSignature(function_name="upper", type_signature=Exact([StringType]),
                                return_type=ReturnTypeStrategy.SAME_AS_INPUT)

        # Create mock arguments
        string_col = MockColumn("text_col", StringType)
        plan = MockPlan()

        # Should validate and return correct type
        return_type = sig.validate_and_infer_type([string_col], plan)
        assert return_type == StringType

        # Should fail validation with wrong type
        int_col = MockColumn("int_col", IntegerType)
        with pytest.raises(TypeMismatchError, match="upper Argument 0: expected StringType, got IntegerType"):
            sig.validate_and_infer_type([int_col], plan)

    def test_dynamic_return_type_with_function(self):
        """Test DYNAMIC return type with custom function."""
        sig = FunctionSignature(function_name="array_constructor", type_signature=VariadicUniform(expected_min_args=1),
                                return_type=ReturnTypeStrategy.DYNAMIC)

        def dynamic_return_func(arg_types, logical_plan):
            return ArrayType(arg_types[0])

        string_col = MockColumn("text_col", StringType)
        plan = MockPlan()

        # Should use dynamic function for return type
        return_type = sig.validate_and_infer_type([string_col], plan, dynamic_return_func)
        assert return_type == ArrayType(StringType)


class TestReturnTypeCompatibility:
    """Test return type strategy compatibility validation."""

    def test_same_as_input_with_exact_different_types_fails(self):
        """Test that SAME_AS_INPUT fails with Exact signature having different types."""
        with pytest.raises(InternalError, match="SAME_AS_INPUT not compatible"):
            FunctionSignature(function_name="bad_func", type_signature=Exact([StringType, IntegerType]),
                              return_type=ReturnTypeStrategy.SAME_AS_INPUT)

    def test_same_as_input_with_exact_same_types_succeeds(self):
        """Test that SAME_AS_INPUT works with Exact signature having same types."""
        # Should work fine
        sig = FunctionSignature(function_name="good_func", type_signature=Exact([StringType, StringType]),
                                return_type=ReturnTypeStrategy.SAME_AS_INPUT)
        assert sig.return_type == ReturnTypeStrategy.SAME_AS_INPUT

    def test_promoted_requires_numeric_signature(self):
        """Test that PROMOTED return type requires Numeric signature."""
        with pytest.raises(InternalError, match="PROMOTED return type strategy only compatible"):
            FunctionSignature(function_name="bad_func", type_signature=Exact([StringType]),
                              return_type=ReturnTypeStrategy.PROMOTED)


class TestScalarFunctionIntegration:
    """Test that ValidatedSignature expressions work with MockPlan."""

    def test_validated_signatures_with_mock_plan(self):
        """Test that ValidatedSignature expressions work correctly with MockPlan."""
        # Create a plan with a string column
        plan = MockPlan([ColumnField("text_col", ArrayType(StringType))])

        # Create a StrLengthExpr and test it
        col_expr = ColumnExpr("text_col")
        array_length_expr = ArrayLengthExpr(col_expr)

        result = array_length_expr.to_column_field(plan)
        assert result.data_type == IntegerType

        # Test that type validation works (should fail for wrong type)
        plan_wrong_type = MockPlan([ColumnField("int_col", IntegerType)])
        col_expr_wrong = ColumnExpr("int_col")
        array_length_expr_wrong = ArrayLengthExpr(col_expr_wrong)

        with pytest.raises(TypeMismatchError, match="array_size expects argument 0 to be an array type, got IntegerType"):
            array_length_expr_wrong.to_column_field(plan_wrong_type)
