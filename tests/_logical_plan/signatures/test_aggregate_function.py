"""Tests for AggregateFunction base class functionality."""

import pytest

from fenic.core._logical_plan.expressions.aggregate import AvgExpr, CountExpr, SumExpr
from fenic.core.error import TypeMismatchError
from fenic.core.types.datatypes import (
    DoubleType,
    EmbeddingType,
    IntegerType,
    StringType,
)
from fenic.core.types.schema import ColumnField


class MockColumn:
    """Mock column expression for testing."""
    
    def __init__(self, name: str, data_type):
        self.name = name
        self.data_type = data_type
    
    def to_column_field(self, plan):
        return ColumnField(self.name, self.data_type)
    
    def __str__(self):
        return self.name


class MockPlan:
    """Mock logical plan for testing."""
    
    def __init__(self, columns=None):
        self.columns = columns or []


class TestAggregateFunction:
    """Test AggregateFunction base class."""
    
    def test_aggregate_function_uses_registry_for_validation(self):
        """Test that AggregateFunction uses function registry for validation."""
        # SumExpr should use the registry system
        int_col = MockColumn("int_col", IntegerType)
        plan = MockPlan()
        
        sum_expr = SumExpr(int_col)
        result = sum_expr.to_column_field(plan)
        
        # Should validate successfully and return same type for sum
        assert result.data_type == IntegerType
        assert result.name == "sum(int_col)"
    
    def test_aggregate_function_signature_validation(self):
        """Test that signature validation works for aggregate functions."""
        # Sum should reject string types
        string_col = MockColumn("str_col", StringType) 
        plan = MockPlan()
        
        sum_expr = SumExpr(string_col)
        
        # Should fail validation
        with pytest.raises(TypeMismatchError):
            sum_expr.to_column_field(plan)
    
    def test_avg_dynamic_return_type(self):
        """Test that AvgExpr correctly handles dynamic return types."""
        plan = MockPlan()
        
        # Test numeric types return DoubleType
        int_col = MockColumn("int_col", IntegerType)
        avg_expr = AvgExpr(int_col)
        result = avg_expr.to_column_field(plan)
        assert result.data_type == DoubleType
        
        # Test embedding types return same type
        embedding_col = MockColumn("emb_col", EmbeddingType(dimensions=128, embedding_model="test"))
        avg_expr_emb = AvgExpr(embedding_col)
        result_emb = avg_expr_emb.to_column_field(plan)
        assert isinstance(result_emb.data_type, EmbeddingType)
        assert result_emb.data_type.embedding_model == "test"
        assert result_emb.data_type.dimensions == 128
    
    def test_count_accepts_any_type(self):
        """Test that CountExpr accepts any input type."""
        plan = MockPlan()
        
        # Test various types
        for data_type in [IntegerType, StringType, DoubleType]:
            col = MockColumn("col", data_type)
            count_expr = CountExpr(col)
            result = count_expr.to_column_field(plan)
            
            # Count always returns IntegerType
            assert result.data_type == IntegerType
            assert result.name == "count(col)"
    
    def test_aggregate_function_children(self):
        """Test that children() method works correctly."""
        int_col = MockColumn("int_col", IntegerType)
        sum_expr = SumExpr(int_col)
        
        children = sum_expr.children()
        assert len(children) == 1
        assert children[0] is int_col
    
    def test_aggregate_function_str_representation(self):
        """Test string representation of aggregate functions."""
        int_col = MockColumn("int_col", IntegerType)
        sum_expr = SumExpr(int_col)
        
        assert str(sum_expr) == "sum(int_col)"