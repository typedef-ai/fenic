"""Tests for implicit casting in function signatures."""

from fenic.api import col
from fenic.api.functions import semantic, text
from fenic.core._logical_plan.expressions.basic import CastExpr
from fenic.core.types import StringType


def test_semantic_map_mixed_types_with_casting(local_session):
    """Test semantic.map with multiple columns requiring different casting behaviors."""
    # Create dataframe with mixed types
    data = {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["NYC", "SF"]}
    df = local_session.create_dataframe(data)

    # Instruction references string and integer columns
    result_expr = semantic.map(
        "{name} is {age} years old in {city}",
        max_output_tokens=20,
        temperature=0.0
    )

    df_with_select = df.select(result_expr.alias("result"))
    plan = df_with_select._logical_plan

    semantic_map_expr = plan._exprs[0].expr

    # Should have 3 arguments: name, age, city
    assert len(semantic_map_expr.args) == 3

    # name (string) - no cast needed
    name_arg = semantic_map_expr.args[0]
    assert not isinstance(name_arg, CastExpr)

    # age (integer) - should be cast to string  
    age_arg = semantic_map_expr.args[1]
    assert isinstance(age_arg, CastExpr)
    assert age_arg.dest_type == StringType

    # city (string) - no cast needed
    city_arg = semantic_map_expr.args[2]
    assert not isinstance(city_arg, CastExpr)


def test_text_concat_allows_implicit_casting(local_session):
    """Test that text.concat allows implicit casting from integer to string."""
    # Create dataframe with mixed types
    data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    df = local_session.create_dataframe(data)

    # text.concat should allow implicit casting from integer to string
    result_expr = text.concat(col("name"), col("age"))

    df_with_select = df.select(result_expr.alias("result"))
    plan = df_with_select._logical_plan

    concat_expr = plan._exprs[0].expr

    # Should have 2 arguments: name, age
    assert len(concat_expr.args) == 2

    # name (string) - no cast needed
    name_arg = concat_expr.args[0]
    assert not isinstance(name_arg, CastExpr)

    # age (integer) - should be cast to string
    age_arg = concat_expr.args[1]
    assert isinstance(age_arg, CastExpr)
    assert age_arg.dest_type == StringType
