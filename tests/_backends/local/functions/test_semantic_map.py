import polars as pl
import pytest

import fenic as fc
from fenic import (
    ColumnField,
    MapExample,
    MapExampleCollection,
    Schema,
    StringType,
    col,
    lit,
    semantic,
)
from fenic.api.functions.text import concat as concat
from fenic.core._logical_plan.expressions import (
    AliasExpr,
    ColumnExpr,
    LogicalExpr,
    SemanticMapExpr,
)
from fenic.core._logical_plan.plans import Projection
from fenic.core.error import TypeMismatchError


def test_semantic_map(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    state_prompt = "What state does {name} live in given that they live in {city}?"
    df_select = source.select(
        semantic.map(state_prompt).alias("state"),
        col("name"),
        semantic.map(instruction="What is the typical weather in {city} in summer?").alias("weather"),
    )
    result = df_select.to_polars()
    assert result.schema == {
        "state": pl.String,
        "name": pl.String,
        "weather": pl.String,
    }

    weather_prompt = "What is the typical weather in {city} in summer?"
    df_with_column = source.with_column(
        "weather",
        semantic.map(instruction=weather_prompt),
    )
    result = df_with_column.to_polars()
    assert result.schema == {
        "name": pl.String,
        "city": pl.String,
        "weather": pl.String,
    }

def test_semantic_map_with_examples(local_session):
    source = local_session.create_dataframe({"name": ["Alice"], "city": ["New York"]})
    weather_prompt = "What is the weather in {city}?"
    weather_collection = MapExampleCollection()
    weather_collection.create_example(
        MapExample(
            input={"city": "Seattle"},
            output="It is rainy and 60 degrees",
        )
    ).create_example(
        MapExample(
            input={"city": "Los Angeles"},
            output="It is sunny and 70 degrees",
        )
    )
    df_with_column = source.with_column(
        "weather",
        semantic.map(
            instruction=weather_prompt,
            examples=weather_collection,
        ),
    )
    result = df_with_column.to_polars()
    assert result.schema == {
        "name": pl.String,
        "city": pl.String,
        "weather": pl.String,
    }


def test_semantic_map_with_nulls(local_session):
    # have a data source with some nulls.
    source = local_session.create_dataframe(
        {"name": ["Alice", "Bob"], "city": ["New York", None]}
    )
    state_prompt = "What state does {name} live in given that they live in {city}?"
    df_select = source.select(
        col("name"),
        semantic.map(state_prompt).alias("state"),
    )
    result = df_select.to_polars()
    assert result.schema == {
        "name": pl.String,
        "state": pl.String,
    }
    result_list = result["state"].to_list()
    assert len(result_list) == 2
    # Make sure that Bob's state is None.
    assert result_list[1] is None


def test_semantic_map_with_concat_nested_column(local_session):
    """Test semantic.map with a nested string concatenation as a template value."""
    source = local_session.create_dataframe({
        "first_name": ["Alice", "Bob"],
        "last_name": ["Smith", "Jones"],
        "city": ["New York", "Los Angeles"],
        "state": ["New York", "California"],
        "job": ["Engineer", "Doctor"],
    })
    # Concatenate first_name and last_name as a template value
    nested_expr = {"full_name": (concat(col("first_name"), lit(" "), col("last_name"))).alias("full_name")}
    prompt = "The person's full name is {full_name} and they live in {city},{state} and work as a {job}. Write a short description of the person."
    df = source.select(
        col("first_name"),
        col("last_name"),
        semantic.map(
            instruction=prompt,
            bindings=nested_expr
        ).alias("desc")
    )

    fenic_schema = df.schema
    assert fenic_schema == Schema(column_fields=[
        ColumnField(name="first_name", data_type=StringType),
        ColumnField(name="last_name", data_type=StringType),
        ColumnField(name="desc", data_type=StringType),
    ])

    result = df.to_polars()
    assert result.schema == {
        "first_name": pl.String,
        "last_name": pl.String,
        "desc": pl.String,
    }
    assert len(result) == 2
    for val in result["desc"].to_list():
        assert isinstance(val, str) and len(val) > 0


def test_semantic_map_with_nested_non_string_expr_should_raise_validation_error(local_session):
    """Test that semantic.map with a nested non-string expression should raise a validation error."""
    source = local_session.create_dataframe({
        "first_name": ["Alice", "Bob"],
        "last_name": ["Smith", "Jones"],
    })
    nested_expr = {"name_length": (fc.text.length(col("first_name")) + fc.text.length(col("last_name"))).alias("name_length")}
    prompt = "The length of the name is {name_length}."
    with pytest.raises(TypeMismatchError, match="expected StringType, got IntegerType"):
        df = source.select(semantic.map(instruction=prompt, bindings=nested_expr))
        df.to_polars()


def test_semantic_map_with_nested_semantic_map(local_session):
    """Test semantic.map with another semantic.map as a nested column reference."""
    source = local_session.create_dataframe({
        "name": ["Alice", "Bob"],
        "city": ["New York", "San Francisco"],
    })
    # First semantic.map: get state from city
    state_prompt = "What state is {city} in?"
    state_expr = {"state": semantic.map(state_prompt).alias("state")}
    # Second semantic.map: use state in the prompt
    weather_prompt = "What is the weather like in {city}, {state}?"
    df = source.select(
        col("name"),
        col("city"),
        semantic.map(
            instruction=weather_prompt,
            bindings=state_expr
        ).alias("weather_report")
    )
    fenic_schema = df.schema
    assert fenic_schema == Schema(column_fields=[
        ColumnField(name="name", data_type=StringType),
        ColumnField(name="city", data_type=StringType),
        ColumnField(name="weather_report", data_type=StringType),
    ])

    result = df.to_polars()
    assert result.schema == {
        "name": pl.String,
        "city": pl.String,
        "weather_report": pl.String,
    }
    assert len(result) == 2
    for val in result["weather_report"].to_list():
        assert isinstance(val, str) and len(val) > 0


def test_semantic_map_placeholder_deduplication(local_session):
    source = local_session.create_dataframe({
        "name": ["Alice", "Bob"],
        "city": ["New York", "Los Angeles"],
    })
    expensive_expr = concat(
        col("name"),
        lit(" (processed)"),
        fc.text.upper(col("name"))
    ).alias("processed_name")
    prompt = "Hello {processed_name}, how are you {processed_name}? You live in {city}."
    bindings = {
        "processed_name": expensive_expr,
        "city": col("city")
    }
    df = source.select(
        col("name"),
        col("city"),
        semantic.map(
            instruction=prompt,
            bindings=bindings
        ).alias("greeting")
    )
    result = df.to_polars()
    assert len(result) == 2
    for val in result["greeting"].to_list():
        assert isinstance(val, str) and len(val) > 0
    plan = df._logical_plan
    if isinstance(plan, Projection):
        map_expr_children: list[LogicalExpr] = []
        for expr in plan.exprs():
            if isinstance(expr, AliasExpr) and isinstance(expr.expr, SemanticMapExpr):
                map_expr_children = expr.expr.children()
                break

        placeholder_names = []
        for expr in map_expr_children:
            if isinstance(expr, ColumnExpr) or isinstance(expr, AliasExpr):
                placeholder_names.append(expr.name)
        assert sorted(placeholder_names) == sorted(["processed_name", "city"])
    else:
        raise ValueError("Unexpected logical plan type")