import re

import pytest

from fenic import (
    BooleanType,
    Column,
    FloatType,
    IntegerType,
    StringType,
    avg,
    col,
    lit,
    max,
    min,
    sum,
    when,
)
from fenic.core._logical_plan.expressions import (
    AggregateExpr,
    AliasExpr,
    ArithmeticExpr,
    BooleanExpr,
    ColumnExpr,
    EqualityComparisonExpr,
    LiteralExpr,
    NotExpr,
    NumericComparisonExpr,
    Operator,
)
from fenic.core.error import TypeMismatchError, ValidationError


def test_col():
    """Test the `col` function creates a Column with the correct expression."""
    column = col("age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ColumnExpr)
    assert column._logical_expr.name == "age"


def test_col_alias():
    """Test the `col` function creates a Column with the correct expression."""
    column = col("age").alias("age2")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, AliasExpr)
    assert column._logical_expr.name == "age2"
    assert column._logical_expr.expr.name == "age"


def test_literal_expression():
    """Test creating a literal expression with different types."""
    int_lit = lit(10)
    float_lit = lit(3.14)
    bool_lit = lit(True)
    str_lit = lit("test")

    assert isinstance(int_lit._logical_expr, LiteralExpr)
    assert int_lit._logical_expr.data_type == IntegerType
    assert float_lit._logical_expr.data_type == FloatType
    assert bool_lit._logical_expr.data_type == BooleanType
    assert str_lit._logical_expr.data_type == StringType


def test_col_math():
    column = col("age") + lit(1)
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = col("age") + col("another_age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = col("age") - 1
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = col("age") - col("another_age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = col("age") * 1
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = col("age") * col("another_age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = col("age") / 1
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = col("age") / col("another_age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)


def test_column_reverse_math():
    column = 1 + col("age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)

    column = 1 - col("age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)
    assert column._logical_expr.left.literal == 1
    assert column._logical_expr.right.name == "age"

    column = 1 * col("age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)
    assert column._logical_expr.left.name == "age"
    assert column._logical_expr.right.literal == 1

    column = 1 - col("age")
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, ArithmeticExpr)
    assert column._logical_expr.right.name == "age"
    assert column._logical_expr.left.literal == 1


def test_boolean_expression():
    """Test boolean operations between columns and literals."""
    column = col("age") & lit(True)
    assert isinstance(column._logical_expr, BooleanExpr)
    assert column._logical_expr.op == Operator.AND

    column = col("age") | col("another_age")
    assert isinstance(column._logical_expr, BooleanExpr)
    assert column._logical_expr.op == Operator.OR

    column = col("age") & col("another_age") | col("yet_another_age")
    assert isinstance(column._logical_expr, BooleanExpr)
    assert column._logical_expr.op == Operator.OR

    column = col("age") | col("another_age") & col("yet_another_age")
    assert isinstance(column._logical_expr, BooleanExpr)
    assert column._logical_expr.op == Operator.OR


def test_boolean_expression_with_columns():
    """Test boolean operations between columns and literals."""
    with pytest.raises(TypeError, match="Cannot use Column in boolean context"):
        col("age") and col("another_age")

    with pytest.raises(TypeError, match="Cannot use Column in boolean context"):
        col("age") or col("another_age")

    with pytest.raises(TypeError, match="Cannot use Column in boolean context"):
        not col("age")


def test_column_equality():
    """Test equality and inequality expressions."""
    column = col("age") == col("another_age")
    assert isinstance(column._logical_expr, EqualityComparisonExpr)
    assert column._logical_expr.op == Operator.EQ

    column = col("age") != lit(5)
    assert isinstance(column._logical_expr, EqualityComparisonExpr)
    assert column._logical_expr.op == Operator.NOT_EQ

    column = col("age") >= 1
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, NumericComparisonExpr)
    assert column._logical_expr.left.name == "age"
    assert column._logical_expr.right.literal == 1
    assert column._logical_expr.op == Operator.GTEQ

    column = 1 >= col("age")
    assert isinstance(column, Column)
    assert column._logical_expr.op == Operator.LTEQ

    column = col("age") > 1
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, NumericComparisonExpr)
    assert column._logical_expr.op == Operator.GT

    column = 1 > col("age")
    assert isinstance(column, Column)
    assert column._logical_expr.op == Operator.LT

    column = col("age") <= 1
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, NumericComparisonExpr)
    assert column._logical_expr.op == Operator.LTEQ

    column = 1 <= col("age")
    assert isinstance(column, Column)
    assert column._logical_expr.op == Operator.GTEQ

    column = col("age") < 1
    assert isinstance(column, Column)
    assert isinstance(column._logical_expr, NumericComparisonExpr)
    assert column._logical_expr.op == Operator.LT

    column = 1 < col("age")
    assert isinstance(column, Column)
    assert column._logical_expr.op == Operator.GT


def test_not_expression():
    column = ~(col("age") == 1)
    assert isinstance(column._logical_expr, NotExpr)
    assert isinstance(column._logical_expr.expr, EqualityComparisonExpr)
    assert column._logical_expr.expr.op == Operator.EQ
    assert column._logical_expr.expr.left.name == "age"
    assert column._logical_expr.expr.right.literal == 1


def test_nested_expression():
    """Test nested arithmetic expressions."""
    column = col("age") + 1 + 2
    assert isinstance(column._logical_expr, ArithmeticExpr)
    assert column._logical_expr.op == Operator.PLUS
    assert isinstance(column._logical_expr.left, ArithmeticExpr)
    assert isinstance(column._logical_expr.left.left, ColumnExpr)
    assert column._logical_expr.right.literal == 2


def test_sum_aggregation():
    column = sum(col("age"))
    assert isinstance(column._logical_expr, AggregateExpr)
    assert column._logical_expr.agg_name == "sum"
    assert isinstance(column._logical_expr.expr, ColumnExpr)
    assert column._logical_expr.expr.name == "age"


def test_avg_aggregation():
    column = avg(col("age"))
    assert isinstance(column._logical_expr, AggregateExpr)
    assert column._logical_expr.agg_name == "avg"
    assert isinstance(column._logical_expr.expr, ColumnExpr)
    assert column._logical_expr.expr.name == "age"


def test_min_aggregation():
    column = min(col("age"))
    assert isinstance(column._logical_expr, AggregateExpr)
    assert column._logical_expr.agg_name == "min"
    assert isinstance(column._logical_expr.expr, ColumnExpr)
    assert column._logical_expr.expr.name == "age"


def test_max_aggregation():
    column = max(col("age"))
    assert isinstance(column._logical_expr, AggregateExpr)
    assert column._logical_expr.agg_name == "max"
    assert isinstance(column._logical_expr.expr, ColumnExpr)
    assert column._logical_expr.expr.name == "age"


def test_column_comparisons(local_session):
    # Create test data
    data = {"age": [20, 25, 30], "salary": [30000, 45000, 70000]}
    df = local_session.create_dataframe(data)

    # Test greater than
    result = df.filter(col("age") > 25).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 30

    # Test less than or equal
    result = df.filter(col("age") <= 25).to_polars()
    assert len(result) == 2
    assert all(age <= 25 for age in result["age"])

    # Test equality
    result = df.filter(col("age") == 25).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 25


def test_column_logical_operations(local_session):
    data = {"age": [20, 25, 30], "salary": [30000, 45000, 70000]}
    df = local_session.create_dataframe(data)

    # Test AND
    result = df.filter((col("age") > 20) & (col("salary") < 50000)).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 25

    # Test OR
    result = df.filter((col("age") < 25) | (col("salary") > 60000)).to_polars()
    assert len(result) == 2


def test_column_arithmetic(local_session):
    data = {"value1": [10, 20, 30], "value2": [1, 2, 3]}
    df = local_session.create_dataframe(data)

    # Test addition
    result = df.filter((col("value1") + col("value2")) > 21).to_polars()
    assert len(result) == 2
    assert result["value1"][0] == 20

    # Test multiplication
    result = df.filter((col("value1") * col("value2")) > 50).to_polars()
    assert len(result) == 1
    assert result["value1"][0] == 30


def test_getitem_list(local_session):
    """Test getItem() and [] access for list columns."""
    data = {"list_col": [[1, 2, 3], [4, 5, 6]]}
    df = local_session.create_dataframe(data)
    # Test getItem() access
    result = df.select(col("list_col").get_item(0)).to_polars()
    # Check column name follows PySpark convention
    assert result.columns == ["list_col[0]"]
    # Convert to dictionary using correct Polars API
    result_dict = result.to_dict(as_series=False)
    assert result_dict["list_col[0]"] == [1, 4]
    # Test [] syntax
    result2 = df.select(col("list_col")[0]).to_polars()
    assert result2.columns == ["list_col[0]"]
    result2_dict = result2.to_dict(as_series=False)
    assert result2_dict["list_col[0]"] == [1, 4]


def test_getitem_dict(local_session):
    """Test getItem with dictionary column."""
    # Create DataFrame with dict column
    data = {"id": [1, 2], "dict_col": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    df = local_session.create_dataframe(data)
    # Test accessing by key
    result = df.select(col("dict_col").get_item("a")).to_polars()
    assert result.columns == ["dict_col[a]"]
    assert result.to_dict(as_series=False) == {"dict_col[a]": [1, 3]}
    # Test using [] syntax
    result = df.select(col("dict_col")["b"]).to_polars()
    assert result.columns == ["dict_col[b]"]
    assert result.to_dict(as_series=False) == {"dict_col[b]": [2, 4]}


def test_getitem_edgecases(local_session):
    """Test error cases for getItem."""
    data = {"list_col": [[1, 2, 3], [3, 4]], "dict_col": [{"a": 1}, {"a": 2}]}
    df = local_session.create_dataframe(data)

    assert df.select(col("list_col").get_item(2)).to_polars().to_dict(
        as_series=False
    ) == {"list_col[2]": [3, None]}

    with pytest.raises(ValueError):
        df.select(col("dict_col").get_item("b"))


def test_struct_dot_notation(local_session):
    """Test accessing nested struct fields using attribute access."""
    data = {"my_col": [{"a": {"nested": 1}, "b": 2}, {"a": {"nested": 3}, "b": 4}]}
    df = local_session.create_dataframe(data)

    # Test single level of nesting
    result = df.select(df.my_col.a).to_polars()
    assert result.columns == ["my_col[a]"]
    assert result.to_dict(as_series=False) == {
        "my_col[a]": [{"nested": 1}, {"nested": 3}]
    }

    # Test multiple levels of nesting
    result = df.select(df.my_col.a.nested).to_polars()
    assert result.columns == ["my_col[a][nested]"]
    assert result.to_dict(as_series=False) == {"my_col[a][nested]": [1, 3]}


def test_is_null(local_session):
    # Create data with null values across different types
    data = {
        "int_col": [20, None, 30],
        "float_col": [30.5, 45.2, None],
        "str_col": [None, "hello", "world"],
        "bool_col": [True, None, False],
        "list_col": [[1, 2], None, [3, 4]],
        "dict_col": [{"a": 1}, None, {"b": 2}],
    }
    df = local_session.create_dataframe(data)

    # Test is_null on integer column
    result = df.filter(col("int_col").is_null()).to_polars()
    assert len(result) == 1
    assert result["str_col"][0] == "hello"

    # Test is_null on float column
    result = df.filter(col("float_col").is_null()).to_polars()
    assert len(result) == 1
    assert result["int_col"][0] == 30

    # Test is_null on string column
    result = df.filter(col("str_col").is_null()).to_polars()
    assert len(result) == 1
    assert result["int_col"][0] == 20

    # Test is_null on boolean column
    result = df.filter(col("bool_col").is_null()).to_polars()
    assert len(result) == 1
    assert result["str_col"][0] == "hello"

    # Test is_null on complex types
    result = df.filter(col("list_col").is_null()).to_polars()
    assert len(result) == 1

    result = df.filter(col("dict_col").is_null()).to_polars()
    assert len(result) == 1


def test_is_not_null(local_session):
    # Create data with null values across different types
    data = {
        "int_col": [20, None, 30],
        "float_col": [30.5, 45.2, None],
        "str_col": [None, "hello", "world"],
        "bool_col": [True, None, False],
        "list_col": [[1, 2], None, [3, 4]],
        "dict_col": [{"a": 1}, None, {"b": 2}],
    }
    df = local_session.create_dataframe(data)

    # Test is_not_null on integer column
    result = df.filter(col("int_col").is_not_null()).to_polars()
    assert len(result) == 2
    assert set([x for x in result["int_col"] if x is not None]) == {20, 30}

    # Test is_not_null on float column
    result = df.filter(col("float_col").is_not_null()).to_polars()
    assert len(result) == 2
    assert set([x for x in result["float_col"] if x is not None]) == {30.5, 45.2}

    # Test is_not_null on string column
    result = df.filter(col("str_col").is_not_null()).to_polars()
    assert len(result) == 2
    assert set(result["str_col"]) == {"hello", "world"}

    # Test is_not_null on boolean column
    result = df.filter(col("bool_col").is_not_null()).to_polars()
    assert len(result) == 2

    # Test is_not_null on complex types
    result = df.filter(col("list_col").is_not_null()).to_polars()
    assert len(result) == 2

    result = df.filter(col("dict_col").is_not_null()).to_polars()
    assert len(result) == 2


def test_contains(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").contains("ell")).to_polars()
    assert len(result) == 1
    assert result["text_col"][0] == "hello"


def test_contains_search_column(local_session):
    data = {
        "text_col": ["hello", "world", "foo", "bar"],
        "substring": ["ell", "orl", "food", "bard"],
    }
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").contains(col("substring"))).to_polars()
    assert len(result) == 2
    assert result["text_col"][0] == "hello"
    assert result["text_col"][1] == "world"


def test_like(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").like("%ell%")).to_polars()
    assert len(result) == 1
    assert result["text_col"][0] == "hello"

    # test that like is case sensitive
    sensitive_result = df.filter(col("text_col").like("%ELL%")).to_polars()
    assert len(sensitive_result) == 0


def test_ilike(local_session):
    # test that ilike is case insensitive
    data = {"text_col": ["HELLO", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").ilike("%ell%")).to_polars()
    assert len(result) == 1
    assert result["text_col"][0] == "HELLO"


def test_rlike(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").rlike("^[hw].*")).to_polars()
    assert len(result) == 2
    assert result["text_col"][0] == "hello"
    assert result["text_col"][1] == "world"

    # Test with invalid regex
    with pytest.raises(ValueError):
        bad_pattern = r"[.*"
        df.filter(col("text_col").rlike(bad_pattern))


def test_starts_with(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").starts_with("h")).to_polars()
    assert len(result) == 1
    assert result["text_col"][0] == "hello"

    # test that starts_with does not accept regex
    with pytest.raises(ValueError):
        df.filter(col("text_col").starts_with("^"))


def test_starts_with_search_column(local_session):
    data = {
        "text_col": ["hello", "world", "foo", "bar"],
        "starting": ["hell", "wor", "b", "fo"],
    }
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").starts_with(col("starting"))).to_polars()
    assert len(result) == 2
    assert result["text_col"][0] == "hello"
    assert result["text_col"][1] == "world"


def test_ends_with(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").ends_with("o")).to_polars()
    assert len(result) == 2
    assert result["text_col"][0] == "hello"
    assert result["text_col"][1] == "foo"

    # test that ends_with does not accept regex
    with pytest.raises(ValueError):
        df.filter(col("text_col").ends_with("o$"))


def test_ends_with_search_column(local_session):
    data = {
        "text_col": ["hello", "world", "foo", "bar"],
        "ending": ["o", "rld", "ood", "ard"],
    }
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").ends_with(col("ending"))).to_polars()
    assert len(result) == 2
    assert result["text_col"][0] == "hello"
    assert result["text_col"][1] == "world"


def test_contains_any(local_session):
    data = {"text_col": ["hello", "world", "foo", "bar"]}
    df = local_session.create_dataframe(data)
    result = df.filter(col("text_col").contains_any(["ell", "orl"])).to_polars()
    assert len(result) == 2
    assert result["text_col"][0] == "hello"
    assert result["text_col"][1] == "world"

    result_sensitive = df.filter(
        col("text_col").contains_any(["ELL", "orl"], case_insensitive=False)
    ).to_polars()
    assert len(result_sensitive) == 1
    assert result_sensitive["text_col"][0] == "world"

    result_insensitive = df.filter(
        col("text_col").contains_any(["ELL", "orl"], case_insensitive=True)
    ).to_polars()
    assert len(result_insensitive) == 2
    assert result_insensitive["text_col"][0] == "hello"
    assert result_insensitive["text_col"][1] == "world"


def test_not_df(local_session):
    """Test the negation operator (~) with a real dataframe."""
    data = {"age": [20, 25, 30], "name": ["Alice", "Bob", "Charlie"]}
    df = local_session.create_dataframe(data)

    # Test negating equality
    result = df.filter(~(col("age") == 25)).to_polars()
    assert len(result) == 2
    assert set(result["age"]) == {20, 30}

    # Test negating inequality
    result = df.filter(~(col("age") != 25)).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 25

    # Test negating comparison
    result = df.filter(~(col("age") > 25)).to_polars()
    assert len(result) == 2
    assert set(result["age"]) == {20, 25}

    # Test negating is_null
    data_with_nulls = {"value": [10, None, 30]}
    df_nulls = local_session.create_dataframe(data_with_nulls)

    result = df_nulls.filter(~(col("value").is_null())).to_polars()
    assert len(result) == 2
    assert set([x for x in result["value"] if x is not None]) == {10, 30}

    # Test negating complex conditions
    result = df.filter(~((col("age") > 20) & (col("age") < 30))).to_polars()
    assert len(result) == 2
    assert set(result["age"]) == {20, 30}

    # Test double negation
    result = df.filter(~(~(col("age") == 25))).to_polars()
    assert len(result) == 1
    assert result["age"][0] == 25


def test_conditions(local_session):
    data = {
        "first_age": [20, 25, 30, 40, None],
        "second_age": [40, 25, None, 10, None],
        "name": ["Alice", "Bob", "Charlie", "Dave", "no_entry"],
    }
    df = local_session.create_dataframe(data)

    # Test with default and None
    result = df.select(
        col("first_age")
        .when(col("first_age") > 20, lit("first_old"))
        .alias("age_group")
    ).to_polars()
    assert result["age_group"][0] is None
    assert result["age_group"][1] == "first_old"
    assert result["age_group"][2] == "first_old"
    assert result["age_group"][3] == "first_old"
    assert result["age_group"][4] is None

    # Test otherwise with None
    result = df.select(
        col("first_age")
        .when(col("first_age") > 20, lit("first_old"))
        .otherwise(lit("first_young"))
        .alias("age_group")
    ).to_polars()
    assert result["age_group"][0] == "first_young"
    assert result["age_group"][1] == "first_old"
    assert result["age_group"][2] == "first_old"
    assert result["age_group"][3] == "first_old"
    assert result["age_group"][4] == "first_young"

    # Test overlapping condition ordering - first met condition takes precedence
    result = df.select(
        col("first_age")
        .when(col("first_age") > 20, lit("first_old"))
        .when(col("second_age") > 20, lit("second_old"))
        .alias("age_group")
    ).to_polars()
    assert result["age_group"][0] == "second_old"
    assert result["age_group"][1] == "first_old"
    assert result["age_group"][2] == "first_old"
    assert result["age_group"][3] == "first_old"
    assert result["age_group"][4] is None

    # Test overlapping condition ordering with otherwise- first met condition takes precedence
    result = df.select(
        col("first_age")
        .when(col("first_age") > 25, lit("first_old"))
        .when(col("second_age") > 25, lit("second_old"))
        .otherwise(lit("young"))
        .alias("age_group")
    ).to_polars()
    assert result["age_group"][0] == "second_old"
    assert result["age_group"][1] == "young"
    assert result["age_group"][2] == "first_old"
    assert result["age_group"][3] == "first_old"
    assert result["age_group"][4] == "young"

    # Test starting chain with function
    result = df.select(
        when(col("first_age") > 25, lit("first_old"))
        .when(col("second_age") > 25, lit("second_old"))
        .otherwise(lit("young"))
        .alias("age_group")
    ).to_polars()
    assert result["age_group"][0] == "second_old"
    assert result["age_group"][1] == "young"
    assert result["age_group"][2] == "first_old"
    assert result["age_group"][3] == "first_old"
    assert result["age_group"][4] == "young"


def test_conditions_errors(local_session):
    data = {
        "age": [20, 25, 30, 40, None],
        "name": ["Alice", "Bob", "Charlie", "Dave", "passed"],
    }
    df = local_session.create_dataframe(data)
    with pytest.raises(TypeError, match="can only be called on when"):
        df.select(
            col("age").otherwise(lit("young")).alias("age_group")
        ).to_polars()

    with pytest.raises(TypeError, match="condition must be a boolean expression."):
        df.select(
            col("age").when(col("age") + 3, lit("old")).alias("age_group")
        ).to_polars()


def test_conditions_derived_columns(local_session):
    data = {
        "first_age": [20, 25, 30, 40, None],
        "second_age": [40, 25, None, 10, None],
        "name": ["Alice", "Bob", "Charlie", "Dave", "no_entry"],
    }

    df = local_session.create_dataframe(data)
    result = df.select(
        when(col("first_age") + col("second_age") > 50, lit("old")).alias("age_group")
    ).to_polars()
    assert result["age_group"][0] == "old"
    assert result["age_group"][1] is None
    assert result["age_group"][2] is None
    assert result["age_group"][3] is None
    assert result["age_group"][4] is None

    df = local_session.create_dataframe(data)
    result = df.select(
        when(col("first_age") + col("second_age") > 50, lit("old"))
        .otherwise(lit("young"))
        .alias("age_group")
    ).to_polars()
    assert result["age_group"][0] == "old"
    assert result["age_group"][1] == "young"
    assert result["age_group"][2] == "young"
    assert result["age_group"][3] == "young"
    assert result["age_group"][4] == "young"

    # Test resulting when expression used in another expression
    result = df.select(
        (
            when(col("first_age") > 25, lit(100))
            .when(col("second_age") > 25, lit(200))
            .otherwise(lit(0))
            + lit(11)
        ).alias("modified_age")
    ).to_polars()
    assert result["modified_age"][0] == 211
    assert result["modified_age"][1] == 11
    assert result["modified_age"][2] == 111
    assert result["modified_age"][3] == 111
    assert result["modified_age"][4] == 11
    # Test resulting otherwise expression used in another expression
    result = df.with_column(
        "modified_age",
        when(col("first_age") > 25, lit(100)).when(col("second_age") > 25, lit(200))
        + lit(11),
    ).to_polars()
    assert result["modified_age"][0] == 211
    assert result["modified_age"][1] is None
    assert result["modified_age"][2] == 111
    assert result["modified_age"][3] == 111
    assert result["modified_age"][4] is None

def test_in_expr(local_session):
    data = {
        "name": ["Alice", "Bob", "Charlie", "Dave"],
        "name_list": [["Alice", "Bob"], ["Charlie", "Dave"], ["Alice", "Bob"], ["Charlie", "Dave"]],
        "name_struct": [{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}, {"name": "Dave"}],
        "name_list_struct": [[{"name": "Alice"}, {"name": "Bob"}], [{"name": "Charlie"}, {"name": "Dave"}], [{"name": "Alice"}, {"name": "Bob"}], [{"name": "Charlie"}, {"name": "Dave"}]],
    }
    df = local_session.create_dataframe(data)
    result = df.select(col("name").is_in(["Alice", "Bob"]).alias("is_in")).to_polars()
    assert result['is_in'].to_list() == [True, True, False, False]

    result = df.select(col("name").is_in(col("name_list")).alias("is_in")).to_polars()
    assert result['is_in'].to_list() == [True, False, False, True]

    result = df.select(col("name_struct").is_in(col("name_list_struct")).alias("is_in")).to_polars()
    assert result['is_in'].to_list() == [True, False, False, True]


    with pytest.raises(TypeMismatchError, match="The element being searched for must match the array's element type."):
        result = df.select(col("name").is_in([1, 2, 3])).to_polars()

    with pytest.raises(ValidationError, match=re.escape("Cannot apply IN on [5, True]. List argument to IN must be be a valid Python List literal.")):
        result = df.select(col("name").is_in([5, True])).to_polars()
