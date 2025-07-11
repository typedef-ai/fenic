import polars as pl
import pytest

from fenic import col, lit, semantic
from fenic.api.functions.text import concat as concat
from fenic.core._logical_plan.expressions import AliasExpr, SemanticReduceExpr
from fenic.core.error import ValidationError


def test_semantic_reduce(local_session):
    source = local_session.create_dataframe({
        "title": ["Document 1", "Document 2", "Document 3"],
        "body": ["Content about AI", "Content about ML", "Content about data science"]
    })
    
    df_agg = source.agg(
        semantic.reduce("Summarize these documents: {title} - {body}").alias("summary")
    )
    result = df_agg.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_bindings(local_session):
    source = local_session.create_dataframe({
        "first_name": ["Alice", "Bob", "Charlie"],
        "last_name": ["Smith", "Jones", "Brown"],
        "feedback": ["Great service", "Good experience", "Excellent support"]
    })
    
    bindings = {
        "full_name": concat(col("first_name"), lit(" "), col("last_name"))
    }
    
    df = source.agg(
        semantic.reduce(
            instruction="Summarize customer feedback from: {full_name} - {feedback}",
            bindings=bindings
        ).alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_pre_aliased_expressions(local_session):
    source = local_session.create_dataframe({
        "first_name": ["Alice", "Bob"],
        "last_name": ["Smith", "Jones"],
        "feedback": ["Great service", "Good experience"]
    })
    
    bindings = {
        "full_name": concat(col("first_name"), lit(" "), col("last_name")).alias("full_name")
    }
    
    df = source.agg(
        semantic.reduce(
            instruction="Summarize feedback from: {full_name} - {feedback}",
            bindings=bindings
        ).alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_invalid_alias_should_raise_error(local_session):
    source = local_session.create_dataframe({
        "first_name": ["Alice"],
        "last_name": ["Smith"]
    })
    bindings = {
        "full_name": concat(col("first_name"), lit(" "), col("last_name")).alias("wrong_name")
    }
    with pytest.raises(ValidationError, match="Alias name must match the key"):
        df = source.agg(
            semantic.reduce(
                instruction="Summarize: {full_name}",
                bindings=bindings
            )
        )
        df.to_polars()


def test_semantic_reduce_with_nested_semantic_operations(local_session):
    source = local_session.create_dataframe({
        "title": ["AI Guide", "ML Tutorial", "Data Science"],
        "content": ["Introduction to AI", "Machine Learning Basics", "Data Analysis"]
    })
    
    classify_expr = {"category": semantic.classify("content", ["AI", "ML", "Data Science"]).alias("category")}
    
    df = source.agg(
        semantic.reduce(
            instruction="Summarize {category} content: {title} - {content}",
            bindings=classify_expr
        ).alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_concat_expression(local_session):
    source = local_session.create_dataframe({
        "title": ["AI Guide", "ML Tutorial"],
        "author": ["Alice Smith", "Bob Jones"],
        "content": ["Introduction to AI", "Machine Learning Basics"]
    })
    
    bindings = {
        "full_title": concat(col("title"), lit(" by "), col("author"))
    }
    
    df = source.agg(
        semantic.reduce(
            instruction="Summarize: {full_title} - {content}",
            bindings=bindings
        ).alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_nulls(local_session):
    source = local_session.create_dataframe({
        "title": ["Document 1", "Document 2", None],
        "body": ["Content 1", None, "Content 3"]
    })
    
    df = source.agg(
        semantic.reduce("Summarize: {title} - {body}").alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_logical_plan_deduplication(local_session):
    source = local_session.create_dataframe({
        "title": ["Document 1", "Document 2"],
        "body": ["Content 1", "Content 2"]
    })
    df = source.agg(
        semantic.reduce("Summarize: {title} - {body}. Also summarize: {title} again.").alias("summary")
    )
    logical_plan = df._logical_plan
    # For aggregate plans, access the aggregate expressions
    assert hasattr(logical_plan, 'agg_exprs')
    agg_exprs = logical_plan.agg_exprs()
    assert len(agg_exprs) == 1
    reduce_expr = agg_exprs[0]
    if isinstance(reduce_expr, AliasExpr):
        reduce_expr = reduce_expr.expr
    assert isinstance(reduce_expr, SemanticReduceExpr)
    instruction_exprs = reduce_expr.exprs
    assert len(instruction_exprs) == 2
    expr_names = [expr.name for expr in instruction_exprs if hasattr(expr, 'name')]
    assert expr_names == ["title", "body"]


def test_semantic_reduce_expression_evaluated_once(local_session):
    source = local_session.create_dataframe({
        "first_name": ["Alice", "Bob"],
        "last_name": ["Smith", "Jones"],
        "city": ["New York", "Los Angeles"]
    })
    
    full_name_expr = concat(col("first_name"), lit(" "), col("last_name")).alias("full_name")
    bindings = {
        "full_name": full_name_expr
    }
    
    df = source.agg(
        semantic.reduce(
            instruction="Summarize: {full_name} from {city}. Also: {full_name} again.",
            bindings=bindings
        ).alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_complex_bindings(local_session):
    source = local_session.create_dataframe({
        "first_name": ["Alice", "Bob"],
        "last_name": ["Smith", "Jones"],
        "city": ["New York", "Los Angeles"],
        "state": ["NY", "CA"],
        "feedback": ["Great service", "Good experience"]
    })
    bindings = {
        "full_name": concat(col("first_name"), lit(" "), col("last_name")),
        "location": concat(col("city"), lit(", "), col("state")),
        "feedback_summary": semantic.analyze_sentiment("feedback").alias("feedback_summary")
    }
    df = source.agg(
        semantic.reduce(
            instruction="Summarize feedback from {full_name} in {location}: {feedback_summary}",
            bindings=bindings
        ).alias("summary")
    )
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_simple_column_references(local_session):
    source = local_session.create_dataframe({
        "title": ["Document 1", "Document 2"],
        "body": ["Content 1", "Content 2"]
    })
    
    df = source.agg(
        semantic.reduce("Summarize: {title} - {body}").alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str)


def test_semantic_reduce_with_mixed_simple_and_complex(local_session):
    source = local_session.create_dataframe({
        "first_name": ["Alice", "Bob"],
        "last_name": ["Smith", "Jones"],
        "city": ["New York", "Los Angeles"],
        "feedback": ["Great service", "Good experience"]
    })
    
    bindings = {
        "full_name": concat(col("first_name"), lit(" "), col("last_name"))
    }
    
    df = source.agg(
        semantic.reduce(
            instruction="Summarize feedback from {full_name} in {city}: {feedback}",
            bindings=bindings
        ).alias("summary")
    )
    
    result = df.to_polars()
    assert result.schema == {"summary": pl.String}
    assert len(result) == 1
    assert isinstance(result["summary"][0], str) 