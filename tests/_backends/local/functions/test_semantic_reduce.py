import polars as pl

from fenic import ColumnField, Schema, StringType, col, lit, semantic
from fenic.api.functions.text import concat as concat
from fenic.api.types import InstructionTemplate
from fenic.core._logical_plan.expressions import AliasExpr, SemanticReduceExpr
from fenic.core._logical_plan.plans import Aggregate


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

def test_semantic_reduce_with_nested_semantic_operations(local_session):
    source = local_session.create_dataframe({
        "title": ["AI Guide", "ML Tutorial", "Data Science"],
        "content": ["Introduction to AI", "Machine Learning Basics", "Data Analysis"]
    })
    
    template = InstructionTemplate(
        "Summarize {category} content: {title} - {content}",
        category=semantic.classify("content", ["AI", "ML", "Data Science"])
    )
    
    df = source.agg(
        semantic.reduce(template).alias("summary")
    )

    fenic_schema = df.schema
    assert fenic_schema == Schema(column_fields=[
        ColumnField(name="summary", data_type=StringType),
    ])
    
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
    
    template = InstructionTemplate(
        "Summarize: {full_title} - {content}",
        full_title=concat(col("title"), lit(" by "), col("author"))
    )
    
    df = source.agg(
        semantic.reduce(template).alias("summary")
    )

    fenic_schema = df.schema
    assert fenic_schema == Schema(column_fields=[
        ColumnField(name="summary", data_type=StringType),
    ])
    
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
    if isinstance(logical_plan, Aggregate):
        agg_exprs = logical_plan.agg_exprs()
        reduce_expr = agg_exprs[0]
        if isinstance(reduce_expr, AliasExpr):
            reduce_expr = reduce_expr.expr
        assert isinstance(reduce_expr, SemanticReduceExpr)
        instruction_exprs = reduce_expr.exprs
        assert len(instruction_exprs) == 2
        expr_names = [expr.name for expr in instruction_exprs]
        assert expr_names == ["title", "body"]
