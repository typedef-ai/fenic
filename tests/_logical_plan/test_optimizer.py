from pydantic import BaseModel, Field

from fenic.api.functions import col, lit, semantic, text
from fenic.core._logical_plan.optimizer import (
    LogicalPlanOptimizer,
    MergeFiltersRule,
    NotFilterPushdownRule,
    SemanticFilterRewriteRule,
)


def test_merge_filters_basic(local_session):
    df = local_session.create_dataframe({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    df = (
        df.filter(col("id") > 1)
        .filter(col("id") < 3)
        .filter(col("value") == "b")
        .filter(col("id") == 2)
    )
    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=((id = lit(2)) AND ((value = lit(b)) AND ((id < lit(3)) AND (id > lit(1))))))
  InMemorySource(schema=[ColumnField(name='id', data_type=IntegerType), ColumnField(name='value', data_type=StringType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_merge_filters_with_cache(local_session):
    df = local_session.create_dataframe({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    df = df.filter(col("id") > 1).cache()
    df = df.filter(col("id") < 3).filter(col("value") == "b")
    df = df.filter(col("id") == 2).cache()
    df = df.filter(col("id") > 1)
    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=(id > lit(1)))
  Filter(predicate=((id = lit(2)) AND ((value = lit(b)) AND (id < lit(3))))) (cached=true)
    Filter(predicate=(id > lit(1))) (cached=true)
      InMemorySource(schema=[ColumnField(name='id', data_type=IntegerType), ColumnField(name='value', data_type=StringType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_semantic_predicate_rewrite_basic(local_session):
    source = local_session.create_dataframe(
        {
            "blurb1": [
                "some irrelevant text",
                "more text that doesn't matter",
            ],
            "blurb2": [
                "even more irrelevant text",
                "even more even more irrelevant text",
            ],
            "a_boolean_column": [
                True,
                False,
            ],
            "a_numeric_column": [
                1,
                -1,
            ],
        }
    )
    df = source.filter(
        semantic.predicate("something that references {blurb1} and {blurb2}")
        & (col("a_boolean_column"))
        & (col("a_numeric_column") > 0)
        & semantic.predicate("something that references {blurb2}")
    )
    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=semantic.predicate_b5897940(blurb2))
  Filter(predicate=semantic.predicate_9cfbfb96(blurb1, blurb2))
    Filter(predicate=(a_boolean_column AND (a_numeric_column > lit(0))))
      InMemorySource(schema=[ColumnField(name='blurb1', data_type=StringType), ColumnField(name='blurb2', data_type=StringType), ColumnField(name='a_boolean_column', data_type=BooleanType), ColumnField(name='a_numeric_column', data_type=IntegerType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_semantic_predicate_rewrite_complex(local_session):
    source = local_session.create_dataframe(
        {
            "blurb1": [
                "some irrelvant text",
                "more text that doesn't matter",
            ],
            "blurb2": [
                "even more irrelevant text",
                "even more even more irrelevant text",
            ],
            "a_boolean_column": [
                True,
                False,
            ],
            "a_numeric_column": [
                1,
                -1,
            ],
        }
    )
    df = source.filter(
        (
            semantic.predicate("something that references {blurb1}")
            & semantic.predicate("something that references {blurb2}")
        )
        & (
            (
                semantic.predicate("something that references {blurb1} and {blurb2}")
                | semantic.predicate("something that references {blurb2} and {blurb1}")
            )
        )
    )

    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=(semantic.predicate_9cfbfb96(blurb1, blurb2) OR semantic.predicate_c26d7053(blurb2, blurb1)))
  Filter(predicate=semantic.predicate_b5897940(blurb2))
    Filter(predicate=semantic.predicate_bbe57368(blurb1))
      InMemorySource(schema=[ColumnField(name='blurb1', data_type=StringType), ColumnField(name='blurb2', data_type=StringType), ColumnField(name='a_boolean_column', data_type=BooleanType), ColumnField(name='a_numeric_column', data_type=IntegerType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_semantic_predicate_rewrite_noop(local_session):
    source = local_session.create_dataframe(
        {
            "blurb1": [
                "some irrelvant text",
                "more text that doesn't matter",
            ],
            "a_numeric_column": [
                1,
                -1,
            ],
        }
    )
    df = source.filter(
        (
            semantic.predicate("something that references {blurb1}")
            | (col("a_numeric_column") > 0)
        )
    )

    was_modified = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .was_modified
    )
    assert not was_modified

    df = source.filter(
        (
            semantic.predicate("something that references {blurb1}")
            | (col("a_numeric_column") > 0)
            & semantic.predicate("something else that references {blurb1}")
        )
    )

    was_modified = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .was_modified
    )
    assert not was_modified


def test_semantic_predicate_preserves_cache(local_session):
    source = local_session.create_dataframe(
        {
            "blurb1": [
                "some irrelvant text",
                "more text that doesn't matter",
            ],
            "a_numeric_column": [
                1,
                -1,
            ],
        }
    )
    df = source.filter(
        (
            semantic.predicate("something that references {blurb1}")
            & (col("a_numeric_column") > 0)
        )
    ).cache()
    df = df.select(col("blurb1"))
    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Projection(exprs=[blurb1])
  Filter(predicate=semantic.predicate_bbe57368(blurb1)) (cached=true)
    Filter(predicate=(a_numeric_column > lit(0)))
      InMemorySource(schema=[ColumnField(name='blurb1', data_type=StringType), ColumnField(name='a_numeric_column', data_type=IntegerType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_combine_merge_filters_and_semantic_predicate_rewrite(local_session):
    source = local_session.create_dataframe(
        {
            "blurb1": [
                "some irrelvant text",
                "more text that doesn't matter",
            ],
            "a_numeric_column": [
                1,
                -1,
            ],
        }
    )
    df = source.filter(semantic.predicate("something that references {blurb1}"))
    df = df.filter(col("a_numeric_column") > 0).cache()
    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=semantic.predicate_bbe57368(blurb1)) (cached=true)
  Filter(predicate=(a_numeric_column > lit(0)))
    InMemorySource(schema=[ColumnField(name='blurb1', data_type=StringType), ColumnField(name='a_numeric_column', data_type=IntegerType)])"""
    assert str(plan).strip() == golden_repr.strip()

    # Ensure that rewrites cannot eliminate cached dataframes
    df = source.filter(semantic.predicate("something that references {blurb1}")).cache()
    df = df.filter(col("a_numeric_column") > 0)
    was_modified = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .was_modified
    )
    assert not was_modified


def test_semantic_predicate_rewrite_with_other_semantic_exprs(local_session):
    class TestModel(BaseModel):
        blurb: str = Field(..., description="The blurb")

    source = local_session.create_dataframe(
        {
            "blurb1": ["some irrelevant text", "more text that doesn't matter"],
            "id": [100, 200],
            "status": ["active", "inactive"],
        }
    )
    df = source.filter(
        (
            text.concat(
                lit("banana"), semantic.map("something that references {blurb1}")
            )
            == "banana"
        )
        & semantic.predicate("something else that references {blurb1}")
        & (semantic.analyze_sentiment(col("blurb1")) == "positive")
        & (semantic.classify(col("blurb1"), ["a", "b", "c"]) == "a")
        & (semantic.extract(col("blurb1"), TestModel).blurb == "blurb")
        & (col("id") > 100)
        & (col("status") == "active")
    )
    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=(semantic.extract_cc39f7ea(blurb1)[blurb] = lit(blurb)))
  Filter(predicate=(semantic.classify(blurb1, ['a', 'b', 'c']) = lit(a)))
    Filter(predicate=(semantic.analyze_sentiment(blurb1) = lit(positive)))
      Filter(predicate=semantic.predicate_c22500eb(blurb1))
        Filter(predicate=(concat(lit(banana), semantic.map_bbe57368(blurb1)) = lit(banana)))
          Filter(predicate=((id > lit(100)) AND (status = lit(active))))
            InMemorySource(schema=[ColumnField(name='blurb1', data_type=StringType), ColumnField(name='id', data_type=IntegerType), ColumnField(name='status', data_type=StringType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_semantic_predicate_rewrite_complex_with_other_semantic_exprs(local_session):
    class TestModel(BaseModel):
        blurb: str = Field(..., description="The blurb")

    source = local_session.create_dataframe(
        {
            "blurb1": ["some irrelevant text", "more text that doesn't matter"],
            "blurb2": ["some irrelevant text", "more text that doesn't matter"],
            "a_numeric_column": [1, -1],
            "a_boolean_column": [True, False],
        }
    )
    df = source.filter(
        (
            (
                semantic.predicate("something that references {blurb1}")
                | col("a_boolean_column")
            )
            & (col("a_numeric_column") > 0)
            & (
                (
                    semantic.classify(col("blurb1"), ["a", "b", "c"])
                    == semantic.classify(col("blurb2"), ["a", "b", "c"])
                )
            )
            & (
                (
                    (
                        semantic.extract(col("blurb1"), TestModel).blurb
                        == semantic.extract(col("blurb2"), TestModel).blurb
                    )
                    | semantic.predicate("something that references {blurb2}")
                )
            )
        )
    )
    plan = (
        LogicalPlanOptimizer([MergeFiltersRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=((semantic.extract_efa5a1cb(blurb1)[blurb] = semantic.extract_efa5a1cb(blurb2)[blurb]) OR semantic.predicate_b5897940(blurb2)))
  Filter(predicate=(semantic.classify(blurb1, ['a', 'b', 'c']) = semantic.classify(blurb2, ['a', 'b', 'c'])))
    Filter(predicate=(semantic.predicate_bbe57368(blurb1) OR a_boolean_column))
      Filter(predicate=(a_numeric_column > lit(0)))
        InMemorySource(schema=[ColumnField(name='blurb1', data_type=StringType), ColumnField(name='blurb2', data_type=StringType), ColumnField(name='a_numeric_column', data_type=IntegerType), ColumnField(name='a_boolean_column', data_type=BooleanType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_not_filter_pushdown_simple(local_session):
    source = local_session.create_dataframe(
        {
            "a_boolean_column": [True, False],
        }
    )
    df = source.filter(~(col("a_boolean_column")))
    plan = (
        LogicalPlanOptimizer([NotFilterPushdownRule()]).optimize(df._logical_plan).plan
    )
    golden_repr = """Filter(predicate=NOT a_boolean_column)
  InMemorySource(schema=[ColumnField(name='a_boolean_column', data_type=BooleanType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_not_filter_pushdown_or(local_session):
    source = local_session.create_dataframe(
        {
            "a_boolean_column": [True, False],
            "a_numeric_column": [1, -1],
        }
    )
    df = source.filter(~(col("a_boolean_column") | (col("a_numeric_column") > 0)))
    plan = (
        LogicalPlanOptimizer([NotFilterPushdownRule()]).optimize(df._logical_plan).plan
    )
    golden_repr = """Filter(predicate=(NOT a_boolean_column AND NOT (a_numeric_column > lit(0))))
  InMemorySource(schema=[ColumnField(name='a_boolean_column', data_type=BooleanType), ColumnField(name='a_numeric_column', data_type=IntegerType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_not_filter_pushdown_with_other_semantic_exprs(local_session):
    source = local_session.create_dataframe(
        {
            "a_boolean_column": [True, False],
            "blurb1": ["some irrelevant text", "more text that doesn't matter"],
        }
    )
    df = source.filter(
        ~(
            col("a_boolean_column")
            | (semantic.classify(col("blurb1"), ["a", "b", "c"]) == "a")
        )
    )
    plan = (
        LogicalPlanOptimizer([NotFilterPushdownRule(), SemanticFilterRewriteRule()])
        .optimize(df._logical_plan)
        .plan
    )
    golden_repr = """Filter(predicate=NOT (semantic.classify(blurb1, ['a', 'b', 'c']) = lit(a)))
  Filter(predicate=NOT a_boolean_column)
    InMemorySource(schema=[ColumnField(name='a_boolean_column', data_type=BooleanType), ColumnField(name='blurb1', data_type=StringType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_not_filter_pushdown_and(local_session):
    # Should not push down the NOT to the AND
    source = local_session.create_dataframe(
        {
            "a_boolean_column": [True, False],
            "a_numeric_column": [1, -1],
        }
    )
    df = source.filter(~(col("a_boolean_column") & (col("a_numeric_column") > 0)))
    plan = (
        LogicalPlanOptimizer([NotFilterPushdownRule()]).optimize(df._logical_plan).plan
    )
    golden_repr = """Filter(predicate=NOT (a_boolean_column AND (a_numeric_column > lit(0))))
  InMemorySource(schema=[ColumnField(name='a_boolean_column', data_type=BooleanType), ColumnField(name='a_numeric_column', data_type=IntegerType)])"""
    assert str(plan).strip() == golden_repr.strip()


def test_not_filter_pushdown_nested_ors(local_session):
    source = local_session.create_dataframe(
        {
            "bool_a": [True, False, True, False],
            "bool_b": [False, True, True, False],
            "bool_c": [True, True, False, False],
            "num_x": [10, 20, 30, 40],
        }
    )

    # Create a complex filter with nested expressions and NOT at the top level
    # ~((bool_a & bool_b) | (num_x > 20 & num_y <= 30) | (~bool_c & (num_z == 0 | num_z > 5)))
    complex_filter = ~(
        col("bool_a") | (col("bool_b") | ~(col("bool_c") & (col("num_x") > 15)))
    )

    # Apply the filter
    filtered_df = source.filter(complex_filter)

    # Optimize the plan
    optimizer = LogicalPlanOptimizer([NotFilterPushdownRule()])
    plan = optimizer.optimize(filtered_df._logical_plan).plan
    golden_repr = """Filter(predicate=(NOT bool_a AND (NOT bool_b AND (bool_c AND (num_x > lit(15))))))
  InMemorySource(schema=[ColumnField(name='bool_a', data_type=BooleanType), ColumnField(name='bool_b', data_type=BooleanType), ColumnField(name='bool_c', data_type=BooleanType), ColumnField(name='num_x', data_type=IntegerType)])"""
    assert str(plan).strip() == golden_repr.strip()
