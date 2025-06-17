import polars as pl
import pytest

from fenic._backends.local.physical_plan import (
    FilterExec,
    InMemorySourceExec,
    ProjectionExec,
    UnionExec,
)
from fenic._backends.local.transpiler.expr_converter import (
    ExprConverter,
)
from fenic._backends.local.transpiler.plan_converter import (
    PlanConverter,
)
from fenic.core._logical_plan.expressions import (
    AliasExpr,
    ArithmeticExpr,
    BooleanExpr,
    ColumnExpr,
    LiteralExpr,
    NumericComparisonExpr,
    Operator,
)
from fenic.core._logical_plan.plans import (
    Filter,
    InMemorySource,
    Projection,
    Union,
)
from fenic.core.types import IntegerType


def test_convert_column_expr(local_session):
    expr_converter = ExprConverter(local_session._session_state)
    col_expr = ColumnExpr("age")
    physical = expr_converter.convert(col_expr)
    assert isinstance(physical, pl.Expr)


def test_convert_literal_expr(local_session):
    expr_converter = ExprConverter(local_session._session_state)
    lit_expr = LiteralExpr(42, IntegerType)
    physical = expr_converter.convert(lit_expr)
    assert isinstance(physical, pl.Expr)


def test_convert_alias_expr(local_session):
    expr_converter = ExprConverter(local_session._session_state)
    base = ColumnExpr("age")
    alias = AliasExpr(base, "user_age")
    physical = expr_converter.convert(alias)
    assert isinstance(physical, pl.Expr)


def test_convert_arithmetic_expr(local_session):
    expr_converter = ExprConverter(local_session._session_state)
    left = LiteralExpr(10, IntegerType)
    right = LiteralExpr(5, IntegerType)
    add_expr = ArithmeticExpr(left, right, Operator.PLUS)
    physical = expr_converter.convert(add_expr)
    assert isinstance(physical, pl.Expr)


def test_convert_boolean_expr(local_session):
    expr_converter = ExprConverter(local_session._session_state)
    left = ColumnExpr("age")
    right = LiteralExpr(18, IntegerType)
    bool_expr = BooleanExpr(left, right, Operator.GT)
    physical = expr_converter.convert(bool_expr)
    assert isinstance(physical, pl.Expr)


def test_unsupported_expr(local_session):
    expr_converter = ExprConverter(local_session._session_state)
    class UnsupportedExpr:
        pass

    with pytest.raises(NotImplementedError):
        expr_converter.convert(UnsupportedExpr())


def test_convert_source_plan(local_session):
    df = pl.DataFrame({"a": [1, 2, 3]})
    source = InMemorySource(df, local_session._session_state)
    plan_converter = PlanConverter(local_session._session_state)
    physical = plan_converter.convert(
        source,
    )
    assert isinstance(physical, InMemorySourceExec)


def test_convert_projection_plan(local_session):
    df = pl.DataFrame({"a": [1, 2, 3]})
    source = InMemorySource(df, local_session._session_state)
    plan_converter = PlanConverter(local_session._session_state)
    proj = Projection(source, [ColumnExpr("a")])
    physical = plan_converter.convert(
        proj,
    )
    assert isinstance(physical, ProjectionExec)


def test_convert_filter_plan(local_session):
    df = pl.DataFrame({"a": [1, 2, 3]})
    source = InMemorySource(df, local_session._session_state)
    plan_converter = PlanConverter(local_session._session_state)
    filter_expr = NumericComparisonExpr(
        ColumnExpr("a"), LiteralExpr(2, IntegerType), Operator.GT
    )
    filt = Filter(source, filter_expr)
    physical = plan_converter.convert(
        filt,
    )
    assert isinstance(physical, FilterExec)


def test_convert_union_plan(local_session):
    plan_converter = PlanConverter(local_session._session_state)
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [3, 4]})
    source1 = InMemorySource(df1, local_session._session_state)
    source2 = InMemorySource(df2, local_session._session_state)
    union = Union([source1, source2])
    physical = plan_converter.convert(
        union,
    )
    assert isinstance(physical, UnionExec)
