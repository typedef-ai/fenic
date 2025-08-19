"""Core parameterized views primitives for Fenic.

This module provides building blocks for parameterized views:
- ViewParameter: describes a single parameter
- ParameterizedView: executes a base view with parameter-driven filters
- ViewFilters: common reusable filter functions
- Factory helpers: string_param, int_param, float_param, bool_param, enum_param

This module intentionally has no dependencies on FastMCP or Pydantic so it can
be used in any environment. Integration layers (e.g., MCP generators) should
import these primitives and compose external concerns separately.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, get_origin

from fenic.api.dataframe.dataframe import DataFrame
from fenic.api.functions.core import col, lit
from fenic.core.error import TypeMismatchError


@dataclass
class QueryParameter:
    """A parameter for a parameterized view.

    Attributes:
        name: Parameter key name
        type: Expected Python type (supports list types via typing.List)
        description: Human-readable description of the parameter
        filter_fn: Callable that applies the parameter to a DataFrame
        required: Whether the parameter is required
        default: Default value when not provided
        enum_values: Optional set of allowed values (for lists: each element validated)
    """

    name: str
    type: Type
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum_values: Optional[List[Any]] = None


@dataclass
class ParamaterizedQuery:
    """A view with parameters that can be used to construct queries dynamically."""

    name: str
    description: str
    base_view: str
    parameters: Dict[str, QueryParameter] = field(default_factory=dict)
    parameter_mapping: Dict[str, str] = field(default_factory=dict)

    def execute(self, session: Any, **kwargs) -> DataFrame:
        """Execute the view with the given parameters against a session."""
        # Local import to avoid circular import at module import time
        from fenic.api.session.session import Session  # noqa: F401

        self._validate_parameters(kwargs)
        validated_params = {k: v for k, v in kwargs.items()}
        df = session.view(self.base_view)
        for param_name, value in validated_params.items():
            if value is None:
                continue
            param = self.parameters.get(param_name)
            if param is None:
                continue
            mapped_column = self.parameter_mapping.get(param_name, param_name)
            ctx = FilterContext(
                param_name=param_name,
                column=mapped_column,
                parameters=validated_params,
                session=session,
            )
            df = param.filter_fn(df, value, ctx)
        return df

    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate that required parameters are present and correctly typed."""
        for param_name, param in self.parameters.items():
            if param.required and param_name not in parameters:
                raise ValueError(f"Required parameter '{param_name}' not provided")
            if param_name not in parameters:
                continue
            value = parameters[param_name]
            if value is None:
                continue
            origin = get_origin(param.type)
            # Handle list types
            if origin is list or origin is List:
                if not isinstance(value, list):
                    raise TypeMismatchError(f"Parameter '{param_name}' expects a list, got {type(value).__name__}")
                if param.enum_values:
                    invalid = [v for v in value if v not in param.enum_values]
                    if invalid:
                        raise ValueError(
                            f"Parameter '{param_name}' contains invalid values: {invalid}. "
                            f"Valid values are: {param.enum_values}"
                        )
            else:
                if not isinstance(value, param.type):
                    raise TypeMismatchError(
                        f"Parameter '{param_name}' expects {param.type.__name__}, got {type(value).__name__}"
                    )
                if param.enum_values and value not in param.enum_values:
                    raise ValueError(
                        f"Parameter '{param_name}' must be one of {param.enum_values}, got {value}"
                    )


class ViewFilters:
    """Common filter operations for view parameters.

    All filter functions accept (df, value, ctx) to allow accessing multiple
    columns or other parameters when needed.
    """

    @staticmethod
    def equals(df: DataFrame, value: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("equals filter requires a mapped column in context")
        return df.filter(col(ctx.column) == value)

    @staticmethod
    def not_equals(df: DataFrame, value: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("not_equals filter requires a mapped column in context")
        return df.filter(col(ctx.column) != value)

    @staticmethod
    def greater_than(df: DataFrame, value: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("greater_than filter requires a mapped column in context")
        return df.filter(col(ctx.column) > value)

    @staticmethod
    def greater_equal(df: DataFrame, value: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("greater_equal filter requires a mapped column in context")
        return df.filter(col(ctx.column) >= value)

    @staticmethod
    def less_than(df: DataFrame, value: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("less_than filter requires a mapped column in context")
        return df.filter(col(ctx.column) < value)

    @staticmethod
    def less_equal(df: DataFrame, value: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("less_equal filter requires a mapped column in context")
        return df.filter(col(ctx.column) <= value)

    @staticmethod
    def in_list(df: DataFrame, value: List[Any], ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("in_list filter requires a mapped column in context")
        return df.filter(col(ctx.column).is_in(lit(value)))

    @staticmethod
    def not_in_list(df: DataFrame, value: List[Any], ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("not_in_list filter requires a mapped column in context")
        return df.filter(~col(ctx.column).is_in(lit(value)))

    @staticmethod
    def like(df: DataFrame, value: str, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("like filter requires a mapped column in context")
        return df.filter(col(ctx.column).like(value))

    @staticmethod
    def ilike(df: DataFrame, value: str, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("ilike filter requires a mapped column in context")
        return df.filter(col(ctx.column).ilike(value))

    @staticmethod
    def rlike(df: DataFrame, value: str, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("rlike filter requires a mapped column in context")
        return df.filter(col(ctx.column).rlike(value))

    @staticmethod
    def contains(df: DataFrame, value: str, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("contains filter requires a mapped column in context")
        return df.filter(col(ctx.column).contains(value))

    @staticmethod
    def contains_any(df: DataFrame, value: List[str], ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("contains_any filter requires a mapped column in context")
        return df.filter(col(ctx.column).contains_any(lit(value)))

    @staticmethod
    def starts_with(df: DataFrame, value: str, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("starts_with filter requires a mapped column in context")
        return df.filter(col(ctx.column).starts_with(value))

    @staticmethod
    def ends_with(df: DataFrame, value: str, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("ends_with filter requires a mapped column in context")
        return df.filter(col(ctx.column).ends_with(value))

    @staticmethod
    def is_null(df: DataFrame, _: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("is_null filter requires a mapped column in context")
        return df.filter(col(ctx.column).is_null())

    @staticmethod
    def is_not_null(df: DataFrame, _: Any, ctx: FilterContext) -> DataFrame:
        if not ctx.column:
            raise ValueError("is_not_null filter requires a mapped column in context")
        return df.filter(col(ctx.column).is_not_null())


# Factory helpers

def string_param(name: str, description: str, filter_fn: Callable = ViewFilters.equals, **kwargs) -> QueryParameter:
    return QueryParameter(name, str, description, filter_fn, **kwargs)


def int_param(name: str, description: str, filter_fn: Callable = ViewFilters.equals, **kwargs) -> QueryParameter:
    return QueryParameter(name, int, description, filter_fn, **kwargs)


def float_param(name: str, description: str, filter_fn: Callable = ViewFilters.equals, **kwargs) -> QueryParameter:
    return QueryParameter(name, float, description, filter_fn, **kwargs)


def bool_param(name: str, description: str, filter_fn: Callable = ViewFilters.equals, **kwargs) -> QueryParameter:
    return QueryParameter(name, bool, description, filter_fn, **kwargs)


def enum_param(
    name: str,
    description: str,
    values: List[str],
    filter_fn: Callable = ViewFilters.in_list,
    **kwargs,
) -> QueryParameter:
    return QueryParameter(name, List[str], description, filter_fn, enum_values=values, **kwargs)
