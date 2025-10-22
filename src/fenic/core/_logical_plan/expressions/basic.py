from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Literal, Union

from fenic.core._utils.type_inference import infer_dtype_from_polars

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

import pandas as pd
import polars as pl

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import (
    LogicalExpr,
    UnparameterizedExpr,
    ValidatedDynamicSignature,
    ValidatedSignature,
)
from fenic.core._logical_plan.signatures.signature_validator import SignatureValidator
from fenic.core.error import PlanError, TypeMismatchError, ValidationError
from fenic.core.types import (
    ArrayType,
    BooleanType,
    ColumnField,
    DataType,
    DateType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    IntegerType,
    JsonType,
    MarkdownType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    TranscriptType,
)
from fenic.core.types.datatypes import (
    _HtmlType,
    _PrimitiveType,
)


class ColumnExpr(LogicalExpr):
    """Expression representing a column reference."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        column_field = next(
            (f for f in plan.schema().column_fields if f.name == self.name), None
        )
        if column_field is None:
            raise ValueError(
                f"Column '{self.name}' not found in schema. "
                f"Available columns: {', '.join(sorted(f.name for f in plan.schema().column_fields))}"
            )
        return column_field

    def children(self) -> List[LogicalExpr]:
        return []

    def _eq_specific(self, other: ColumnExpr) -> bool:
        return self.name == other.name


class LiteralExpr(LogicalExpr):
    """Expression representing a literal value."""

    def __init__(self, literal: Any, data_type: DataType):
        self.literal = literal
        self.data_type = data_type

    def __str__(self) -> str:
        return f"lit({self.literal})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        return ColumnField(str(self), self.data_type)

    def children(self) -> List[LogicalExpr]:
        return []

    def _eq_specific(self, other: LiteralExpr) -> bool:
        return self.literal == other.literal and self.data_type == other.data_type


class SeriesLiteralExpr(LogicalExpr):
    """Expression representing a Polars or pandas Series as a literal column.

    This expression allows users to directly add Series data to a DataFrame using
    with_column or with_columns. The Series is stored as a Polars Series internally
    and serialized via Arrow IPC format for cloud backend compatibility.

    Works with both local and cloud backends.
    """

    def __init__(self, series: Union[pl.Series, pd.Series]):
        """Initialize a SeriesLiteralExpr.

        Args:
            series: A Polars or pandas Series to be used as a literal column
        """
        if isinstance(series, pd.Series):
            self.series = pl.from_pandas(series)
        else:
            self.series = series

        if self.series.dtype == pl.Null:
            raise ValidationError(
                "Series cannot contain all nulls unless a dtype is specified. "
                "Use `null(dtype)` (for primitive types) or `empty(ArrayType(...))` or `empty(StructType(...))` (for collections) instead, "
                "or specify a dtype for the Series."
            )
        self.data_type = infer_dtype_from_polars(self.series.dtype)

        if len(self.series) == 0:
            if isinstance(self.data_type, (ArrayType, StructType)):
                raise ValidationError(f"Series length cannot be 0. To add a column with empty values, use `empty({self.data_type})` instead.")
            else:
                raise ValidationError(f"Series length cannot be 0. To add a column with null values, use `null({self.data_type})` instead.")


    def __str__(self) -> str:
        return f"series(len={len(self.series)})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        return ColumnField(str(self), self.data_type)

    def children(self) -> List[LogicalExpr]:
        return []

    def _eq_specific(self, other: SeriesLiteralExpr) -> bool:
        try:
            return self.series.equals(other.series)
        except Exception:
            return False


class UnresolvedLiteralExpr(LogicalExpr):
    def __init__(self, data_type: DataType, parameter_name: str):
        self.data_type = data_type
        self.parameter_name = parameter_name


    def __str__(self) -> str:
        return f"param({self.parameter_name})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        return ColumnField(str(self), self.data_type)

    def children(self) -> List[LogicalExpr]:
        return []

    def _eq_specific(self, other: UnresolvedLiteralExpr) -> bool:
        return (self.parameter_name == other.parameter_name and
                self.data_type == other.data_type)


class AliasExpr(LogicalExpr):
    """Expression representing a column alias."""

    def __init__(self, expr: LogicalExpr, name: str):
        self.expr = expr
        self.name = name

    def __str__(self) -> str:
        return f"{self.expr} AS {self.name}"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        return ColumnField(str(self.name), self.expr.to_column_field(plan, session_state).data_type)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: AliasExpr) -> bool:
        return self.name == other.name

class SortExpr(LogicalExpr):
    """Expression representing a column sorted in ascending or descending order."""

    def __init__(self, expr: LogicalExpr, ascending=True, nulls_last=False):
        self.expr = expr
        self.ascending = ascending
        self.nulls_last = nulls_last

    def __str__(self) -> str:
        direction = "asc" if self.ascending else "desc"
        return f"{direction}({self.expr})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        return ColumnField(str(self), self.expr.to_column_field(plan, session_state).data_type)

    def column_expr(self) -> LogicalExpr:
        return self.expr

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: SortExpr) -> bool:
        return self.ascending == other.ascending and self.nulls_last == other.nulls_last

class IndexExpr(UnparameterizedExpr, LogicalExpr):
    """Expression representing an index or field access operation."""

    def __init__(self, expr: LogicalExpr, index: LogicalExpr):
        self.expr = expr
        self.index = index
        self.input_type: Literal["array", "struct"] = None

    def __str__(self) -> str:
        return f"{self.expr}[{self.index}]"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        expr_field = self.expr.to_column_field(plan, session_state)
        index_field = self.index.to_column_field(plan, session_state)
        expr_type = expr_field.data_type
        index_type = index_field.data_type

        if isinstance(expr_type, ArrayType):
            self.input_type = "array"
            if index_type != IntegerType:
                raise TypeMismatchError.from_message(
                    f"Expected IntegerType index for array access, but got {index_type}."
                )
            return ColumnField(str(self), expr_type.element_type)

        elif isinstance(expr_type, StructType):
            self.input_type = "struct"
            if not isinstance(self.index, LiteralExpr):
                raise TypeMismatchError.from_message(
                    "Struct field access requires a literal string index (e.g. 'field' or fc.lit('field'))."
                )
            if self.index.data_type != StringType:
                raise TypeMismatchError.from_message(
                    f"Expected StringType index for struct access, but got {self.index.data_type}."
                )
            for field in expr_type.struct_fields:
                if field.name == self.index.literal:
                    return ColumnField(str(self), field.data_type)
            available = ', '.join(sorted(f.name for f in expr_type.struct_fields))
            raise ValidationError(
                f"Field '{self.index.literal}' not found in struct. Available fields: {available}."
            )

        else:
            raise TypeMismatchError.from_message(
                f"get_item cannot be applied to type {expr_type}. Supported types: ArrayType, StructType."
            )

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.index]


class ArrayExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    """Expression representing array creation from multiple columns."""

    function_name = "array"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self.exprs

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return ArrayType with element type matching the first argument."""
        # Signature validation ensures all args have the same type
        return ArrayType(arg_types[0])


class StructExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    """Expression representing struct creation from multiple columns."""

    function_name = "struct"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self.exprs

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return StructType with fields based on argument names and types."""
        struct_fields = []
        for (arg, arg_type) in zip(self.children(), arg_types, strict=True):
            # Use alias name if available, otherwise use string representation
            field_name = str(arg) if not isinstance(arg, AliasExpr) else arg.name
            struct_fields.append(StructField(field_name, arg_type))
        return StructType(struct_fields)


class UDFExpr(LogicalExpr):
    """User-defined function expression.

    Warning:
        UDFExpr cannot be serialized and is not supported in cloud execution.
        This expression contains arbitrary Python code that cannot be transmitted
        to remote workers. Use built-in fenic functions for cloud compatibility.
    """

    function_name = "udf"
    def __init__(
        self,
        func: Callable,
        args: List[LogicalExpr],
        return_type: DataType,
    ):
        self.func = func
        self.args = args
        self.return_type = return_type

    def __str__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.func.__name__}({args_str})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        for arg in self.args:
            _ = arg.to_column_field(plan, session_state)
        return ColumnField(str(self), self.return_type)

    def children(self) -> List[LogicalExpr]:
        return self.args

    def _eq_specific(self, other: UDFExpr) -> bool:
        # For dynamic UDFs, we can only check identity since its tricky to compare Callables
        return self.func is other.func and self.return_type == other.return_type


class AsyncUDFExpr(LogicalExpr):
    """Expression for async user-defined functions with configurable concurrency and retries."""

    def __init__(
        self,
        func: Callable,
        args: List[LogicalExpr],
        return_type: DataType,
        max_concurrency: int = 10,
        timeout_seconds: float = 30,
        num_retries: int = 0,
    ):
        self.func = func
        self.args = args
        self.return_type = return_type
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.num_retries = num_retries

    def __str__(self):
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.func.__name__}({args_str})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        for arg in self.args:
            _ = arg.to_column_field(plan, session_state)
        return ColumnField(str(self), self.return_type)

    def children(self) -> List[LogicalExpr]:
        return self.args

    def _eq_specific(self, other: AsyncUDFExpr) -> bool:
        # For dynamic UDFs, we can only check identity since its tricky to compare Callables
        return (
            self.func is other.func
            and self.return_type == other.return_type
            and self.max_concurrency == other.max_concurrency
            and self.timeout_seconds == other.timeout_seconds
            and self.num_retries == other.num_retries
        )


class IsNullExpr(LogicalExpr):

    def __init__(self, expr: LogicalExpr, is_null: bool):
        self.expr = expr
        self.is_null = is_null

    def __str__(self):
        return f"{self.expr} IS {'' if self.is_null else 'NOT'} NULL"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        _ = self.expr.to_column_field(plan, session_state)
        return ColumnField(str(self), BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: IsNullExpr) -> bool:
        return self.is_null == other.is_null


class FlattenExpr(ValidatedDynamicSignature, UnparameterizedExpr, LogicalExpr):
    function_name = "flatten"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan, session_state: BaseSessionState) -> DataType:
        """Return array of inner element type (flatten one level)."""
        array_type = arg_types[0]
        if isinstance(array_type, ArrayType) and isinstance(array_type.element_type, ArrayType):
            return ArrayType(array_type.element_type.element_type)
        raise TypeError(f"Expected Array<Array<T>>, got {array_type}")


class CastExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, dest_type: DataType):
        self.expr = expr
        self.dest_type = dest_type
        self.source_type = None

    def __str__(self):
        return f"cast({self.expr} AS {self.dest_type})"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        self.source_type = self.expr.to_column_field(plan, session_state).data_type
        src = self.source_type
        dst = self.dest_type
        if not _can_cast(src, dst):
            raise PlanError(f"Unsupported cast: {src} → {dst}")
        return ColumnField(str(self), dst)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]

    def _eq_specific(self, other: CastExpr) -> bool:
        return self.dest_type == other.dest_type


class NotExpr(UnparameterizedExpr, LogicalExpr):
    def __init__(self, expr: LogicalExpr):
        self.expr = expr

    def __str__(self):
        return f"NOT {self.expr}"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        if self.expr.to_column_field(plan, session_state).data_type != BooleanType:
            raise TypeError(
                f"Type mismatch: Cannot apply NOT to non-boolean types. "
                f"Type: {self.expr.to_column_field(plan, session_state).data_type}. "
                f"Only boolean types are supported."
            )
        return ColumnField(str(self), BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class CoalesceExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression representing coalesce operation (first non-null value)."""

    function_name = "coalesce"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self.exprs


class InExpr(UnparameterizedExpr, LogicalExpr):
    def __init__(self, expr: LogicalExpr, other: LogicalExpr):
        self.expr = expr
        self.other = other

    def __str__(self):
        return f"{self.expr} IN {self.other}"

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        if not isinstance(self.other.to_column_field(plan, session_state).data_type, ArrayType):
            raise TypeMismatchError.from_message(
                f"The 'other' argument to IN must be an ArrayType. "
                f"Got: {self.other.to_column_field(plan, session_state).data_type}. "
                f"Expression: {self.expr} IN {self.other}"
            )
        if self.expr.to_column_field(plan, session_state).data_type != self.other.to_column_field(plan, session_state).data_type.element_type:
            raise TypeMismatchError.from_message(
                f"The element being searched for must match the array's element type. "
                f"Searched element type: {self.expr.to_column_field(plan, session_state).data_type}, "
                f"Array element type: {self.other.to_column_field(plan, session_state).data_type.element_type}. "
                f"Expression: {self.expr} IN {self.other}"
            )
        return ColumnField(str(self), BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.other]

class GreatestExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression representing the greatest value of a list of expressions."""

    function_name = "greatest"
    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self.exprs

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        column_field = super().to_column_field(plan, session_state)
        first_expr_type = self.exprs[0].to_column_field(plan, session_state).data_type
        if not isinstance(first_expr_type, _PrimitiveType):
            raise TypeMismatchError.from_message(
                f"fc.greatest() only supports primitive types (StringType, BooleanType, "
                f"FloatType, IntegerType, etc). Got: {first_expr_type} instead. "
            )
        return column_field

class LeastExpr(ValidatedSignature, UnparameterizedExpr, LogicalExpr):
    """Expression representing the least value of a list of expressions."""

    function_name = "least"
    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        self._validator = SignatureValidator(self.function_name)

    @property
    def validator(self) -> SignatureValidator:
        return self._validator

    def children(self) -> List[LogicalExpr]:
        return self.exprs

    def to_column_field(self, plan: LogicalPlan, session_state: BaseSessionState) -> ColumnField:
        column_field = super().to_column_field(plan, session_state)
        first_expr_type = self.exprs[0].to_column_field(plan, session_state).data_type
        if not isinstance(first_expr_type, _PrimitiveType):
            raise TypeMismatchError.from_message(
                f"fc.least() only supports primitive types (StringType, BooleanType, "
                f"FloatType, IntegerType, etc). Got: {first_expr_type} instead. "
            )

        return column_field

UNIMPLEMENTED_TYPES = (_HtmlType, TranscriptType, DocumentPathType)
def _can_cast(src: DataType, dst: DataType) -> bool:
    if type(src) in UNIMPLEMENTED_TYPES or type(dst) in UNIMPLEMENTED_TYPES:
        raise NotImplementedError(f"Unimplemented type: Cannot cast {src} → {dst}")

    if isinstance(src, EmbeddingType):
        return NotImplementedError(f"Unimplemented type: Cannot cast {src} → {dst}")

    if (src == ArrayType(element_type=FloatType) or src == ArrayType(element_type=DoubleType)) and isinstance(dst, EmbeddingType):
        return True

    if src == dst:
        return True

    if dst == MarkdownType:
        return _can_cast(src, StringType)

    if src == MarkdownType:
        return _can_cast(StringType, dst)

    if dst == JsonType or src == JsonType:
        return True

    if isinstance(src, _PrimitiveType) and isinstance(dst, _PrimitiveType):
        # Disallow string → bool
        if src == StringType and dst == BooleanType:
            return False
        if src == BooleanType and (dst == DateType or dst == TimestampType):
            # Disallow bool → date or timestamp
            # The results are confusing and not useful.
            # e.g. True -> 1970-01-02 (Jan 2, 1970)
            return False
        if (src == DateType or src == TimestampType) and (dst == BooleanType):
            return False
        return True

    if isinstance(src, ArrayType) and isinstance(dst, ArrayType):
        return _can_cast(src.element_type, dst.element_type)

    if isinstance(src, StructType) and isinstance(dst, StructType):
        src_fields = {f.name: f.data_type for f in src.struct_fields}
        dst_fields = {f.name: f.data_type for f in dst.struct_fields}
        for name, dst_type in dst_fields.items():
            if name in src_fields and not _can_cast(src_fields[name], dst_type):
                return False
        return True

    return False
