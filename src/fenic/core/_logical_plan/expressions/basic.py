from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List

if TYPE_CHECKING:
    from fenic.core._logical_plan import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.scalar_function import ScalarFunction
from fenic.core.error import PlanError, TypeMismatchError
from fenic.core.types import (
    ArrayType,
    BooleanType,
    ColumnField,
    DataType,
    DocumentPathType,
    DoubleType,
    EmbeddingType,
    FloatType,
    JsonType,
    MarkdownType,
    StringType,
    StructField,
    StructType,
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

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
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


class LiteralExpr(LogicalExpr):
    """Expression representing a literal value."""

    def __init__(self, literal: Any, data_type: DataType):
        self.literal = literal
        self.data_type = data_type

    def __str__(self) -> str:
        return f"lit({self.literal})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return ColumnField(str(self), self.data_type)

    def children(self) -> List[LogicalExpr]:
        return []


class AliasExpr(LogicalExpr):
    """Expression representing a column alias."""

    def __init__(self, expr: LogicalExpr, name: str):
        self.expr = expr
        self.name = name

    def __str__(self) -> str:
        return f"{self.expr} AS {self.name}"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return ColumnField(str(self.name), self.expr.to_column_field(plan).data_type)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class SortExpr(LogicalExpr):
    """Expression representing a column sorted in ascending or descending order."""

    def __init__(self, expr: LogicalExpr, ascending=True, nulls_last=False):
        self.expr = expr
        self.ascending = ascending
        self.nulls_last = nulls_last

    def __str__(self) -> str:
        direction = "asc" if self.ascending else "desc"
        return f"{direction}({self.expr})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return ColumnField(str(self), self.expr.to_column_field(plan).data_type)

    def column_expr(self) -> LogicalExpr:
        return self.expr

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class IndexExpr(LogicalExpr):
    """Expression representing an index or field access operation."""

    def __init__(self, expr: LogicalExpr, index: Any):
        self.expr = expr
        self.index = index

    def __str__(self) -> str:
        return f"{self.expr}[{self.index}]"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        if not isinstance(
            self.expr.to_column_field(plan).data_type, (ArrayType, StructType)
        ):
            raise TypeError(
                f"Type mismatch: Cannot apply get_item to non-array, non-struct types. "
                f"Type: {self.expr.to_column_field(plan).data_type}. "
                f"Only array and struct types are supported."
            )

        if isinstance(
            self.expr.to_column_field(plan).data_type, ArrayType
        ) and isinstance(self.index, int):
            return ColumnField(
                str(self), self.expr.to_column_field(plan).data_type.element_type
            )

        elif isinstance(
            self.expr.to_column_field(plan).data_type, StructType
        ) and isinstance(self.index, str):
            for field in self.expr.to_column_field(plan).data_type.struct_fields:
                if field.name == self.index:
                    return ColumnField(str(self), field.data_type)
            raise ValueError(
                f"Field '{self.index}' not found in struct. Available fields: {', '.join(sorted(f.name for f in self.expr.to_column_field(plan).data_type.struct_fields))}"
            )

        else:
            raise TypeError(
                f"Type mismatch: Cannot apply get_item with {type(self.index).__name__} index to {type(self.expr.to_column_field(plan).data_type).__name__}. "
                f"Array types require integer indices, struct types require string field names."
            )

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ArrayExpr(ScalarFunction):
    function_name = "array"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        super().__init__(*exprs)

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return ArrayType with element type matching the first argument."""
        # Signature validation ensures all args have the same type
        return ArrayType(arg_types[0])


class StructExpr(ScalarFunction):
    function_name = "struct"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        super().__init__(*exprs)

    def _infer_dynamic_return_type(self, arg_types: List[DataType], plan: LogicalPlan) -> DataType:
        """Return StructType with fields based on argument names and types."""
        struct_fields = []
        for (arg, arg_type) in zip(self._children, arg_types, strict=True):
            # Use alias name if available, otherwise use string representation
            field_name = str(arg) if not isinstance(arg, AliasExpr) else arg.name
            struct_fields.append(StructField(field_name, arg_type))
        return StructType(struct_fields)


class UDFExpr(LogicalExpr):
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

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        for arg in self.args:
            _ = arg.to_column_field(plan)
        return ColumnField(str(self), self.return_type)

    def children(self) -> List[LogicalExpr]:
        return self.args


class IsNullExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, is_null: bool):
        self.expr = expr
        self.is_null = is_null

    def __str__(self):
        return f"{self.expr} IS {'' if self.is_null else 'NOT'} NULL"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        return ColumnField(str(self), BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class ArrayLengthExpr(ScalarFunction):
    function_name = "array_size"

    def __init__(self, expr: LogicalExpr):
        self.expr = expr
        super().__init__(expr)

class ArrayContainsExpr(ScalarFunction):
    function_name = "array_contains"

    def __init__(self, expr: LogicalExpr, other: LogicalExpr):
        self.expr = expr
        self.other = other
        super().__init__(expr, other)

class CastExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, dest_type: DataType):
        self.expr = expr
        self.dest_type = dest_type
        self.source_type = None

    def __str__(self):
        return f"cast({self.expr} AS {self.dest_type})"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        self.source_type = self.expr.to_column_field(plan).data_type
        src = self.source_type
        dst = self.dest_type
        if not _can_cast(src, dst):
            raise PlanError(f"Unsupported cast: {src} → {dst}")
        return ColumnField(str(self), dst)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class NotExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr):
        self.expr = expr

    def __str__(self):
        return f"NOT {self.expr}"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        if self.expr.to_column_field(plan).data_type != BooleanType:
            raise TypeError(
                f"Type mismatch: Cannot apply NOT to non-boolean types. "
                f"Type: {self.expr.to_column_field(plan).data_type}. "
                f"Only boolean types are supported."
            )
        return ColumnField(str(self), BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr]


class CoalesceExpr(ScalarFunction):
    function_name = "coalesce"

    def __init__(self, exprs: List[LogicalExpr]):
        self.exprs = exprs
        super().__init__(*exprs)


class InExpr(LogicalExpr):
    def __init__(self, expr: LogicalExpr, other: LogicalExpr):
        self.expr = expr
        self.other = other

    def __str__(self):
        return f"{self.expr} IN {self.other}"

    def to_column_field(self, plan: LogicalPlan) -> ColumnField:
        if not isinstance(self.other.to_column_field(plan).data_type, ArrayType):
            raise TypeMismatchError.from_message(
                f"The 'other' argument to IN must be an ArrayType. "
                f"Got: {self.other.to_column_field(plan).data_type}. "
                f"Expression: {self.expr} IN {self.other}"
            )
        if self.expr.to_column_field(plan).data_type != self.other.to_column_field(plan).data_type.element_type:
            raise TypeMismatchError.from_message(
                f"The element being searched for must match the array's element type. "
                f"Searched element type: {self.expr.to_column_field(plan).data_type}, "
                f"Array element type: {self.other.to_column_field(plan).data_type.element_type}. "
                f"Expression: {self.expr} IN {self.other}"
            )
        return ColumnField(str(self), BooleanType)

    def children(self) -> List[LogicalExpr]:
        return [self.expr, self.other]

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
