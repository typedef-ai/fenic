from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import duckdb
import polars as pl
import sqlglot.errors
import sqlglot.expressions as sqlglot_exprs

from fenic._constants import SQL_PLACEHOLDER_RE
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions import (
    ColumnExpr,
    LogicalExpr,
    SortExpr,
)
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._logical_plan.utils import validate_scalar_expr
from fenic.core._utils.misc import generate_unique_view_name
from fenic.core._utils.schema import (
    convert_custom_schema_to_polars_schema,
    convert_polars_schema_to_custom_schema,
)
from fenic.core.error import InternalError, PlanError, TypeMismatchError
from fenic.core.types import (
    ArrayType,
    BooleanType,
    ColumnField,
    EmbeddingType,
    IntegerType,
    Schema,
    StructType,
)

logger = logging.getLogger(__name__)

class Projection(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            exprs: List[LogicalExpr],
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._input = input
        for expr in exprs:
            validate_scalar_expr(expr, "projection")
        self._exprs = exprs
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, exprs: List[LogicalExpr], schema: Schema) -> Projection:
        return Projection(input, exprs, None, schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, exprs: List[LogicalExpr], session_state: BaseSessionState) -> Projection:
        return Projection(input, exprs, session_state, None)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        fields = []
        for expr in self._exprs:
            fields.append(expr.to_column_field(self._input, session_state))
        return Schema(fields)

    def exprs(self) -> List[LogicalExpr]:
        return self._exprs

    def _repr(self) -> str:
        exprs_str = ", ".join(str(expr) for expr in self._exprs)
        return f"Projection(exprs=[{exprs_str}])"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Projection must have exactly one child")
        result = self.from_session_state(children[0], self._exprs, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Projection) -> bool:
        return (
            len(self._exprs) == len(other._exprs)
            and all(expr1 == expr2 for expr1, expr2 in zip(self._exprs, other._exprs, strict=True))
        )

class Filter(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            predicate: LogicalExpr,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._input = input
        validate_scalar_expr(predicate, "filter")
        self._predicate = predicate
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, predicate: LogicalExpr, schema: Schema) -> Filter:
        return Filter(input, predicate, schema=schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, predicate: LogicalExpr, session_state: BaseSessionState) -> Filter:
        return Filter(input, predicate, session_state=session_state)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        actual_type = self._predicate.to_column_field(self._input, session_state).data_type
        if actual_type != BooleanType:
            raise PlanError(
                f"Filter predicate must return a boolean value, but got {actual_type}. "
                "Examples of valid filters:\n"
                "- df.filter(col('age') > 18)\n"
                "- df.filter(col('status') == 'active')\n"
                "- df.filter(col('is_valid'))"
            )
        return self._input.schema()

    def predicate(self) -> LogicalExpr:
        return self._predicate

    def exprs(self) -> List[LogicalExpr]:
        return [self._predicate]

    def _repr(self) -> str:
        """Return the representation for the Filter node."""
        return f"Filter(predicate={self._predicate})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Filter must have exactly one child")
        result = self.from_session_state(children[0], self._predicate, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Filter) -> bool:
        return self._predicate == other._predicate

class Union(LogicalPlan):
    def __init__(
            self,
            inputs: List[LogicalPlan],
            session_state:Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._inputs = inputs
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, inputs: List[LogicalPlan], schema: Schema) -> Union:
        return Union(inputs, None, schema)

    @classmethod
    def from_session_state(cls, inputs: List[LogicalPlan], session_state: BaseSessionState) -> Union:
        return Union(inputs, session_state, None)

    def children(self) -> List[LogicalPlan]:
        return self._inputs

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        schemas = [input_plan.schema() for input_plan in self._inputs]

        # Check that all schemas have the same columns and types (order doesn't matter)
        first_schema = schemas[0]
        first_schema_fields = {f.name: f.data_type for f in first_schema.column_fields}

        for i, schema in enumerate(schemas[1:], start=1):
            schema_fields = {f.name: f.data_type for f in schema.column_fields}

            if schema_fields != first_schema_fields:
                # Provide detailed diagnostics
                first_names = set(first_schema_fields.keys())
                current_names = set(schema_fields.keys())

                if first_names != current_names:
                    missing = sorted(first_names - current_names)
                    extra = sorted(current_names - first_names)

                    error_parts = [
                        f"Cannot union DataFrames: DataFrame #{i} has different columns than DataFrame #0."
                    ]

                    if missing:
                        error_parts.append(f"Missing columns: {missing}")
                    if extra:
                        error_parts.append(f"Extra columns: {extra}")

                    error_parts.extend([
                        "",
                        f"DataFrame #0 schema:\n{first_schema}",
                        "",
                        f"DataFrame #{i} schema:\n{schema}",
                        "",
                        "All DataFrames in a union must have exactly the same columns. "
                        "Consider using .select() to ensure matching columns before the union operation. "
                    ])

                    raise PlanError("\n".join(error_parts))

                # Same columns but different types
                type_mismatches = []
                for name in sorted(first_names):
                    if first_schema_fields[name] != schema_fields[name]:
                        type_mismatches.append(
                            f"  - Column '{name}': DataFrame #0 has {first_schema_fields[name]}, "
                            f"DataFrame #{i} has {schema_fields[name]}"
                        )

                if type_mismatches:
                    error_message = (
                        f"Cannot union DataFrames: DataFrame #{i} has incompatible column types.\n\n"
                        f"Type mismatches:\n" + "\n".join(type_mismatches) + "\n\n"
                        f"DataFrame #0 schema:\n{first_schema}\n\n"
                        f"DataFrame #{i} schema:\n{schema}\n\n"
                        f"All DataFrames in a union must have matching column types. "
                        f"Consider using .cast() to align column types before the union operation."
                    )
                    raise PlanError(error_message)

        return schemas[0]

    def _repr(self) -> str:
        return "Union"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        result = Union.from_session_state(children, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Union) -> bool:
        return True  # Union has no specific attributes beyond children


class Limit(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            n: int,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._input = input
        self.n = n
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, n: int, schema: Schema) -> Limit:
        return Limit(input, n, schema=schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, n: int, session_state: BaseSessionState) -> Limit:
        return Limit(input, n, session_state=session_state)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        return self._input.schema()

    def _repr(self) -> str:
        return f"Limit(n={self.n})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Limit must have exactly one child")
        result = self.from_session_state(children[0], self.n, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Limit) -> bool:
        return self.n == other.n


class Explode(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            expr: LogicalExpr,
            keep_null_and_empty: bool = False,
            session_state:Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        validate_scalar_expr(expr, "explode")
        self._input = input
        self._expr = expr
        self.keep_null_and_empty = keep_null_and_empty
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, expr: LogicalExpr, schema: Schema,
                    keep_null_and_empty: bool = False) -> Explode:
        return Explode(input, expr, keep_null_and_empty, schema=schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, expr: LogicalExpr, session_state: BaseSessionState,
                          keep_null_and_empty: bool = False) -> Explode:
        return Explode(input, expr, keep_null_and_empty, session_state=session_state)

    def children(self) -> list[LogicalPlan]:
        return [self._input]

    def exprs(self) -> List[LogicalExpr]:
        return [self._expr]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        input_schema = self._input.schema()
        exploded_field = self._expr.to_column_field(self._input, session_state)

        if not isinstance(exploded_field.data_type, ArrayType):
            raise ValueError(
                f"Explode operator expected an array column for field {exploded_field.name}, "
                f"but received {exploded_field.data_type} instead."
            )

        new_fields = []
        for field in input_schema.column_fields:
            if field.name == exploded_field.name:
                new_field = ColumnField(
                    name=field.name, data_type=exploded_field.data_type.element_type
                )
                new_fields.append(new_field)
            else:
                new_fields.append(field)

        if exploded_field.name not in {f.name for f in input_schema.column_fields}:
            new_field = ColumnField(
                name=exploded_field.name,
                data_type=exploded_field.data_type.element_type,
            )
            new_fields.append(new_field)

        return Schema(column_fields=new_fields)

    def _repr(self) -> str:
        return f"Explode({self._expr})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Explode must have exactly one child")
        result = Explode.from_session_state(children[0], self._expr, session_state, self.keep_null_and_empty)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Explode) -> bool:
        return self._expr == other._expr and self.keep_null_and_empty == other.keep_null_and_empty


class ExplodeWithIndex(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            expr: LogicalExpr,
            index_name: str,
            value_name: str,
            keep_null_and_empty: bool = False,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        validate_scalar_expr(expr, "explode_with_index")
        self._input = input
        self._expr = expr
        self.index_name = index_name
        self.value_name = value_name
        self.keep_null_and_empty = keep_null_and_empty
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, expr: LogicalExpr, index_name: str,
                    value_name: str, schema: Schema,
                    keep_null_and_empty: bool = False) -> ExplodeWithIndex:
        return ExplodeWithIndex(input, expr, index_name, value_name, keep_null_and_empty, schema=schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, expr: LogicalExpr, index_name: str,
                          value_name: str, session_state: BaseSessionState,
                          keep_null_and_empty: bool = False) -> ExplodeWithIndex:
        return ExplodeWithIndex(input, expr, index_name, value_name, keep_null_and_empty,
                               session_state=session_state)

    def children(self) -> list[LogicalPlan]:
        return [self._input]

    def exprs(self) -> List[LogicalExpr]:
        return [self._expr]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        input_schema = self._input.schema()
        exploded_field = self._expr.to_column_field(self._input, session_state)

        if not isinstance(exploded_field.data_type, ArrayType):
            raise ValueError(
                f"ExplodeWithIndex operator expected an array column for field {exploded_field.name}, "
                f"but received {exploded_field.data_type} instead."
            )

        # Determine the value column name

        new_fields = []
        # Add index column first (0-based position within array)
        new_fields.append(ColumnField(name=self.index_name, data_type=IntegerType))
        exploded_field_processed = False
        # Add or replace the exploded column
        for field in input_schema.column_fields:
            if field.name == exploded_field.name:
                exploded_field_processed = True
                if self.value_name is not None and self.value_name != field.name:
                    # Keep original column and add new value column
                    new_fields.append(field)
                    new_fields.append(ColumnField(
                        name=self.value_name,
                        data_type=exploded_field.data_type.element_type
                    ))
                else:
                    # Replace the array column with its element type
                    new_fields.append(ColumnField(
                        name=self.value_name,
                        data_type=exploded_field.data_type.element_type
                    ))
            else:
                new_fields.append(field)

        # If the exploded column wasn't in the original schema, add it
        if not exploded_field_processed:
            new_fields.append(ColumnField(
                name=self.value_name,
                data_type=exploded_field.data_type.element_type,
            ))

        return Schema(column_fields=new_fields)

    def _repr(self) -> str:
        return f"ExplodeWithIndex({self._expr}, index={self.index_name}, value={self.value_name})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("ExplodeWithIndex must have exactly one child")
        result = ExplodeWithIndex.from_session_state(
            children[0], self._expr, self.index_name, self.value_name, session_state, self.keep_null_and_empty
        )
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: ExplodeWithIndex) -> bool:
        return (self._expr == other._expr and
                self.index_name == other.index_name and
                self.value_name == other.value_name and
                self.keep_null_and_empty == other.keep_null_and_empty)


class DropDuplicates(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            subset: List[ColumnExpr],
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._input = input
        self.subset = subset
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, subset: List[ColumnExpr], schema: Schema) -> DropDuplicates:
        return DropDuplicates(input, subset, None, schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, subset: List[ColumnExpr], session_state: BaseSessionState) -> DropDuplicates:
        return DropDuplicates(input, subset, session_state, None)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        return self._input.schema()

    def exprs(self) -> List[LogicalExpr]:
        return list(self.subset)

    def _repr(self) -> str:
        return f"DropDuplicates(subset={', '.join(str(expr) for expr in self.subset)})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("DropDuplicates must have exactly one child")
        result = DropDuplicates.from_session_state(children[0], self.subset, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: DropDuplicates) -> bool:
        return (
            len(self.subset) == len(other.subset)
            and all(expr1 == expr2 for expr1, expr2 in zip(self.subset, other.subset, strict=True))
        )

    def _subset(self) -> List[str]:
        subset: List[str] = []
        for col in self.subset:
            subset.append(str(col))
        return subset


class Sort(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            sort_exprs: List[SortExpr],
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._input = input
        self._sort_exprs = sort_exprs
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, sort_exprs: List[SortExpr], schema: Schema) -> Sort:
        return Sort(input, sort_exprs, schema=schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, sort_exprs: List[SortExpr], session_state: BaseSessionState) -> Sort:
        return Sort(input, sort_exprs, session_state=session_state)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        return self._input.schema()

    def sort_exprs(self) -> List[LogicalExpr]:
        return self._sort_exprs

    def exprs(self) -> List[LogicalExpr]:
        return list(self._sort_exprs)

    def _repr(self) -> str:
        return f"Sort(cols={', '.join(str(expr) for expr in self._sort_exprs)})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Sort must have exactly one child")
        result = self.from_session_state(children[0], self._sort_exprs, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Sort) -> bool:
        return (
            len(self._sort_exprs) == len(other._sort_exprs)
            and all(expr1 == expr2 for expr1, expr2 in zip(self._sort_exprs, other._sort_exprs, strict=True))
        )

class Unnest(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            exprs: List[ColumnExpr],
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._input = input
        self._exprs = exprs
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, input: LogicalPlan, exprs: List[ColumnExpr], schema: Schema) -> Unnest:
        return Unnest(input, exprs, None, schema)

    @classmethod
    def from_session_state(cls, input: LogicalPlan, exprs: List[ColumnExpr], session_state: BaseSessionState) -> Unnest:
        return Unnest(input, exprs, session_state, None)

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        column_fields = []
        for field in self._input.schema().column_fields:
            if field.name in {expr.name for expr in self._exprs}:
                if isinstance(field.data_type, StructType):
                    for field_ in field.data_type.struct_fields:
                        column_fields.append(
                            ColumnField(name=field_.name, data_type=field_.data_type)
                        )
                else:
                    raise ValueError(
                        f"Unnest operator expected a struct column for field {field.name}, "
                        f"but received {field.data_type} instead."
                    )
            else:
                column_fields.append(field)
        return Schema(column_fields)

    def exprs(self) -> List[LogicalExpr]:
        return list(self._exprs)

    def _repr(self) -> str:
        return f"Unnest(exprs={', '.join(str(expr) for expr in self._exprs)})"

    def col_names(self) -> List[str]:
        return [expr.name for expr in self._exprs]

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("Unnest must have exactly one child")
        result = Unnest.from_session_state(children[0], self._exprs, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: Unnest) -> bool:
        return (
            len(self._exprs) == len(other._exprs)
            and all(expr1 == expr2 for expr1, expr2 in zip(self._exprs, other._exprs, strict=True))
        )


class SQL(LogicalPlan):
    def __init__(
            self,
            inputs: List[LogicalPlan],
            template_names: List[str],
            templated_query: str,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        # Note: inputs[i] corresponds to template_names[i]
        if len(inputs) != len(template_names):
            raise InternalError("inputs and template_names must have the same length")
        self._inputs = inputs
        self._template_names = template_names
        self._templated_query = templated_query
        self.resolved_query, self.view_names = self._replace_query_placeholders()
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, inputs: List[LogicalPlan], template_names: List[str], templated_query: str, schema: Schema) -> SQL:
        return SQL(inputs, template_names, templated_query, None, schema)

    @classmethod
    def from_session_state(cls,
        inputs: List[LogicalPlan],
        template_names: List[str],
        templated_query: str,
        session_state: BaseSessionState) -> SQL:
        return SQL(inputs, template_names, templated_query, session_state, None)

    def children(self) -> List[LogicalPlan]:
        return self._inputs

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _repr(self) -> str:
        return f"SQL(query={self._templated_query})"

    def _replace_query_placeholders(self) -> Tuple[str, Dict[str, str]]:
        template_name_to_view_name: Dict[str, str] = {}

        def replace_placeholder(match: re.Match) -> str:
            placeholder = match.group(1)
            if placeholder not in template_name_to_view_name:
                view_name = generate_unique_view_name()
                template_name_to_view_name[placeholder] = view_name
            return template_name_to_view_name[placeholder]

        replaced_sql = SQL_PLACEHOLDER_RE.sub(replace_placeholder, self._templated_query)
        view_names = [template_name_to_view_name[name] for name in self._template_names]
        return replaced_sql, view_names

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        self._validate_query()
        db_conn = duckdb.connect()
        for view_name, input in zip(self.view_names, self._inputs, strict=True):
            polars_schema = convert_custom_schema_to_polars_schema(input.schema())
            db_conn.register(view_name, pl.DataFrame(schema=polars_schema))
        try:
            pl_result = db_conn.execute(f"SET timezone = 'UTC'; {self.resolved_query}").pl()
            return convert_polars_schema_to_custom_schema(pl_result.schema)
        except Exception as e:
            raise PlanError(f"Failed to plan SQL query: {self._templated_query}") from e


    def _validate_query(self) -> None:
        try:
            statements = sqlglot.parse(self.resolved_query, read="duckdb")
        except sqlglot.ParseError as e:
            raise PlanError(
                f"SQL parsing failed. "
                f"Check your SQL syntax for missing commas, unmatched parentheses, "
                f"incorrect keywords, or invalid table/column names. "
                f"Query: {self._templated_query}"
            ) from e

        if not statements:
            raise PlanError(
                f"Failed to parse SQL query: `{self._templated_query}`. Make sure the query is syntactically valid and non-empty."
            )

        if len(statements) != 1:
            raise PlanError(
                f"Expected a single SQL statement in session.sql(), but found {len(statements)}. "
                f"Multiple statements are not supported. Please provide only one statement at a time."
            )

        root_expr = statements[0]
        _validate_select_only_tree(root_expr)

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        """Create and return a new instance of the SQL plan with the given children."""
        if len(children) == 0:
            raise InternalError("SQL node must have at least one child")
        # The list of input session states is empty because the session state has already been validated.
        result = SQL.from_session_state(children, self._template_names, self._templated_query, session_state)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: SQL) -> bool:
        return (
            self._template_names == other._template_names
            and self._templated_query == other._templated_query
        )

@dataclass
class CentroidInfo:
    centroid_column: str
    num_dimensions: int

class SemanticCluster(LogicalPlan):
    def __init__(
            self,
            input: LogicalPlan,
            by_expr: LogicalExpr,
            num_clusters: int,
            max_iter: int,
            num_init: int,
            label_column: str,
            centroid_column: Optional[str],
            request_timeout: Optional[float] = None,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        validate_scalar_expr(by_expr, "semantic.with_cluster_labels")
        self._input = input
        self._by_expr = by_expr
        self._num_clusters = num_clusters
        self._max_iter = max_iter
        self._num_init = num_init
        self._label_column = label_column
        self._centroid_column = centroid_column
        self._centroid_info: Optional[CentroidInfo] = None
        self.request_timeout = request_timeout
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(
        cls,
        input: LogicalPlan,
        by_expr: LogicalExpr,
        num_clusters: int,
        max_iter: int,
        num_init: int,
        label_column: str,
        centroid_column: Optional[str],
        request_timeout: Optional[float] = None,
        schema: Schema = None,
    ) -> SemanticCluster:
        return SemanticCluster(
            input,
            by_expr,
            num_clusters,
            max_iter,
            num_init,
            label_column,
            centroid_column,
            request_timeout,
            schema=schema,
        )

    @classmethod
    def from_session_state(
        cls,
        input: LogicalPlan,
        by_expr: LogicalExpr,
        num_clusters: int,
        max_iter: int,
        num_init: int,
        label_column: str,
        centroid_column: Optional[str],
        session_state: BaseSessionState,
        request_timeout: Optional[float] = None,
    ) -> SemanticCluster:
        return SemanticCluster(
            input,
            by_expr,
            num_clusters,
            max_iter,
            num_init,
            label_column,
            centroid_column,
            request_timeout,
            session_state=session_state,
        )

    def children(self) -> List[LogicalPlan]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        by_expr_type = self._by_expr.to_column_field(self._input, session_state).data_type
        if not isinstance(by_expr_type, EmbeddingType):
            raise TypeMismatchError.from_message(
                f"semantic.with_cluster_labels by expression must be an embedding column type (EmbeddingType); "
                f"got: {by_expr_type}"
            )

        new_fields = [ColumnField(self._label_column, IntegerType)]
        if self._centroid_column:
            new_fields.append(ColumnField(self._centroid_column, by_expr_type))
            self._centroid_info = CentroidInfo(self._centroid_column, by_expr_type.dimensions)

        return Schema(column_fields=self._input.schema().column_fields + new_fields)

    def exprs(self) -> List[LogicalExpr]:
        return [self._by_expr]

    def _repr(self) -> str:
        return f"SemanticCluster(by_expr={str(self._by_expr)}, num_clusters={self._num_clusters})"

    def num_clusters(self) -> int:
        return self._num_clusters

    def max_iter(self) -> int:
        return self._max_iter

    def num_init(self) -> int:
        return self._num_init

    def centroid_info(self) -> Optional[CentroidInfo]:
        return self._centroid_info

    def by_expr(self) -> LogicalExpr:
        return self._by_expr

    def label_column(self) -> str:
        return self._label_column

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 1:
            raise ValueError("SemanticCluster must have exactly one child")
        result = SemanticCluster.from_session_state(
            children[0],
            self._by_expr,
            num_clusters=self._num_clusters,
            max_iter=self._max_iter,
            num_init=self._num_init,
            label_column=self._label_column,
            centroid_column=self._centroid_column,
            session_state=session_state,
            request_timeout=self.request_timeout,
        )
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: SemanticCluster) -> bool:
        return (
            self._by_expr == other._by_expr
            and self._num_clusters == other._num_clusters
            and self._max_iter == other._max_iter
            and self._num_init == other._num_init
            and self._label_column == other._label_column
            and self._centroid_column == other._centroid_column
            and self.request_timeout == other.request_timeout
        )

DDL_DML_NODES = (
    sqlglot_exprs.Insert,
    sqlglot_exprs.Delete,
    sqlglot_exprs.Update,
    sqlglot_exprs.Create,
    sqlglot_exprs.Drop,
    sqlglot_exprs.Alter,
    sqlglot_exprs.TruncateTable,
    sqlglot_exprs.Use,
    sqlglot_exprs.Grant,
    sqlglot_exprs.Fetch,
    sqlglot_exprs.Set,
)

def _validate_select_only_tree(expr: sqlglot.Expression):
    for node in expr.walk():
        if isinstance(node, DDL_DML_NODES):
            raise PlanError(
                f"Only read-only queries are supported in session.sql(). Found unsupported statement type: {node.__class__.__name__}. "
                "Please remove any DML or DDL statements (e.g., INSERT, UPDATE, CREATE) from the query."
            )
