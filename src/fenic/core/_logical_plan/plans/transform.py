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
    AggregateExpr,
    ColumnExpr,
    LogicalExpr,
    SortExpr,
)
from fenic.core._logical_plan.plans.base import ensure_same_session
from fenic.core._logical_plan.plans.node import LogicalPlanNode
from fenic.core._utils.misc import generate_unique_arrow_view_name
from fenic.core._utils.schema import (
    convert_custom_schema_to_polars_schema,
    convert_polars_schema_to_custom_schema,
)
from fenic.core.error import (
    InternalError,
    PlanError,
    TypeMismatchError,
    ValidationError,
)
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

class Projection(LogicalPlanNode):
    def __init__(self, exprs: List[LogicalExpr]):
        super().__init__()
        self._exprs = exprs

    def children(self) -> List[LogicalPlanNode]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        if self._input is None:
            raise ValidationError("Projection must have an input")

        fields = []
        for expr in self._exprs:
            if isinstance(expr, AggregateExpr):
                raise ValueError(
                    "Aggregate expressions are not allowed in projections. "
                    "Please use the agg() method instead."
                )

            fields.append(expr.to_column_field(self._input, session_state))
        return Schema(fields)

    def exprs(self) -> List[LogicalExpr]:
        return self._exprs

    def _repr(self) -> str:
        exprs_str = ", ".join(str(expr) for expr in self._exprs)
        return f"Projection(exprs=[{exprs_str}])"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("Projection must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(
            cls,
            node: LogicalPlanNode,
            children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = Projection(node._exprs)
        new_node.set_input(children[0])
        return new_node

class Filter(LogicalPlanNode):
    def __init__(self, predicate: LogicalExpr):
        super().__init__()
        self._predicate = predicate

    def set_input(self, input: LogicalPlanNode):
        self._input = input

    def children(self) -> List[LogicalPlanNode]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        actual_type = self._predicate.to_column_field(self._input, session_state).data_type
        if actual_type != BooleanType:
            raise ValueError(
                f"Filter predicate must return a boolean value, but got {actual_type}. "
                "Examples of valid filters:\n"
                "- df.filter(col('age') > 18)\n"
                "- df.filter(col('status') == 'active')\n"
                "- df.filter(col('is_valid'))"
            )
        if isinstance(self._predicate, AggregateExpr):
            raise ValueError(
                "Aggregate expressions are not allowed in projections. "
                "Please use the agg() method instead."
            )
        if isinstance(self._predicate, SortExpr):
            raise ValueError(
                "Sort expressions are not allowed in projections. "
                "Please use the sort() method instead."
            )
        return self._input.schema()

    def predicate(self) -> LogicalExpr:
        return self._predicate

    def _repr(self) -> str:
        """Return the representation for the Filter node."""
        return f"Filter(predicate={self._predicate})"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("Filter must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(
            cls,
            node: LogicalPlanNode,
            children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = Filter(node._predicate)
        new_node.set_input(children[0])
        return new_node

class Union(LogicalPlanNode):
    def __init__(self, inputs: List[LogicalPlanNode]):
        super().__init__()
        self._inputs = inputs

    def set_input(self, input: LogicalPlanNode):
        # Skip setting the input as it is a list of LogicalPlanNodes
        pass

    def children(self) -> List[LogicalPlanNode]:
        return self._inputs

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        schemas = [input_plan.schema() for input_plan in self._inputs]

        # Check that all schemas have the same columns and types
        first_schema = schemas[0]
        first_schema_fields = {f.name: f.data_type for f in first_schema.column_fields}

        for schema in schemas[1:]:
            schema_fields = {f.name: f.data_type for f in schema.column_fields}
            if set(schema_fields.keys()) != set(first_schema_fields.keys()):
                raise ValueError(
                    "Cannot union DataFrames with different columns. "
                    "All DataFrames must have exactly the same column names."
                )
            for name, type_ in schema_fields.items():
                if type_ != first_schema_fields[name]:
                    raise ValueError(
                        f"Cannot union DataFrames: column '{name}' must be type {first_schema_fields[name]} "
                        "across all DataFrames. Consider casting columns to matching types before union."
                    )

        return schemas[0]

    def _repr(self) -> str:
        return "Union"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return self.copy(self, children)

    @classmethod
    def _create_new_node(
            cls,
            node: LogicalPlanNode,
            children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return Union(children)

    @classmethod
    def validate_same_session(
            cls,
            current_session_state: BaseSessionState,
            input_session_states: list[BaseSessionState]) -> None:
        for input_session_state in input_session_states:
            ensure_same_session(input_session_state, current_session_state)


class Limit(LogicalPlanNode):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def children(self) -> List[LogicalPlanNode]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        return self._input.schema()

    def _repr(self) -> str:
        return f"Limit(n={self.n})"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("Limit must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(
            cls,
            node: LogicalPlanNode,
            children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = Limit(node.n)
        new_node.set_input(children[0])
        return new_node

class Explode(LogicalPlanNode):
    def __init__(self, expr: LogicalExpr):
        super().__init__()
        self._expr = expr

    def children(self) -> list[LogicalPlanNode]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        input_schema = self._input.schema()
        exploded_field = self._expr.to_column_field(self._input)

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

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("Explode must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = Explode(node._expr)
        new_node.set_input(children[0])
        return new_node

class DropDuplicates(LogicalPlanNode):
    def __init__(
        self,
        subset: List[ColumnExpr],
    ):
        super().__init__()
        self.subset = subset

    def children(self) -> List[LogicalPlanNode]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        return self._input.schema()

    def _repr(self) -> str:
        return f"DropDuplicates(subset={', '.join(str(expr) for expr in self.subset)})"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("DropDuplicates must have exactly one child")
        return self.copy(self, children)

    def _subset(self) -> List[str]:
        subset: List[str] = []
        for col in self.subset:
            subset.append(str(col))
        return subset

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = DropDuplicates(node.subset)
        new_node.set_input(children[0])
        return new_node

class Sort(LogicalPlanNode):
    def __init__(
        self,
        sort_exprs: List[SortExpr],
    ):
        super().__init__()
        self._sort_exprs = sort_exprs

    def children(self) -> List[LogicalPlanNode]:
        return [self._input]

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        return self._input.schema()

    def sort_exprs(self) -> List[LogicalExpr]:
        return self._sort_exprs

    def _repr(self) -> str:
        return f"Sort(cols={', '.join(str(expr) for expr in self._sort_exprs)})"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("Sort must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = Sort(node._sort_exprs)
        new_node.set_input(children[0])
        return new_node


class Unnest(LogicalPlanNode):
    def __init__(self, exprs: List[ColumnExpr]):
        super().__init__()
        self._exprs = exprs

    def children(self) -> List[LogicalPlanNode]:
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

    def _repr(self) -> str:
        return f"Unnest(exprs={', '.join(str(expr) for expr in self._exprs)})"

    def col_names(self) -> List[str]:
        return [expr.name for expr in self._exprs]

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("Unnest must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = Unnest(node._exprs)
        new_node.set_input(children[0])
        return new_node

class SQL(LogicalPlanNode):
    def __init__(
            self,
            inputs: List[LogicalPlanNode],
            template_names: List[str],
            templated_query: str,
        ):
        super().__init__()
        # Note: inputs[i] corresponds to template_names[i]
        if len(inputs) != len(template_names):
            raise InternalError("inputs and template_names must have the same length")
        self._inputs = inputs
        self._template_names = template_names
        self._templated_query = templated_query
        self.resolved_query, self.view_names = self._replace_query_placeholders()

    def set_input(self, input: LogicalPlanNode):
        # Skip setting the input.
        pass

    def children(self) -> List[LogicalPlanNode]:
        return self._inputs

    def _repr(self) -> str:
        return f"SQL(query={self._templated_query})"

    def _replace_query_placeholders(self) -> Tuple[str, Dict[str, str]]:
        template_name_to_view_name: Dict[str, str] = {}

        def replace_placeholder(match: re.Match) -> str:
            placeholder = match.group(1)
            if placeholder not in template_name_to_view_name:
                view_name = generate_unique_arrow_view_name()
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
            arrow_result = db_conn.execute(self.resolved_query).arrow()
            return convert_polars_schema_to_custom_schema(pl.from_arrow(arrow_result).schema)
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

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        """Create and return a new instance of the SQL plan with the given children."""
        if len(children) == 0:
            raise InternalError("SQL node must have at least one child")
        return self.copy(self, children)
    
    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return SQL(children, node._template_names, node._templated_query)

    @classmethod
    def validate_same_session(
            cls,
            current_session_state: BaseSessionState,
            input_session_states: list[BaseSessionState]) -> None:
        for input_session_state in input_session_states:
            ensure_same_session(input_session_state, current_session_state)

@dataclass
class CentroidInfo:
    centroid_column: str
    num_dimensions: int

class SemanticCluster(LogicalPlanNode):
    def __init__(
        self,
        by_expr: LogicalExpr,
        num_clusters: int,
        max_iter: int,
        num_init: int,
        label_column: str,
        centroid_column: Optional[str],
        centroid_info: Optional[CentroidInfo] = None,
    ):
        super().__init__()
        self._by_expr = by_expr
        self._num_clusters = num_clusters
        self._max_iter = max_iter
        self._num_init = num_init
        self._label_column = label_column
        self._centroid_column = centroid_column
        self._centroid_info = centroid_info

    def children(self) -> List[LogicalPlanNode]:
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

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 1:
            raise ValueError("SemanticCluster must have exactly one child")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        new_node = SemanticCluster(
            node._by_expr,
            node._num_clusters,
            node._max_iter,
            node._num_init,
            node._label_column,
            node._centroid_column,
            node._centroid_info,
        )
        new_node.set_input(children[0])
        return new_node


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
