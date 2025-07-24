from __future__ import annotations

from typing import List, Literal, Optional

import polars as pl

from fenic._constants import PRETTY_PRINT_INDENT
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.node import LogicalPlanNode
from fenic.core._utils.schema import convert_polars_schema_to_custom_schema
from fenic.core.error import InternalError, PlanError, ValidationError
from fenic.core.types import Schema


class InMemorySource(LogicalPlanNode):
    def __init__(self, source: pl.DataFrame):
        super().__init__()
        self._source = source

    def set_input(self, input: LogicalPlanNode):
        raise ValidationError("InMemorySource cannot have an input")

    def children(self) -> List[LogicalPlanNode]:
        return []

    def _build_schema(self, session_state: BaseSessionState):
        return convert_polars_schema_to_custom_schema(self._source.schema)

    def _repr(self) -> str:
        return f"InMemorySource({self.schema()})"

    def _repr_with_indent(self, level: int) -> str:
        indent = PRETTY_PRINT_INDENT * level
        inner = self.schema()._str_with_indent(base_indent=level + 1)
        return f"InMemorySource(\n{inner}\n{indent})"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 0:
            raise InternalError(
                f"InMemorySource must have no children, got {len(children)}"
            )
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return InMemorySource(node._source)


class FileSource(LogicalPlanNode):
    def __init__(
        self,
        paths: list[str],
        file_format: Literal["csv", "parquet"],
        options: Optional[dict] = None,
    ):
        """A lazy FileSource that stores the file path, file format, options, and immediately infers the schema using the given session state."""
        super().__init__()
        self._paths = paths
        self._file_format = file_format
        self._options = options or {}

    def set_input(self, input: LogicalPlanNode):
        raise ValidationError("FileSource cannot have an input")

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        """Uses DuckDB (via the session state) to perform a minimal read (LIMIT 0) and obtain the schema from the file."""
        if self._schema:
            return self._schema

        if self._file_format == "csv":
            try:
                return session_state.execution.infer_schema_from_csv(
                    self._paths, **self._options
                )
            except Exception as e:
                if self._options.get("schema", None):
                    raise PlanError(
                        "Schema mismatch: The provided schema does not match the structure of the CSV files. "
                        "Please verify that all required columns are present and correctly typed."
                    ) from e
                elif self._options.get("merge_schemas", False):
                    raise PlanError(
                        "Inconsistent CSV schemas: The files appear to have different structures. "
                        "If this is expected, try setting merge_schemas=True to allow automatic merging."
                    ) from e
                else:
                    raise PlanError(
                        "Failed to infer schema from CSV files"
                    ) from e

        elif self._file_format == "parquet":
            try:
                return session_state.execution.infer_schema_from_parquet(
                    self._paths, **self._options
                )
            except Exception as e:
                if self._options.get("merge_schemas", False):
                    raise PlanError(
                        "Inconsistent Parquet schemas: The files appear to have different structures. "
                        "If this is expected, try setting merge_schemas=True to allow automatic merging."
                    ) from e
                else:
                    raise PlanError(
                        "Failed to infer schema from Parquet files"
                    ) from e

    def children(self) -> List[LogicalPlanNode]:
        return []

    def _repr(self) -> str:
        return f"FileSource(paths={self._paths}, format={self._file_format})"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 0:
            raise InternalError(f"FileSource must have no children, got {len(children)}")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return FileSource(
            node._paths, node._file_format, node._options
        )


class TableSource(LogicalPlanNode):
    def __init__(self, table_name: str):
        super().__init__()
        self._table_name = table_name

    def set_input(self, input: LogicalPlanNode):
        raise ValidationError("TableSource cannot have an input")

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        if not session_state.catalog.does_table_exist(self._table_name):
            raise PlanError(
                f"Table '{self._table_name}' does not exist. "
                f"Use session.catalog.list_tables() to see available tables, "
                f"or load data using session.csv() or session.parquet()."
            )
        return session_state.catalog.describe_table(self._table_name)

    def children(self) -> List[LogicalPlanNode]:
        return []

    def _repr(self) -> str:
        return f"TableSource(table_name={self._table_name})"

    def with_children(self, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        if len(children) != 0:
            raise InternalError(f"TableSource must have no children, got {len(children)}")
        return self.copy(self, children)

    @classmethod
    def _create_new_node(cls, node: LogicalPlanNode, children: List[LogicalPlanNode]) -> LogicalPlanNode:
        return TableSource(node._table_name)
