from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional

import polars as pl

from fenic._backends.local.utils.doc_loader import DocFolderLoader
from fenic._constants import PRETTY_PRINT_INDENT
from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.plans.base import LogicalPlan
from fenic.core._utils.schema import (
    convert_polars_schema_to_custom_schema,
)
from fenic.core.error import InternalError, PlanError
from fenic.core.types import Schema
from fenic.core.types.enums import DocContentType

if TYPE_CHECKING:
    from fenic.core._logical_plan.expressions.base import LogicalExpr

class InMemorySource(LogicalPlan):
    def __init__(
            self,
            source: pl.DataFrame,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        self._source = source
        super().__init__(session_state, schema)

    @classmethod
    def from_schema(cls, source: pl.DataFrame, schema: Schema) -> InMemorySource:
        return InMemorySource(source=source, schema=schema)

    @classmethod
    def from_session_state(cls, source: pl.DataFrame, session_state: BaseSessionState) -> InMemorySource:
        return InMemorySource(source=source, session_state=session_state)

    def children(self) -> List[LogicalPlan]:
        return []

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        return convert_polars_schema_to_custom_schema(self._source.schema)

    def _repr(self) -> str:
        return f"InMemorySource({self.schema()})"

    def _repr_with_indent(self, level: int) -> str:
        indent = PRETTY_PRINT_INDENT * level
        inner = self.schema()._str_with_indent(base_indent=level + 1)
        return f"InMemorySource(\n{inner}\n{indent})"

    def with_children(self,
        children: List[LogicalPlan],
        session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 0:
            raise InternalError(
                f"InMemorySource must have no children, got {len(children)}"
            )
        return self.from_schema(self._source, self._schema)

    def _eq_specific(self, other: InMemorySource) -> bool:
        return self._source.equals(other._source)

class FileSource(LogicalPlan):
    def __init__(
            self,
            paths: list[str],
            file_format: Literal["csv", "parquet"],
            options: Optional[dict] = None,
            session_state: Optional[BaseSessionState] = None,
            schema: Optional[Schema] = None):
        """A lazy FileSource that stores the file path, file format, options, and immediately infers the schema using the given session state."""
        self._paths = paths
        self._file_format = file_format
        self._options = options or {}
        super().__init__(session_state, schema)

    @classmethod
    def from_session_state(cls, paths: list[str], file_format: Literal["csv", "parquet"], options: Optional[dict] = None, session_state: Optional[BaseSessionState] = None) -> FileSource:
        return FileSource(paths, file_format, options, session_state, None)

    @classmethod
    def from_schema(cls, paths: list[str], file_format: Literal["csv", "parquet"], options: Optional[dict] = None, schema: Optional[Schema] = None) -> FileSource:
        return FileSource(paths, file_format, options, None, schema)

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        """Uses DuckDB (via the session state) to perform a minimal read (LIMIT 0) and obtain the schema from the file."""
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
                        f"Failed to infer schema from CSV files: {e}"
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
                        f"Failed to infer schema from Parquet files: {e}"
                    ) from e

    def children(self) -> List[LogicalPlan]:
        return []

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _repr(self) -> str:
        return f"FileSource(paths={self._paths}, format={self._file_format})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 0:
            raise InternalError(f"FileSource must have no children, got {len(children)}")
        result = FileSource.from_schema(
            self._paths, self._file_format, self._options, self._schema
        )
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: FileSource) -> bool:
        return (
            self._paths == other._paths
            and self._file_format == other._file_format
            and self._options == other._options
        )

class TableSource(LogicalPlan):
    def __init__(
            self,
            table_name: str,
            session_state: Optional[BaseSessionState],
            schema: Optional[Schema]):
        self._table_name = table_name
        super().__init__(session_state, schema)

    @classmethod
    def from_session_state(cls, table_name: str, session_state: BaseSessionState) -> TableSource:
        return TableSource(table_name, session_state, None)

    @classmethod
    def from_schema(cls, table_name: str, schema: Schema) -> TableSource:
        return TableSource(table_name, None, schema)

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        if not session_state.catalog.does_table_exist(self._table_name):
            raise PlanError(
                f"Table '{self._table_name}' does not exist. "
                f"Use session.catalog.list_tables() to see available tables, "
                f"or load data using session.csv() or session.parquet()."
            )
        return session_state.catalog.describe_table(self._table_name).schema

    def children(self) -> List[LogicalPlan]:
        return []

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _repr(self) -> str:
        return f"TableSource(table_name={self._table_name})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 0:
            raise InternalError(f"TableSource must have no children, got {len(children)}")
        result = TableSource.from_schema(self._table_name, self._schema)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: TableSource) -> bool:
        return self._table_name == other._table_name

class DocSource(LogicalPlan):
    def __init__(
        self,
        paths: list[str],
        content_type: DocContentType,
        exclude: Optional[str] = None,
        recursive: bool = False,
        session_state: Optional[BaseSessionState] = None,
        schema: Optional[Schema] = None):
        self._paths = paths
        self._content_type = content_type
        self._exclude = exclude
        self._recursive = recursive
        super().__init__(session_state, schema)

    @classmethod
    def from_session_state(
        cls,
        paths: list[str],
        content_type: DocContentType,
        exclude: Optional[str] = None,
        recursive: bool = False,
        session_state: Optional[BaseSessionState] = None,
    ) -> DocSource:
        return DocSource(paths, content_type, exclude, recursive, session_state, None)

    @classmethod
    def from_schema(
        cls,
        paths: list[str],
        content_type: DocContentType,
        exclude: Optional[str] = None,
        recursive: bool = False,
        schema: Optional[Schema] = None,
    ) -> DocSource:
        return DocSource(paths, content_type, exclude, recursive, None, schema)

    def _build_schema(self, session_state: BaseSessionState) -> Schema:
        DocFolderLoader.validate_paths(self._paths)
        return DocFolderLoader.get_schema(self._content_type)

    def children(self) -> List[LogicalPlan]:
        return []

    def exprs(self) -> List[LogicalExpr]:
        return []

    def _repr(self) -> str:
        return f"DocSource(path={self._paths}, content_type={self._content_type}, exclude={self._exclude}, recursive={self._recursive})"

    def with_children(self, children: List[LogicalPlan], session_state: Optional[BaseSessionState] = None) -> LogicalPlan:
        if len(children) != 0:
            raise InternalError(f"DocSource must have no children, got {len(children)}")
        result = DocSource.from_schema(
            self._paths, self._content_type, self._exclude, self._recursive, self._schema)
        result.set_cache_info(self.cache_info)
        return result

    def _eq_specific(self, other: DocSource) -> bool:
        return (self._paths == other._paths and
                self._content_type == other._content_type and
                self._exclude == other._exclude and
                self._recursive == other._recursive)
