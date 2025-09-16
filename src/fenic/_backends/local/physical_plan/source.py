from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic.core.error import InternalError

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

from fenic._backends.local.physical_plan.base import (
    PhysicalPlan,
    _with_lineage_uuid,
)
from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic._backends.local.utils.doc_loader import DocFolderLoader
from fenic._backends.local.utils.io_utils import query_files


class InMemorySourceExec(PhysicalPlan):
    def __init__(self, df: pl.DataFrame, session_state: LocalSessionState):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.df = df

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: InMemorySourceExec expects 0 children")
        return apply_ingestion_coercions(self.df)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: InMemorySourceExec expects 0 children")
        return InMemorySourceExec(self.df, self.session_state)

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        materialize_df = _with_lineage_uuid(self.df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df


class FileSourceExec(PhysicalPlan):
    def __init__(
        self,
        paths: list[str],
        file_format: str,
        session_state: LocalSessionState,
        options: dict = None,
    ):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.path_string = "', '".join(paths)
        self.paths = paths
        self.file_format = file_format
        self.options = options or {}

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if child_dfs:
            raise InternalError("Unreachable: SourceExec expects 0 children")

        file_format = self.file_format.lower()
        build_query_fn = {
            "csv": self.session_state.execution._build_read_csv_query,
            "parquet": self.session_state.execution._build_read_parquet_query,
        }.get(file_format)

        if build_query_fn is None:
            raise InternalError(f"Unsupported file format: {self.file_format}")
        query = build_query_fn(self.paths, False, **self.options)
        df = query_files(query=query, paths=self.paths, s3_session=self.session_state.s3_session)
        return apply_ingestion_coercions(df)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: FileSourceExec expects 0 children")
        return FileSourceExec(
            paths=self.paths,
            file_format=self.file_format,
            session_state=self.session_state,
            options=self.options,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        df = self.execute_node([])
        materialize_df = _with_lineage_uuid(df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df


class DuckDBTableSourceExec(PhysicalPlan):
    def __init__(self, table_name: str, session_state: LocalSessionState):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.table_name = table_name

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: TableSourceExec expects 0 children")
        return self.session_state.catalog.read_df_from_table(self.table_name)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: TableSourceExec expects 0 children")
        return DuckDBTableSourceExec(
            table_name=self.table_name,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        df = self.execute_node([])
        materialize_df = _with_lineage_uuid(df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df


class DocSourceExec(PhysicalPlan):
    def __init__(
            self,
            paths: list[str],
            valid_file_extension: str,
            exclude: Optional[str],
            recursive: bool,
            session_state: LocalSessionState,
    ):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.paths = paths
        self.valid_file_extension = valid_file_extension
        self.exclude = exclude
        self.recursive = recursive

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: DocSourceExec expects 0 children")
        df = DocFolderLoader.load_docs_from_folder(
            self.paths,
            self.valid_file_extension,
            self.exclude,
            self.recursive)
        return apply_ingestion_coercions(df)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: DocSourceExec expects 0 children")
        return DocSourceExec(
            paths=self.paths,
            valid_file_extension=self.valid_file_extension,
            exclude=self.exclude,
            recursive=self.recursive,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        df = self.execute_node([])
        materialize_df = _with_lineage_uuid(df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df
