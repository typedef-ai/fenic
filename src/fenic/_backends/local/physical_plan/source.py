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
    DuckDBNodeMixin,
)
from fenic._backends.local.physical_plan.utils import apply_ingestion_coercions
from fenic._backends.local.utils.doc_loader import DocFolderLoader
from fenic._backends.local.utils.io_utils import build_read_sql_query


class InMemorySourceExec(PhysicalPlan):
    def __init__(self, df: pl.DataFrame, session_state: LocalSessionState):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.df = df

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: InMemorySourceExec expects 0 children")
        return apply_ingestion_coercions(self.df)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        materialize_df = _with_lineage_uuid(self.df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: InMemorySourceExec expects 0 children")
        return InMemorySourceExec(self.df, self.session_state)


class FileSourceExec(PhysicalPlan, DuckDBNodeMixin):
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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
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
        df = build_read_sql_query(query=query, paths=self.paths, s3_session=self.session_state.s3_session)
        return apply_ingestion_coercions(df)

    def get_sql_query(self, view_names: List[str]) -> str:
        pass

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        df = self._execute([])
        materialize_df = _with_lineage_uuid(df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: FileSourceExec expects 0 children")
        return FileSourceExec(self.paths, self.file_format, self.session_state, self.options)


class DuckDBTableSourceExec(PhysicalPlan, DuckDBNodeMixin):
    def __init__(self, table_name: str, session_state: LocalSessionState):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.table_name = table_name

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: TableSourceExec expects 0 children")
        return self.session_state.catalog.read_df_from_table(self.table_name)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        df = self._execute([])
        materialize_df = _with_lineage_uuid(df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df

    def get_sql_query(self, _view_names: List[str]) -> str:
        table_name = self.session_state.catalog.get_fully_qualified_table_name(self.table_name)
        return f"SELECT * FROM {table_name}"

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: DuckDBTableSourceExec expects 0 children")
        return DuckDBTableSourceExec(self.table_name, self.session_state)

class CacheReadExec(PhysicalPlan, DuckDBNodeMixin):
    def __init__(self, cache_key: str, session_state: LocalSessionState):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.cache_key = cache_key

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: CacheReadExec expects 0 children")
        df = self.session_state.db_client.intermediate.read_df(self.cache_key)
        return apply_ingestion_coercions(df)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        df = self._execute([])
        materialize_df = _with_lineage_uuid(df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df

    def get_sql_query(self, _view_names: List[str]) -> str:
        return self.session_state.db_client.intermediate.get_read_df_query(self.cache_key)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: CacheReadExec expects 0 children")
        return CacheReadExec(self.cache_key, self.session_state)

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

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: DocSourceExec expects 0 children")
        df = DocFolderLoader.load_docs_from_folder(
            self.paths,
            self.valid_file_extension,
            self.exclude,
            self.recursive)
        return apply_ingestion_coercions(df)

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        df = self._execute([])
        materialize_df = _with_lineage_uuid(df)
        source_operator = self._build_source_operator_lineage(materialize_df)
        leaf_nodes.append(source_operator)
        return source_operator, materialize_df

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: DocSourceExec expects 0 children")
        return DocSourceExec(self.paths, self.valid_file_extension, self.exclude, self.recursive, self.session_state)
