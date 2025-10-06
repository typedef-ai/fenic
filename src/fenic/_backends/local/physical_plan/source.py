from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Optional, Tuple

import polars as pl

import fenic._backends.local.polars_plugins  # noqa: F401
from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.schema_serde import serialize_data_type
from fenic.core.error import InternalError
from fenic.core.types.datatypes import JsonType, MarkdownType, StringType
from fenic.core.types.enums import DocContentType

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
        return apply_ingestion_coercions(self.df, coerce_array=True)

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
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: SourceExec expects 0 children")

        file_format = self.file_format.lower()
        df = query_files(paths=self.paths, file_type=file_format, s3_session=self.session_state.s3_session, **self.options)
        return apply_ingestion_coercions(df, coerce_array=True)

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
            content_type: DocContentType,
            exclude: Optional[str],
            recursive: bool,
            session_state: LocalSessionState,
    ):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.paths = paths
        self.content_type = content_type
        self.exclude = exclude
        self.recursive = recursive

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: DocSourceExec expects 0 children")
        df = DocFolderLoader.load_docs_from_folder(
            self.paths,
            self.content_type,
            self.exclude,
            self.recursive)
        df = apply_ingestion_coercions(df, coerce_array=True)
        if self.content_type in ["markdown", "json"]:
            # overwrite the content column with the casted content
            source_type = json.dumps(serialize_data_type(StringType))
            if self.content_type == "markdown":
                dest_type = json.dumps(serialize_data_type(MarkdownType))
            else:
                dest_type = json.dumps(serialize_data_type(JsonType))
            df = df.with_columns(
                pl.col("content").dtypes.cast(source_type, dest_type).alias("content")
            )
        return df

    

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: DocSourceExec expects 0 children")
        return DocSourceExec(
            paths=self.paths,
            content_type=self.content_type,
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

class CacheReadExec(PhysicalPlan):
    def __init__(self, cache_key: str, session_state: LocalSessionState):
        super().__init__(children=[], cache_info=None, session_state=session_state)
        self.cache_key = cache_key

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 0:
            raise InternalError("Unreachable: CacheReadExec expects 0 children")
        return self.session_state.intermediate_df_client.read_df(self.cache_key)

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 0:
            raise InternalError("Unreachable: CacheReadExec expects 0 children")
        return CacheReadExec(
            cache_key=self.cache_key,
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
