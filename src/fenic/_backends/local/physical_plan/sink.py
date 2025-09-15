from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Literal, Tuple

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.local.physical_plan import PhysicalPlan, DuckDBNodeMixin
from fenic._backends.local.utils.io_utils import does_path_exist, build_write_sql_query
from fenic.core._logical_plan.plans import CacheInfo
from fenic.core.error import InternalError, PlanError
from fenic.core.types import Schema
from fenic.core._utils.misc import generate_unique_arrow_view_name

logger = logging.getLogger(__name__)


class FileSinkExec(PhysicalPlan):
    """Physical plan node for file sink operations."""

    def __init__(
        self,
        child: PhysicalPlan,
        path: str,
        file_type: str,
        mode: Literal["error", "overwrite", "ignore"],
        cache_info: CacheInfo,
        session_state: LocalSessionState,
    ):
        super().__init__(
            children=[child], cache_info=cache_info, session_state=session_state
        )
        self.path = path
        self.file_type = file_type.lower()
        self.mode = mode

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise InternalError("FileSink expects exactly one child DataFrame")

        file_exists = does_path_exist(self.path, self.session_state.s3_session)
        if self.mode == "error" and file_exists:
            raise PlanError(
                f"Cannot save to file '{self.path}' - it already exists and mode is 'error'. "
                f"Choose a different approach: "
                f"1) Use mode='overwrite' to replace the existing file, "
                f"2) Use mode='ignore' to skip saving if file exists, "
                f"3) Use a different file path."
            )
        if self.mode == "ignore" and file_exists:
            logger.warning(f"File {self.path} already exists, ignoring write.")
            return pl.DataFrame()
        df = child_dfs[0]
        query = build_write_sql_query(df=df, path=self.path, s3_session=self.session_state.s3_session, file_type=self.file_type)
        self.session_state.db_client.execute(query)
        return pl.DataFrame()

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        """Build the lineage graph for this sink operation.

        Returns:
                A LineageGraph representing the operation
        """
        raise InternalError("FileSink does not support lineage")


class DuckDBTableSinkExec(PhysicalPlan, DuckDBNodeMixin):
    """Physical plan node for DuckDB table sink operations."""

    def __init__(
        self,
        child: PhysicalPlan,
        table_name: str,
        mode: Literal["append", "overwrite"],
        cache_info: CacheInfo,
        session_state: LocalSessionState,
        schema: Schema,
    ):
        super().__init__(
            children=[child], cache_info=cache_info, session_state=session_state
        )
        self.table_name = table_name
        self.mode = mode
        self.schema = schema

    def _execute(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise InternalError("TableSink expects exactly one child DataFrame")
        view_name = generate_unique_arrow_view_name()
        self.session_state.db_client.register(view_name, child_dfs[0])
        query = self.get_sql_query([view_name])
        self.session_state.db_client.execute(query)
        return pl.DataFrame()

    def get_sql_query(self, view_names: List[str]) -> str:
        if len(view_names) != 1:
            raise InternalError("Unreachable: DuckDBTableSinkExec expects exactly one view name")
        view_name = view_names[0]
        table_name = self.session_state.catalog.get_fully_qualified_table_name(self.table_name)
        if self.mode == "append":
            if self.session_state.catalog.does_table_exist(self.table_name):
                return f"INSERT INTO {table_name} SELECT * FROM {view_name}"
            else:
                return f"CREATE TABLE {table_name} AS SELECT * FROM {view_name}"
        elif self.mode == "overwrite":
            return f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}"
        else:
            raise InternalError(f"Unreachable: DuckDBTableSinkExec mode {self.mode} not supported")

    def _build_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        """Build the lineage graph for this sink operation.

        Returns:
            A LineageGraph representing the operation
        """
        raise InternalError("TableSink does not support lineage")

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: TableSink expects 1 child")
        return DuckDBTableSinkExec(
            child=children[0],
            table_name=self.table_name,
            mode=self.mode,
            cache_info=self.cache_info,
            session_state=self.session_state,
            schema=self.schema
        )
