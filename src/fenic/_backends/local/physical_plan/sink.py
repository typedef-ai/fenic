from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Literal, Tuple

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState

import polars as pl

from fenic._backends.local.lineage import OperatorLineage
from fenic._backends.local.physical_plan import PhysicalPlan
from fenic._backends.local.utils.io_utils import does_path_exist, write_file
from fenic.core._logical_plan.plans import CacheInfo
from fenic.core.error import InternalError, PlanError
from fenic.core.types import Schema

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

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
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
        write_file(df=df, path=self.path, s3_session=self.session_state.s3_session, file_type=self.file_type)
        return pl.DataFrame()


    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: FileSinkExec expects 1 child")
        return FileSinkExec(
            child=children[0],
            path=self.path,
            file_type=self.file_type,
            mode=self.mode,
            cache_info=self.cache_info,
            session_state=self.session_state,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        """Build the lineage graph for this sink operation.

        Returns:
                A LineageGraph representing the operation
        """
        raise InternalError("FileSink does not support lineage")


class DuckDBTableSinkExec(PhysicalPlan):
    """Physical plan node for DuckDB table sink operations."""

    def __init__(
        self,
        child: PhysicalPlan,
        table_name: str,
        mode: Literal["error", "overwrite", "ignore"],
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

    def execute_node(self, child_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        if len(child_dfs) != 1:
            raise InternalError("TableSink expects exactly one child DataFrame")
        df = child_dfs[0]
        table_exists = self.session_state.catalog.does_table_exist(self.table_name)
        if table_exists:
            if self.mode == "error":
                raise PlanError(
                    f"Cannot save to table '{self.table_name}' - it already exists and mode is 'error'. "
                    f"Choose a different approach: "
                    f"1) Use mode='overwrite' to replace the existing table, "
                    f"2) Use mode='append' to add data to the existing table, "
                    f"3) Use mode='ignore' to skip saving if table exists, "
                    f"4) Use a different table name."
                )
            if self.mode == "ignore":
                logger.warning(
                    f"Table {self.table_name} already exists, ignoring write."
                )
                return pl.DataFrame()
            if self.mode == "append":
                self.session_state.catalog.insert_df_to_table(
                    df, self.table_name, self.schema
                )
            elif self.mode == "overwrite":
                self.session_state.catalog.replace_table_with_df(
                    df, self.table_name, self.schema
                )
        else:
            self.session_state.catalog.write_df_to_table(
                df, self.table_name, self.schema
            )

        return pl.DataFrame()

    def with_children(self, children: List[PhysicalPlan]) -> PhysicalPlan:
        if len(children) != 1:
            raise InternalError("Unreachable: DuckDBTableSinkExec expects 1 child")
        return DuckDBTableSinkExec(
            child=children[0],
            table_name=self.table_name,
            mode=self.mode,
            cache_info=self.cache_info,
            session_state=self.session_state,
            schema=self.schema,
        )

    def build_node_lineage(
        self,
        leaf_nodes: List[OperatorLineage],
    ) -> Tuple[OperatorLineage, pl.DataFrame]:
        """Build the lineage graph for this sink operation.

        Returns:
            A LineageGraph representing the operation
        """
        raise InternalError("TableSink does not support lineage")
