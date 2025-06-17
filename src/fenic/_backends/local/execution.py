from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

import polars as pl

from fenic._backends.local.lineage import LocalLineage
from fenic._backends.local.transpiler.transpiler import LocalTranspiler
from fenic._backends.local.utils.io_utils import (
    does_path_exist,
    query_files,
)
from fenic.core._interfaces.execution import BaseExecution
from fenic.core._logical_plan import LogicalPlan
from fenic.core._utils.schema import (
    convert_polars_schema_to_custom_schema,
)
from fenic.core.error import ExecutionError, PlanError, ValidationError
from fenic.core.metrics import QueryMetrics
from fenic.core.types.datatypes import (
    BooleanType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
)
from fenic.core.types.schema import Schema

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fenic._backends.local.session_state import LocalSessionState


class LocalExecution(BaseExecution):
    session_state: LocalSessionState
    transpiler: LocalTranspiler

    def __init__(self, session_state: LocalSessionState):
        self.session_state = session_state
        self.transpiler = LocalTranspiler(session_state)

    def collect(
        self, plan: LogicalPlan, n: Optional[int] = None
    ) -> Tuple[pl.DataFrame, QueryMetrics]:
        """Execute a logical plan and return a Polars DataFrame and query metrics."""
        self.session_state._check_active()
        physical_plan = self.transpiler.transpile(plan)
        try:
            df, metrics = physical_plan.execute()
        except Exception as e:
            raise ExecutionError(f"Failed to execute query: {e}") from e
        if n is not None:
            df = df.limit(n)
        return df, metrics

    def show(self, plan: LogicalPlan, n: int = 10) -> Tuple[str, QueryMetrics]:
        """Execute a logical plan and return a string representation of the sample rows of the DataFrame and query metrics."""
        self.session_state._check_active()
        physical_plan = self.transpiler.transpile(plan)
        try:
            df, metrics = physical_plan.execute()
        except Exception as e:
            raise ExecutionError(f"Failed to execute query: {e}") from e
        with pl.Config(
            fmt_str_lengths=1000,
            set_tbl_hide_dataframe_shape=True,
            set_tbl_hide_column_data_types=True,
            tbl_rows=min(n, df.height),
        ):
            output = str(df)
        return output, metrics

    def count(self, plan: LogicalPlan) -> Tuple[int, QueryMetrics]:
        """Execute a logical plan and return the number of rows in the DataFrame and query metrics."""
        self.session_state._check_active()
        physical_plan = self.transpiler.transpile(plan)
        try:
            df, metrics = physical_plan.execute()
        except Exception as e:
            raise ExecutionError(f"Failed to execute query: {e}") from e
        return df.shape[0], metrics

    def build_lineage(self, plan: LogicalPlan) -> LocalLineage:
        """Build a lineage graph from a logical plan."""
        self.session_state._check_active()
        physical_plan = self.transpiler.transpile(plan)
        try:
            lineage_graph = physical_plan.build_lineage()
        except Exception as e:
            raise ExecutionError(f"Failed to build lineage: {e}") from e
        return LocalLineage(lineage_graph, self.session_state)

    def save_as_table(
        self,
        logical_plan: LogicalPlan,
        table_name: str,
        mode: Literal["error", "append", "overwrite", "ignore"],
    ) -> QueryMetrics:
        """Execute the logical plan and save the result as a table in the current database."""
        self.session_state._check_active()
        table_exists = self.session_state.catalog.does_table_exist(table_name)

        if table_exists:
            if mode == "error":
                raise PlanError(
                    f"Cannot save to table '{table_name}' - it already exists and mode is 'error'. "
                    f"Choose a different approach: "
                    f"1) Use mode='overwrite' to replace the existing table, "
                    f"2) Use mode='append' to add data to the existing table, "
                    f"3) Use mode='ignore' to skip saving if table exists, "
                    f"4) Use a different table name."
                )
            if mode == "ignore":
                logger.warning(f"Table {table_name} already exists, ignoring write.")
                return QueryMetrics()
            if mode == "append":
                saved_schema = self.session_state.catalog.describe_table(table_name)
                plan_schema = logical_plan.schema()
                if saved_schema != plan_schema:
                    raise PlanError(
                        f"Cannot append to table '{table_name}' - schema mismatch detected. "
                        f"The existing table has a different schema than your DataFrame. "
                        f"Existing schema: {saved_schema} "
                        f"Your DataFrame schema: {plan_schema} "
                        f"To fix this: "
                        f"1) Use mode='overwrite' to replace the table with your DataFrame's schema, "
                        f"2) Modify your DataFrame to match the existing table's schema, "
                        f"3) Use a different table name."
                    )
        physical_plan = self.transpiler.transpile(logical_plan)
        try:
            _, metrics = physical_plan.execute()
        except Exception as e:
            raise ExecutionError(f"Failed to execute query: {e}") from e
        return metrics

     # infer schema and save_to_file methods are overridden in the engine execution
     # because the file IO is handled differently in cloud execution.
    def save_to_file(
        self,
        logical_plan: LogicalPlan,
        file_path: str,
        mode: Literal["error", "overwrite", "ignore"] = "error",
    ) -> QueryMetrics:
        """Execute the logical plan and save the result to a file."""
        self.session_state._check_active()

        file_exists = does_path_exist(file_path, self.session_state.s3_session)
        if mode == "error" and file_exists:
            raise PlanError(
                f"Cannot save to file '{file_path}' - it already exists and mode is 'error'. "
                f"Choose a different approach: "
                f"1) Use mode='overwrite' to replace the existing file, "
                f"2) Use mode='ignore' to skip saving if file exists, "
                f"3) Use a different file path."
            )
        if mode == "ignore" and file_exists:
            logger.warning(f"File {file_path} already exists, ignoring write.")
            return QueryMetrics()

        physical_plan = self.transpiler.transpile(logical_plan)
        try:
            _, metrics = physical_plan.execute()
        except Exception as e:
            raise ExecutionError(f"Failed to execute query: {e}") from e
        return metrics

    def infer_schema_from_csv(
        self, paths: list[str], **options: Dict[str, Any]
    ) -> Schema:
        """Infer the schema of a CSV file."""
        self.session_state._check_active()
        query = self._build_read_csv_query(paths, True, **options)
        return self._infer_schema_from_file_scan_query(paths, query)

    def infer_schema_from_parquet(
        self, paths: list[str], **options: Dict[str, Any]
    ) -> Schema:
        """Infer the schema of a Parquet file."""
        self.session_state._check_active()
        query = self._build_read_parquet_query(paths, True, **options)
        return self._infer_schema_from_file_scan_query(paths, query)

    def _infer_schema_from_file_scan_query(
        self, paths: list[str], query: str
    ) -> Schema:
        """Helper method to infer schema from a DuckDB file scan query."""
        query = f"PRAGMA disable_optimizer; {query}"
        df = query_files(query=query, paths=paths, s3_session=self.session_state.s3_session)
        polars_schema = df.schema
        return convert_polars_schema_to_custom_schema(polars_schema)

    def _build_read_csv_query(
        self, paths: list[str], infer_schema: bool, **options: Dict[str, Any]
    ) -> str:
        """Helper method to build a DuckDB read CSV query."""
        merge_schemas = options.get("merge_schemas", False)
        schema: Optional[Schema] = options.get("schema", None)
        duckdb_schema: Dict[str, str] = {}
        paths_str = "', '".join(paths)
        # trunk-ignore-begin(bandit/B608)
        if schema:
            for col_field in schema.column_fields:
                duckdb_type: str | None = None
                if col_field.data_type == StringType:
                    duckdb_type = "VARCHAR"
                elif col_field.data_type == IntegerType:
                    duckdb_type = "BIGINT"
                elif col_field.data_type == FloatType:
                    duckdb_type = "FLOAT"
                elif col_field.data_type == DoubleType:
                    duckdb_type = "DOUBLE"
                elif col_field.data_type == BooleanType:
                    duckdb_type = "BOOLEAN"
                else:
                    raise ValidationError(
                        f"Invalid column type for csv Schema: ColumnField(name='{col_field.name}', data_type={type(col_field.data_type).__name__}). "
                        f"Expected one of: IntegerType, FloatType, DoubleType, BooleanType, or StringType. as data_type"
                        f"Example: Schema([ColumnField(name='id', data_type=IntegerType), ColumnField(name='name', data_type=StringType)])"
                    )
                duckdb_schema[col_field.name] = duckdb_type
            duckdb_schema_string = json.dumps(duckdb_schema).replace('"', "'")
            query = f"SELECT * FROM read_csv(['{paths_str}'], columns = {duckdb_schema_string})"
        elif merge_schemas:
            query = f"SELECT * FROM read_csv(['{paths_str}'], union_by_name=true)"
        else:
            query = f"SELECT * FROM read_csv(['{paths_str}'])"
        if infer_schema:
            query = f"{query} WHERE 1=0"
        # trunk-ignore-end(bandit/B608)
        return query

    def _build_read_parquet_query(
        self, paths: list[str], infer_schema: bool, **options: Dict[str, Any]
    ) -> str:
        """Helper method to build a DuckDB read Parquet query."""
        merge_schemas = options.get("merge_schemas", False)
        paths_str = "', '".join(paths)
        # trunk-ignore-begin(bandit/B608)
        if merge_schemas:
            query = f"SELECT * FROM read_parquet(['{paths_str}'], union_by_name=true)"
        else:
            query = f"SELECT * FROM read_parquet(['{paths_str}'])"
        if infer_schema:
            query = f"{query} WHERE 1=0"
        # trunk-ignore-end(bandit/B608)
        return query
