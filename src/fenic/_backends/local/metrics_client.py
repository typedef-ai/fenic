"""Client for managing metrics storage in DuckDB."""

import logging
from typing import Dict, List

import duckdb
import polars as pl

from fenic.core.metrics import QueryMetrics
from fenic.core.types import Field, FieldType, Schema

logger = logging.getLogger(__name__)

METRICS_TABLE_NAME = "metrics"
METRICS_SCHEMA = Schema([
    Field("row_id", FieldType.Int64),
    Field("execution_id", FieldType.String),
    Field("session_id", FieldType.String),
    Field("start_ts", FieldType.Float64),
    Field("end_ts", FieldType.Float64),
    Field("execution_time_ms", FieldType.Float64),
    Field("num_output_rows", FieldType.Int64),
    Field("lm_cost", FieldType.Float64),
    Field("lm_input_tokens", FieldType.Int64),
    Field("lm_cached_input_tokens", FieldType.Int64),
    Field("lm_output_tokens", FieldType.Int64),
    Field("lm_requests", FieldType.Int64),
    Field("rm_cost", FieldType.Float64),
    Field("rm_input_tokens", FieldType.Int64),
    Field("rm_requests", FieldType.Int64),
])


class MetricsClient:
    """Client for managing metrics storage in a local DuckDB instance."""

    def __init__(self, connection: duckdb.DuckDBPyConnection):
        """Initialize the metrics client.
        
        Args:
            connection: DuckDB connection instance
        """
        self.db_conn = connection
        self._row_counter = 0
        self._initialize_metrics_table()

    def _initialize_metrics_table(self):
        """Initialize the metrics table if it doesn't exist."""
        try:
            # Check if metrics table exists
            table_exists = self.db_conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                (METRICS_TABLE_NAME,)
            ).fetchone()[0] > 0

            if not table_exists:
                # Create metrics table with proper schema
                create_sql = f"""
                CREATE TABLE {METRICS_TABLE_NAME} (
                    row_id BIGINT PRIMARY KEY,
                    execution_id VARCHAR,
                    session_id VARCHAR,
                    start_ts DOUBLE,
                    end_ts DOUBLE,
                    execution_time_ms DOUBLE,
                    num_output_rows BIGINT,
                    lm_cost DOUBLE,
                    lm_input_tokens BIGINT,
                    lm_cached_input_tokens BIGINT,
                    lm_output_tokens BIGINT,
                    lm_requests BIGINT,
                    rm_cost DOUBLE,
                    rm_input_tokens BIGINT,
                    rm_requests BIGINT
                );
                """
                self.db_conn.execute(create_sql)
                logger.info(f"Created metrics table: {METRICS_TABLE_NAME}")
            else:
                # Get the current max row_id to continue sequence
                max_id_result = self.db_conn.execute(
                    f"SELECT COALESCE(MAX(row_id), 0) FROM {METRICS_TABLE_NAME}"
                ).fetchone()
                self._row_counter = max_id_result[0] if max_id_result else 0
                logger.debug(f"Metrics table exists, continuing from row_id: {self._row_counter}")

        except Exception as e:
            logger.error(f"Failed to initialize metrics table: {e}")
            raise

    def append_metrics(self, query_metrics: QueryMetrics) -> int:
        """Append QueryMetrics to the metrics table.
        
        Args:
            query_metrics: QueryMetrics instance to store
            
        Returns:
            int: The row_id assigned to this metrics entry
        """
        try:
            # Increment row counter
            self._row_counter += 1
            row_id = self._row_counter

            # Convert QueryMetrics to row dict
            row_data = query_metrics.to_row_dict()
            row_data["row_id"] = row_id

            # Create DataFrame with single row
            df = pl.DataFrame([row_data])

            # Insert into DuckDB
            temp_view_name = f"temp_metrics_{row_id}"
            self.db_conn.register(temp_view_name, df)
            
            insert_sql = f"""
            INSERT INTO {METRICS_TABLE_NAME}
            SELECT * FROM {temp_view_name}
            """
            self.db_conn.execute(insert_sql)
            
            # Clean up temp view
            self.db_conn.execute(f"DROP VIEW IF EXISTS {temp_view_name}")
            
            logger.debug(f"Appended metrics with execution_id: {query_metrics.execution_id}, row_id: {row_id}")
            return row_id

        except Exception as e:
            logger.error(f"Failed to append metrics: {e}")
            raise

    def get_session_aggregated_metrics(self, session_id: str) -> Dict[str, float]:
        """Get aggregated metrics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing aggregated metrics for the session
        """
        try:
            query = f"""
            SELECT 
                COUNT(*) as total_queries,
                SUM(execution_time_ms) as total_execution_time_ms,
                SUM(num_output_rows) as total_output_rows,
                SUM(lm_cost) as total_lm_cost,
                SUM(lm_input_tokens) as total_lm_input_tokens,
                SUM(lm_cached_input_tokens) as total_lm_cached_input_tokens,
                SUM(lm_output_tokens) as total_lm_output_tokens,
                SUM(lm_requests) as total_lm_requests,
                SUM(rm_cost) as total_rm_cost,
                SUM(rm_input_tokens) as total_rm_input_tokens,
                SUM(rm_requests) as total_rm_requests
            FROM {METRICS_TABLE_NAME}
            WHERE session_id = ?
            """
            
            result = self.db_conn.execute(query, (session_id,)).fetchone()
            
            if result and result[0] > 0:  # Check if we have any records
                return {
                    "total_queries": result[0],
                    "total_execution_time_ms": result[1] or 0.0,
                    "total_output_rows": result[2] or 0,
                    "total_lm_cost": result[3] or 0.0,
                    "total_lm_input_tokens": result[4] or 0,
                    "total_lm_cached_input_tokens": result[5] or 0,
                    "total_lm_output_tokens": result[6] or 0,
                    "total_lm_requests": result[7] or 0,
                    "total_rm_cost": result[8] or 0.0,
                    "total_rm_input_tokens": result[9] or 0,
                    "total_rm_requests": result[10] or 0,
                }
            else:
                return {
                    "total_queries": 0,
                    "total_execution_time_ms": 0.0,
                    "total_output_rows": 0,
                    "total_lm_cost": 0.0,
                    "total_lm_input_tokens": 0,
                    "total_lm_cached_input_tokens": 0,
                    "total_lm_output_tokens": 0,
                    "total_lm_requests": 0,
                    "total_rm_cost": 0.0,
                    "total_rm_input_tokens": 0,
                    "total_rm_requests": 0,
                }

        except Exception as e:
            logger.error(f"Failed to get session aggregated metrics: {e}")
            raise

    def read_metrics_table(self) -> pl.DataFrame:
        """Read the entire metrics table as a Polars DataFrame.
        
        Returns:
            Polars DataFrame containing all metrics data
        """
        try:
            return self.db_conn.execute(f"SELECT * FROM {METRICS_TABLE_NAME} ORDER BY row_id").pl()
        except Exception as e:
            logger.error(f"Failed to read metrics table: {e}")
            raise