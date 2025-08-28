"""Writer interface for saving DataFrames to external storage systems."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    from fenic.api.dataframe import DataFrame

from pydantic import ConfigDict, validate_call

from fenic.core._logical_plan.plans import FileSink, TableSink
from fenic.core.error import ValidationError
from fenic.core.metrics import QueryMetrics

logger = logging.getLogger(__name__)


class DataFrameWriter:
    """Interface used to write a DataFrame to external storage systems.
    
    Similar to PySpark's DataFrameWriter.

    Supported External Storage Schemes:
    - Amazon S3 (s3://)
        - Format: s3://{bucket_name}/{path_to_file}

        - Notes:
            - Uses boto3 to aquire AWS credentials.

        - Examples:
            - s3://my-bucket/data.csv
            - s3://my-bucket/data/*.parquet

    - Local Files (file:// or implicit)
        - Format: file://{absolute_or_relative_path}

        - Notes:
            - Paths without a scheme (e.g., ./data.csv or /tmp/data.parquet) are treated as local files
        - Examples:
            - file:///home/user/data.csv
            - ./data/*.parquet
    """

    def __init__(self, dataframe: DataFrame):
        """Initialize a DataFrameWriter.

        Args:
            dataframe: The DataFrame to write.
        """
        self._dataframe = dataframe

    def save_as_table(
        self,
        table_name: str,
        mode: Literal["error", "append", "overwrite", "ignore"] = "error",
    ) -> QueryMetrics:
        """Saves the content of the DataFrame as the specified table.

        Args:
            table_name: Name of the table to save to
            mode: Write mode. Default is "error".
                 - error: Raises an error if table exists
                 - append: Appends data to table if it exists
                 - overwrite: Overwrites existing table
                 - ignore: Silently ignores operation if table exists

        Returns:
            QueryMetrics: The query metrics

        Example: Save with error mode (default)
            ```python
            df.write.save_as_table("my_table")  # Raises error if table exists
            ```

        Example: Save with append mode
            ```python
            df.write.save_as_table("my_table", mode="append")  # Adds to existing table
            ```

        Example: Save with overwrite mode
            ```python
            df.write.save_as_table("my_table", mode="overwrite")  # Replaces existing table
            ```
        """
        sink_plan = TableSink.from_session_state(
            child=self._dataframe._logical_plan,
            table_name=table_name,
            mode=mode,
            session_state=self._dataframe._session_state,
        )

        metrics = self._dataframe._session_state.execution.save_as_table(
            sink_plan, table_name=table_name, mode=mode
        )
        logger.info(metrics.get_summary())
        return metrics

    def save_as_view(
        self,
        view_name: str,
        description: str | None = None,
    ) -> None:
        """Saves the content of the DataFrame as a view.

        Args:
            view_name: Name of the view to save to
            description: Optional human-readable view description to store in the catalog.

        Returns:
            None.
        """
        self._dataframe._session_state.execution.save_as_view(
            logical_plan=self._dataframe._logical_plan, view_name=view_name, view_description=description
        )

    def csv(
        self,
        file_path: Union[str, Path],
        mode: Literal["error", "overwrite", "ignore"] = "overwrite",
    ) -> QueryMetrics:
        """Saves the content of the DataFrame as a single CSV file with comma as the delimiter and headers in the first row.

        Args:
            file_path: Path to save the CSV file to
            mode: Write mode. Default is "overwrite".
                 - error: Raises an error if file exists
                 - overwrite: Overwrites the file if it exists
                 - ignore: Silently ignores operation if file exists

        Returns:
            QueryMetrics: The query metrics

        Example: Save with overwrite mode (default)
            ```python
            df.write.csv("output.csv")  # Overwrites if exists
            ```

        Example: Save with error mode
            ```python
            df.write.csv("output.csv", mode="error")  # Raises error if exists
            ```

        Example: Save with ignore mode
            ```python
            df.write.csv("output.csv", mode="ignore")  # Skips if exists
            ```
        """
        file_path = str(file_path)
        if not file_path.endswith(".csv"):
            raise ValidationError(
                f"CSV writer requires a '.csv' file extension. "
                f"Your path '{file_path}' is missing the extension."
            )

        sink_plan = FileSink.from_session_state(
            child=self._dataframe._logical_plan,
            sink_type="csv",
            path=file_path,
            mode=mode,
            session_state=self._dataframe._session_state,
        )

        metrics = self._dataframe._session_state.execution.save_to_file(
            sink_plan, file_path=file_path, mode=mode
        )
        logger.info(metrics.get_summary())
        return metrics

    def parquet(
        self,
        file_path: Union[str, Path],
        mode: Literal["error", "overwrite", "ignore"] = "overwrite",
    ) -> QueryMetrics:
        """Saves the content of the DataFrame as a single Parquet file.

        Args:
            file_path: Path to save the Parquet file to
            mode: Write mode. Default is "overwrite".
                 - error: Raises an error if file exists
                 - overwrite: Overwrites the file if it exists
                 - ignore: Silently ignores operation if file exists

        Returns:
            QueryMetrics: The query metrics

        Example: Save with overwrite mode (default)
            ```python
            df.write.parquet("output.parquet")  # Overwrites if exists
            ```

        Example: Save with error mode
            ```python
            df.write.parquet("output.parquet", mode="error")  # Raises error if exists
            ```

        Example: Save with ignore mode
            ```python
            df.write.parquet("output.parquet", mode="ignore")  # Skips if exists
            ```
        """
        file_path = str(file_path)
        if not file_path.endswith(".parquet"):
            raise ValidationError(
                f"Parquet writer requires a '.parquet' file extension. "
                f"Your path '{file_path}' is missing the extension."
            )

        sink_plan = FileSink.from_session_state(
            child=self._dataframe._logical_plan,
            sink_type="parquet",
            path=file_path,
            mode=mode,
            session_state=self._dataframe._session_state,
        )

        metrics = self._dataframe._session_state.execution.save_to_file(
            sink_plan, file_path=file_path, mode=mode
        )
        logger.info(metrics.get_summary())
        return metrics


DataFrameWriter.save_as_table = validate_call(config=ConfigDict(strict=True))(
    DataFrameWriter.save_as_table
)
DataFrameWriter.saveAsTable = DataFrameWriter.save_as_table
DataFrameWriter.csv = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(DataFrameWriter.csv)
DataFrameWriter.parquet = validate_call(
    config=ConfigDict(strict=True, arbitrary_types_allowed=True)
)(DataFrameWriter.parquet)
