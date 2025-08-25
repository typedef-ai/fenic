"""Reader interface for loading DataFrames from external storage systems."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

from fenic.core.types import Schema
from fenic.core.types.datatypes import _PrimitiveType

if TYPE_CHECKING:
    from fenic.api.dataframe import DataFrame
    from fenic.core._interfaces import BaseSessionState

from fenic.core._logical_plan.plans import FileSource
from fenic.core.error import ValidationError


class DataFrameReader:
    """Interface used to load a DataFrame from external storage systems.
    
    Similar to PySpark's DataFrameReader.

    Supported External Storage Schemes:
    - Amazon S3 (s3://)
        - Format: s3://{bucket_name}/{path_to_file}

        - Notes:
            - Uses boto3 to aquire AWS credentials.

        - Examples:
            - s3://my-bucket/data.csv
            - s3://my-bucket/data/*.parquet

    - Hugging Face Datasets (hf://)
        - Format: hf://{repo_type}/{repo_id}/{path_to_file}

        - Notes:
            - Supports glob patterns (*, **)
            - Supports dataset revisions and branch aliases (e.g., @refs/convert/parquet, @~parquet)
            - HF_TOKEN environment variable is required to read private datasets.

        - Examples:
            - hf://datasets/datasets-examples/doc-formats-csv-1/data.csv
            - hf://datasets/cais/mmlu/astronomy/*.parquet
            - hf://datasets/datasets-examples/doc-formats-csv-1@~parquet/**/*.parquet

    - Local Files (file:// or implicit)
        - Format: file://{absolute_or_relative_path}

        - Notes:
            - Paths without a scheme (e.g., ./data.csv or /tmp/data.parquet) are treated as local files
        - Examples:
            - file:///home/user/data.csv
            - ./data/*.parquet
    """

    def __init__(self, session_state: BaseSessionState):
        """Creates a DataFrameReader.

        Args:
            session_state: The session state to use for reading
        """
        self._options: Dict[str, Any] = {}
        self._session_state = session_state

    def csv(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        schema: Optional[Schema] = None,
        merge_schemas: bool = False,
    ) -> DataFrame:
        """Load a DataFrame from one or more CSV files.

        Args:
            paths: A single file path, a glob pattern (e.g., "data/*.csv"), or a list of paths.
            schema: (optional) A complete schema definition of column names and their types. Only primitive types are supported.
                - For e.g.:
                    - Schema([ColumnField(name="id", data_type=IntegerType), ColumnField(name="name", data_type=StringType)])
                - If provided, all files must match this schema exactly—all column names must be present, and values must be
                convertible to the specified types. Partial schemas are not allowed.
            merge_schemas: Whether to merge schemas across all files.
                - If True: Column names are unified across files. Missing columns are filled with nulls. Column types are
                inferred and widened as needed.
                - If False (default): Only accepts columns from the first file. Column types from the first file are
                inferred and applied across all files. If subsequent files do not have the same column name and order as the first file, an error is raised.
                - The "first file" is defined as:
                    - The first file in lexicographic order (for glob patterns), or
                    - The first file in the provided list (for lists of paths).

        Notes:
            - The first row in each file is assumed to be a header row.
            - Delimiters (e.g., comma, tab) are automatically inferred.
            - You may specify either `schema` or `merge_schemas=True`, but not both.
            - Any date/datetime columns are cast to strings during ingestion.

        Raises:
            ValidationError: If both `schema` and `merge_schemas=True` are provided.
            ValidationError: If any path does not end with `.csv`.
            PlanError: If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

        Example: Read a single CSV file
            ```python
            df = session.read.csv("file.csv")
            ```

        Example: Read multiple CSV files with schema merging
            ```python
            df = session.read.csv("data/*.csv", merge_schemas=True)
            ```

        Example: Read CSV files with explicit schema
            ```python
            df = session.read.csv(
                ["a.csv", "b.csv"],
                schema=Schema([
                    ColumnField(name="id", data_type=IntegerType),
                    ColumnField(name="value", data_type=FloatType)
                ])
            )            ```
        """
        if schema is not None and merge_schemas:
            raise ValidationError(
                "Cannot specify both 'schema' and 'merge_schemas=True' - these options conflict. "
                "Choose one approach: "
                "1) Use 'schema' to enforce a specific schema: csv(paths, schema=your_schema), "
                "2) Use 'merge_schemas=True' to automatically merge schemas: csv(paths, merge_schemas=True), "
                "3) Use neither to inherit schema from the first file: csv(paths)"
            )
        if schema is not None:
            for col_field in schema.column_fields:
                if not isinstance(
                    col_field.data_type,
                    _PrimitiveType,
                ):
                    raise ValidationError(
                        f"CSV files only support primitive data types in schema definitions. "
                        f"Column '{col_field.name}' has type {type(col_field.data_type).__name__}, but CSV schemas must use: "
                        f"IntegerType, FloatType, DoubleType, BooleanType, or StringType. "
                        f"Example: Schema([ColumnField(name='id', data_type=IntegerType), ColumnField(name='name', data_type=StringType)])"
                    )
        options = {
            "merge_schemas": merge_schemas,
        }
        if schema:
            options["schema"] = schema
        return self._read_file(
            paths, file_format="csv", file_extension=".csv", **options
        )

    def parquet(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        merge_schemas: bool = False,
    ) -> DataFrame:
        """Load a DataFrame from one or more Parquet files.

        Args:
            paths: A single file path, a glob pattern (e.g., "data/*.parquet"), or a list of paths.
            merge_schemas: If True, infers and merges schemas across all files.
                Missing columns are filled with nulls, and differing types are widened to a common supertype.

        Behavior:
            - If `merge_schemas=False` (default), all files must match the schema of the first file exactly.
            Subsequent files must contain all columns from the first file with compatible data types.
            If any column is missing or has incompatible types, an error is raised.
            - If `merge_schemas=True`, column names are unified across all files, and data types are automatically
            widened to accommodate all values.
            - The "first file" is defined as:
                - The first file in lexicographic order (for glob patterns), or
                - The first file in the provided list (for lists of paths).

        Notes:
            - Date and datetime columns are cast to strings during ingestion.

        Raises:
            ValidationError: If any file does not have a `.parquet` extension.
            PlanError: If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

        Example: Read a single Parquet file
            ```python
            df = session.read.parquet("file.parquet")
            ```

        Example: Read multiple Parquet files
            ```python
            df = session.read.parquet("data/*.parquet")
            ```

        Example: Read Parquet files with schema merging
            ```python
            df = session.read.parquet(["a.parquet", "b.parquet"], merge_schemas=True)
            ```
        """
        options = {
            "merge_schemas": merge_schemas,
        }
        return self._read_file(
            paths, file_format="parquet", file_extension=".parquet", **options
        )

    def _read_file(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        file_format: Literal["csv", "parquet"],
        file_extension: str,
        **options,
    ) -> DataFrame:
        """Internal helper method to read files of a specific format.

        Args:
            paths: Path(s) to the file(s). Can be a single path or a list of paths.
            file_format: Format of the file (e.g., "csv", "parquet").
            file_extension: Expected file extension (e.g., ".csv", ".parquet").
            **options: Additional options to pass to the file reader.

        Returns:
            DataFrame loaded from the specified file(s).

        Raises:
            ValidationError: If any path doesn't end with the expected file extension.
            ValidationError: If paths is not a string, Path, or list of strings/Paths.
        """
        # Validate paths type
        if not isinstance(paths, (str, Path, list)):
            raise ValidationError(
                f"Expected paths to be str, Path, or list, got {type(paths).__name__}"
            )

        # Convert to list if it's a single path
        if isinstance(paths, (str, Path)):
            paths_list = [paths]
        else:
            # Validate each item in the list
            for i, path in enumerate(paths):
                if not isinstance(path, (str, Path)):
                    raise ValidationError(
                        f"Expected path at index {i} to be str or Path, got {type(path).__name__}"
                    )
            paths_list = paths

        # Validate file extensions
        for path in paths_list:
            if not str(path).endswith(file_extension):
                raise ValidationError(
                    f"Invalid file extension for {file_format.upper()} reader: '{path}' does not end with '{file_extension}'. "
                    f"Please ensure all paths have the correct extension. "
                    f"Example: 'data/file{file_extension}' or 'data/*{file_extension}'"
                )

        # Convert all paths to strings
        paths_str = [str(p) for p in paths_list]

        logical_node = FileSource.from_session_state(
            paths=paths_str,
            file_format=file_format,
            options=options,
            session_state=self._session_state,
        )
        from fenic.api.dataframe import DataFrame

        return DataFrame._from_logical_plan(logical_node, self._session_state)
