# fenic.api.io

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/api/io/

IO module for reading and writing DataFrames to external storage.

Classes:

- **`DataFrameReader`**
  –

  Interface used to load a DataFrame from external storage systems.
- **`DataFrameWriter`**
  –

  Interface used to write a DataFrame to external storage systems.

## DataFrameReader

```
DataFrameReader(session_state: BaseSessionState)
```

Interface used to load a DataFrame from external storage systems.

Similar to PySpark's DataFrameReader.

Supported External Storage Schemes:
- Amazon S3 (s3://)
- Format: s3://{bucket_name}/{path_to_file}

```
- Notes:
    - Uses boto3 to aquire AWS credentials.

- Examples:
    - s3://my-bucket/data.csv
    - s3://my-bucket/data/*.parquet
```

- Hugging Face Datasets (hf://)

  - Format: hf://{repo_type}/{repo_id}/{path_to_file}
  - Notes:

    - Supports glob patterns (*,* \*)
    - Supports dataset revisions and branch aliases (e.g., @refs/convert/parquet, @~parquet)
    - HF_TOKEN environment variable is required to read private datasets.
  - Examples:

    - hf://datasets/datasets-examples/doc-formats-csv-1/data.csv
    - hf://datasets/cais/mmlu/astronomy/\*.parquet
    - hf://datasets/datasets-examples/doc-formats-csv-1@~parquet/\**/*.parquet
- Local Files (file:// or implicit)

  - Format: file://{absolute_or_relative_path}
  - Notes:

    - Paths without a scheme (e.g., ./data.csv or /tmp/data.parquet) are treated as local files
  - Examples:
    - file:///home/user/data.csv
    - ./data/\*.parquet

Creates a DataFrameReader.

Parameters:

- **`session_state`**
  (`BaseSessionState`)
  –

  The session state to use for reading

Methods:

- **`csv`**
  –

  Load a DataFrame from one or more CSV files.
- **`docs`**
  –

  Load a DataFrame with the document contents of a list of paths (markdown or json).
- **`parquet`**
  –

  Load a DataFrame from one or more Parquet files.
- **`pdf_metadata`**
  –

  Load a DataFrame with metadata of PDF files in a list of paths.

Source code in `src/fenic/api/io/reader.py`

```
def __init__(self, session_state: BaseSessionState):
    """Creates a DataFrameReader.

    Args:
        session_state: The session state to use for reading
    """
    self._options: Dict[str, Any] = {}
    self._session_state = session_state
```

### csv

```
csv(paths: Union[str, Path, list[Union[str, Path]]], schema: Optional[Schema] = None, merge_schemas: bool = False) -> DataFrame
```

Load a DataFrame from one or more CSV files.

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  A single file path, a glob pattern (e.g., "data/\*.csv"), or a list of paths.
- **`schema`**
  (`Optional[Schema]`, default:
  `None`
  )
  –

  (optional) A complete schema definition of column names and their types. Only primitive types are supported.
  - For e.g.:
  - Schema([ColumnField(name="id", data_type=IntegerType), ColumnField(name="name", data_type=StringType)])
  - If provided, all files must match this schema exactly—all column names must be present, and values must be
  convertible to the specified types. Partial schemas are not allowed.
- **`merge_schemas`**
  (`bool`, default:
  `False`
  )
  –

  Whether to merge schemas across all files.
  - If True: Column names are unified across files. Missing columns are filled with nulls. Column types are
  inferred and widened as needed.
  - If False (default): Only accepts columns from the first file. Column types from the first file are
  inferred and applied across all files. If subsequent files do not have the same column name and order as the first file, an error is raised.
  - The "first file" is defined as:
  - The first file in lexicographic order (for glob patterns), or
  - The first file in the provided list (for lists of paths).

Notes

- The first row in each file is assumed to be a header row.
- Delimiters (e.g., comma, tab) are automatically inferred.
- You may specify either `schema` or `merge_schemas=True`, but not both.
- Any date/datetime columns are cast to strings during ingestion.

Raises:

- `ValidationError`
  –

  If both `schema` and `merge_schemas=True` are provided.
- `ValidationError`
  –

  If any path does not end with `.csv`.
- `PlanError`
  –

  If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

Read a single CSV file

```
df = session.read.csv("file.csv")
```

Read multiple CSV files with schema merging

```
df = session.read.csv("data/*.csv", merge_schemas=True)
```

Read CSV files with explicit schema

`python
df = session.read.csv(
["a.csv", "b.csv"],
schema=Schema([
ColumnField(name="id", data_type=IntegerType),
ColumnField(name="value", data_type=FloatType)
])
)`

Source code in `src/fenic/api/io/reader.py`

```
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
```

### docs

```
docs(paths: Union[str, Path, list[Union[str, Path]]], content_type: Literal['markdown', 'json'], exclude: Optional[str] = None, recursive: bool = False) -> DataFrame
```

Load a DataFrame with the document contents of a list of paths (markdown or json).

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  Glob pattern (or list of glob patterns) to the folder(s) to load.
- **`content_type`**
  (`Literal['markdown', 'json']`)
  –

  Content type of the files. One of "markdown" or "json".
- **`exclude`**
  (`Optional[str]`, default:
  `None`
  )
  –

  A regex pattern to exclude files.
  If it is not provided no files will be excluded.
- **`recursive`**
  (`bool`, default:
  `False`
  )
  –

  Whether to recursively load files from the folder.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A dataframe with all the documents found in the paths.
  The content of each document is a row in the dataframe.

Raises:

- `ValidationError`
  –

  If any file does not have a `.md` or `.json` depending on the content_type.
- `UnsupportedFileTypeError`
  –

  If the specified content_type is not "markdown" or "json" .

Notes

- Each row in the dataframe corresponds to a file in the list of paths.
- The dataframe has the following columns:
  - file_path: The path to the file.
  - error: The error message if the file failed to be loaded.
  - content: The content of the file casted to the content_type.
- Recursive loading is supported in conjunction with the '\**' glob pattern,
  e.g. `data/**/*.md` will load all markdown files in the `data` folder and all subfolders
  when recursive is set to True.
  Without recursive = True, then \*\* behaves like a single '*' pattern.

Read all the markdown files in a folder and all its subfolders.

```
df = session.read.docs("data/docs/**/*.md", content_type="markdown", recursive=True)
```

Read a folder of markdown files excluding some files.

```
df = session.read.docs("data/docs/*.md", content_type="markdown", exclude=r"\.bak.md$")
```

Source code in `src/fenic/api/io/reader.py`

```
def docs(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        content_type: Literal["markdown", "json"],
        exclude: Optional[str] = None,
        recursive: bool = False,
) -> DataFrame:
    r"""Load a DataFrame with the document contents of a list of paths (markdown or json).

    Args:
        paths: Glob pattern (or list of glob patterns) to the folder(s) to load.
        content_type: Content type of the files. One of "markdown" or "json".
        exclude: A regex pattern to exclude files.
                 If it is not provided no files will be excluded.
        recursive: Whether to recursively load files from the folder.

    Returns:
        DataFrame: A dataframe with all the documents found in the paths.
                   The content of each document is a row in the dataframe.

    Raises:
        ValidationError: If any file does not have a `.md` or `.json` depending on the content_type.
        UnsupportedFileTypeError: If the specified content_type is not "markdown" or "json" .

    Notes:
        - Each row in the dataframe corresponds to a file in the list of paths.
        - The dataframe has the following columns:
            - file_path: The path to the file.
            - error: The error message if the file failed to be loaded.
            - content: The content of the file casted to the content_type.
        - Recursive loading is supported in conjunction with the '**' glob pattern,
          e.g. `data/**/*.md` will load all markdown files in the `data` folder and all subfolders
               when recursive is set to True.
          Without recursive = True, then ** behaves like a single '*' pattern.

    Example: Read all the markdown files in a folder and all its subfolders.
        ```python
        df = session.read.docs("data/docs/**/*.md", content_type="markdown", recursive=True)
        ```

    Example: Read a folder of markdown files excluding some files.
        ```python
        df = session.read.docs("data/docs/*.md", content_type="markdown", exclude=r"\.bak.md$")
        ```

    """
    path_str_list = validate_paths_and_return_list_of_strings(paths)

    if content_type not in ["markdown", "json"]:
        raise UnsupportedFileTypeError(f"{content_type}, must be 'markdown' or 'json'")

    logical_node = DocSource.from_session_state(
        paths=path_str_list,
        content_type=content_type,
        exclude=exclude,
        recursive=recursive,
        session_state=self._session_state,
    )
    from fenic.api.dataframe import DataFrame

    return DataFrame._from_logical_plan(logical_node, self._session_state)
```

### parquet

```
parquet(paths: Union[str, Path, list[Union[str, Path]]], merge_schemas: bool = False) -> DataFrame
```

Load a DataFrame from one or more Parquet files.

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  A single file path, a glob pattern (e.g., "data/\*.parquet"), or a list of paths.
- **`merge_schemas`**
  (`bool`, default:
  `False`
  )
  –

  If True, infers and merges schemas across all files.
  Missing columns are filled with nulls, and differing types are widened to a common supertype.

Behavior

- If `merge_schemas=False` (default), all files must match the schema of the first file exactly.
  Subsequent files must contain all columns from the first file with compatible data types.
  If any column is missing or has incompatible types, an error is raised.
- If `merge_schemas=True`, column names are unified across all files, and data types are automatically
  widened to accommodate all values.
- The "first file" is defined as:
  - The first file in lexicographic order (for glob patterns), or
  - The first file in the provided list (for lists of paths).

Notes

- Date and datetime columns are cast to strings during ingestion.

Raises:

- `ValidationError`
  –

  If any file does not have a `.parquet` extension.
- `PlanError`
  –

  If schemas cannot be merged or if there's a schema mismatch when merge_schemas=False.

Read a single Parquet file

```
df = session.read.parquet("file.parquet")
```

Read multiple Parquet files

```
df = session.read.parquet("data/*.parquet")
```

Read Parquet files with schema merging

```
df = session.read.parquet(["a.parquet", "b.parquet"], merge_schemas=True)
```

Source code in `src/fenic/api/io/reader.py`

```
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
```

### pdf_metadata

```
pdf_metadata(paths: Union[str, Path, list[Union[str, Path]]], exclude: Optional[str] = None, recursive: bool = False) -> DataFrame
```

Load a DataFrame with metadata of PDF files in a list of paths.

Note

Local execution requires the `pdf` extra: `pip install "fenic[pdf]"`.

Parameters:

- **`paths`**
  (`Union[str, Path, list[Union[str, Path]]]`)
  –

  Glob pattern (or list of glob patterns) to the folder(s) to load.
- **`exclude`**
  (`Optional[str]`, default:
  `None`
  )
  –

  A regex pattern to exclude files.
  If it is not provided no files will be excluded.
- **`recursive`**
  (`bool`, default:
  `False`
  )
  –

  Whether to recursively load files from the folder.

Returns:

- **`DataFrame`** ( `DataFrame`
  ) –

  A dataframe with the metadata of all the PDF files found in the paths.
  the metadata from a single PDF document is a row in the dataframe.

Raises:

- `ValidationError`
  –

  If any file does not have a `.pdf` extension.

Notes

- Each row in the dataframe corresponds to a file in the list of paths.
- The metadata columns are:
  - file_path: The path to the document.
  - error: The error message if the file failed to be loaded.
  - size: Size of the PDF file in bytes.
  - title: Title of the PDF document.
  - author: Author of the PDF document.
  - creation_date: Creation date of the PDF.
  - mod_date: Modification date of the PDF.
  - page_count: Number of pages in the PDF.
  - has_forms: Whether the PDF contains form fields, or fields that accept user input.
  - has_signature_fields: Whether the PDF contains signature fields.
  - image_count: Number of images in the PDF.
  - is_encrypted: Whether the PDF is encrypted.
- Recursive loading is supported in conjunction with the '\**' glob pattern,
  e.g. `data/**/*.pdf` will load all PDF files in the `data` folder and all subfolders
  when recursive is set to True.
  Without recursive = True, then \*\* behaves like a single '*' pattern.

Read the metadata of all the PDF files in a folder and all its subfolders.

```
df = session.read.pdf_metadata("data/docs/**/*.pdf", recursive=True)
```

Read a metadata of PDFS in a folder, excluding some files.

```
df = session.read.pdf_metadata("data/docs/*.pdf", exclude=r"\.backup.pdf$")
```

Source code in `src/fenic/api/io/reader.py`

```
def pdf_metadata(
        self,
        paths: Union[str, Path, list[Union[str, Path]]],
        exclude: Optional[str] = None,
        recursive: bool = False,
) -> DataFrame:
    r"""Load a DataFrame with metadata of PDF files in a list of paths.

    Note:
        Local execution requires the `pdf` extra: `pip install "fenic[pdf]"`.

    Args:
        paths: Glob pattern (or list of glob patterns) to the folder(s) to load.
        exclude: A regex pattern to exclude files.
                 If it is not provided no files will be excluded.
        recursive: Whether to recursively load files from the folder.

    Returns:
        DataFrame: A dataframe with the metadata of all the PDF files found in the paths.
                   the metadata from a single PDF document is a row in the dataframe.

    Raises:
        ValidationError: If any file does not have a `.pdf` extension.

    Notes:
        - Each row in the dataframe corresponds to a file in the list of paths.
        - The metadata columns are:
            - file_path: The path to the document.
            - error: The error message if the file failed to be loaded.
            - size: Size of the PDF file in bytes.
            - title: Title of the PDF document.
            - author: Author of the PDF document.
            - creation_date: Creation date of the PDF.
            - mod_date: Modification date of the PDF.
            - page_count: Number of pages in the PDF.
            - has_forms: Whether the PDF contains form fields, or fields that accept user input.
            - has_signature_fields: Whether the PDF contains signature fields.
            - image_count: Number of images in the PDF.
            - is_encrypted: Whether the PDF is encrypted.
        - Recursive loading is supported in conjunction with the '**' glob pattern,
          e.g. `data/**/*.pdf` will load all PDF files in the `data` folder and all subfolders
               when recursive is set to True.
          Without recursive = True, then ** behaves like a single '*' pattern.

    Example: Read the metadata of all the PDF files in a folder and all its subfolders.
        ```python
        df = session.read.pdf_metadata("data/docs/**/*.pdf", recursive=True)
        ```

    Example: Read a metadata of PDFS in a folder, excluding some files.
        ```python
        df = session.read.pdf_metadata("data/docs/*.pdf", exclude=r"\.backup.pdf$")
        ```

    """
    path_str_list = validate_paths_and_return_list_of_strings(paths)

    logical_node = DocSource.from_session_state(
        paths=path_str_list,
        content_type="pdf",
        exclude=exclude,
        recursive=recursive,
        session_state=self._session_state,
    )
    from fenic.api.dataframe import DataFrame

    return DataFrame._from_logical_plan(logical_node, self._session_state)
```

## DataFrameWriter

```
DataFrameWriter(dataframe: DataFrame)
```

Interface used to write a DataFrame to external storage systems.

Similar to PySpark's DataFrameWriter.

Supported External Storage Schemes:
- Amazon S3 (s3://)
- Format: s3://{bucket_name}/{path_to_file}

```
- Notes:
    - Uses boto3 to aquire AWS credentials.

- Examples:
    - s3://my-bucket/data.csv
    - s3://my-bucket/data/*.parquet
```

- Local Files (file:// or implicit)

  - Format: file://{absolute_or_relative_path}
  - Notes:

    - Paths without a scheme (e.g., ./data.csv or /tmp/data.parquet) are treated as local files
  - Examples:
    - file:///home/user/data.csv
    - ./data/\*.parquet

Initialize a DataFrameWriter.

Parameters:

- **`dataframe`**
  (`DataFrame`)
  –

  The DataFrame to write.

Methods:

- **`csv`**
  –

  Saves the content of the DataFrame as a single CSV file with comma as the delimiter and headers in the first row.
- **`parquet`**
  –

  Saves the content of the DataFrame as a single Parquet file.
- **`save_as_table`**
  –

  Saves the content of the DataFrame as the specified table.
- **`save_as_view`**
  –

  Saves the content of the DataFrame as a view.

Source code in `src/fenic/api/io/writer.py`

```
def __init__(self, dataframe: DataFrame):
    """Initialize a DataFrameWriter.

    Args:
        dataframe: The DataFrame to write.
    """
    self._dataframe = dataframe
```

### csv

```
csv(file_path: Union[str, Path], mode: Literal['error', 'overwrite', 'ignore'] = 'overwrite') -> QueryMetrics
```

Saves the content of the DataFrame as a single CSV file with comma as the delimiter and headers in the first row.

Parameters:

- **`file_path`**
  (`Union[str, Path]`)
  –

  Path to save the CSV file to
- **`mode`**
  (`Literal['error', 'overwrite', 'ignore']`, default:
  `'overwrite'`
  )
  –

  Write mode. Default is "overwrite".
  - error: Raises an error if file exists
  - overwrite: Overwrites the file if it exists
  - ignore: Silently ignores operation if file exists

Returns:

- **`QueryMetrics`** ( `QueryMetrics`
  ) –

  The query metrics

Save with overwrite mode (default)

```
df.write.csv("output.csv")  # Overwrites if exists
```

Save with error mode

```
df.write.csv("output.csv", mode="error")  # Raises error if exists
```

Save with ignore mode

```
df.write.csv("output.csv", mode="ignore")  # Skips if exists
```

Source code in `src/fenic/api/io/writer.py`

```
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
```

### parquet

```
parquet(file_path: Union[str, Path], mode: Literal['error', 'overwrite', 'ignore'] = 'overwrite') -> QueryMetrics
```

Saves the content of the DataFrame as a single Parquet file.

Parameters:

- **`file_path`**
  (`Union[str, Path]`)
  –

  Path to save the Parquet file to
- **`mode`**
  (`Literal['error', 'overwrite', 'ignore']`, default:
  `'overwrite'`
  )
  –

  Write mode. Default is "overwrite".
  - error: Raises an error if file exists
  - overwrite: Overwrites the file if it exists
  - ignore: Silently ignores operation if file exists

Returns:

- **`QueryMetrics`** ( `QueryMetrics`
  ) –

  The query metrics

Save with overwrite mode (default)

```
df.write.parquet("output.parquet")  # Overwrites if exists
```

Save with error mode

```
df.write.parquet("output.parquet", mode="error")  # Raises error if exists
```

Save with ignore mode

```
df.write.parquet("output.parquet", mode="ignore")  # Skips if exists
```

Source code in `src/fenic/api/io/writer.py`

```
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
```

### save_as_table

```
save_as_table(table_name: str, mode: Literal['error', 'append', 'overwrite', 'ignore'] = 'error') -> QueryMetrics
```

Saves the content of the DataFrame as the specified table.

Parameters:

- **`table_name`**
  (`str`)
  –

  Name of the table to save to
- **`mode`**
  (`Literal['error', 'append', 'overwrite', 'ignore']`, default:
  `'error'`
  )
  –

  Write mode. Default is "error".
  - error: Raises an error if table exists
  - append: Appends data to table if it exists
  - overwrite: Overwrites existing table
  - ignore: Silently ignores operation if table exists

Returns:

- **`QueryMetrics`** ( `QueryMetrics`
  ) –

  The query metrics

Save with error mode (default)

```
df.write.save_as_table("my_table")  # Raises error if table exists
```

Save with append mode

```
df.write.save_as_table("my_table", mode="append")  # Adds to existing table
```

Save with overwrite mode

```
df.write.save_as_table("my_table", mode="overwrite")  # Replaces existing table
```

Source code in `src/fenic/api/io/writer.py`

```
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
```

### save_as_view

```
save_as_view(view_name: str, description: str | None = None) -> None
```

Saves the content of the DataFrame as a view.

Parameters:

- **`view_name`**
  (`str`)
  –

  Name of the view to save to
- **`description`**
  (`str | None`, default:
  `None`
  )
  –

  Optional human-readable view description to store in the catalog.

Returns:

- `None`
  –

  None.

Source code in `src/fenic/api/io/writer.py`

```
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
```
