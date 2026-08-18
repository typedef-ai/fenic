# fenic.core.error

Canonical HTML: https://docs.fenic.ai/latest/reference/fenic/core/error/

Fenic error hierarchy.

Classes:

- **`CatalogAlreadyExistsError`**
  –

  Catalog already exists.
- **`CatalogError`**
  –

  Catalog and table management errors.
- **`CatalogNotFoundError`**
  –

  Catalog doesn't exist.
- **`CloudExecutionError`**
  –

  Errors during physical plan execution in a cloud session.
- **`CloudSessionError`**
  –

  Cloud session lifecycle errors.
- **`ColumnNotFoundError`**
  –

  Column doesn't exist.
- **`ConfigurationError`**
  –

  Errors during session configuration or initialization.
- **`DatabaseAlreadyExistsError`**
  –

  Database already exists.
- **`DatabaseNotFoundError`**
  –

  Database doesn't exist.
- **`ExecutionError`**
  –

  Errors during physical plan execution.
- **`FenicError`**
  –

  Base exception for all fenic errors.
- **`FileLoaderError`**
  –

  File loader error.
- **`InternalError`**
  –

  Internal invariant violations.
- **`InvalidExampleCollectionError`**
  –

  Exception raised when a semantic example collection is invalid.
- **`LineageError`**
  –

  Errors during lineage traversal.
- **`PlanError`**
  –

  Errors during logical plan construction and validation.
- **`SessionError`**
  –

  Session lifecycle errors.
- **`TableAlreadyExistsError`**
  –

  Table already exists.
- **`TableNotFoundError`**
  –

  Table doesn't exist.
- **`ToolAlreadyExistsError`**
  –

  Tool already exists.
- **`ToolNotFoundError`**
  –

  Tool doesn't exist.
- **`TypeMismatchError`**
  –

  Type validation errors.
- **`UnsupportedFileTypeError`**
  –

  Unsupported file type error.
- **`ValidationError`**
  –

  Invalid usage of public APIs or incorrect arguments.

## CatalogAlreadyExistsError

```
CatalogAlreadyExistsError(catalog_name: str)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.CatalogAlreadyExistsError[CatalogAlreadyExistsError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.CatalogAlreadyExistsError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.CatalogAlreadyExistsError href "" "fenic.core.error.CatalogAlreadyExistsError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Catalog already exists.

Initialize a catalog already exists error.

Parameters:

- **`catalog_name`**
  (`str`)
  –

  The name of the catalog that already exists.

Source code in `src/fenic/core/error.py`

```
def __init__(self, catalog_name: str):
    """Initialize a catalog already exists error.

    Args:
        catalog_name: The name of the catalog that already exists.
    """
    super().__init__(f"Catalog '{catalog_name}' already exists")
```

## CatalogError

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Catalog and table management errors.

## CatalogNotFoundError

```
CatalogNotFoundError(catalog_name: str)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.CatalogNotFoundError[CatalogNotFoundError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.CatalogNotFoundError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.CatalogNotFoundError href "" "fenic.core.error.CatalogNotFoundError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Catalog doesn't exist.

Initialize a catalog not found error.

Parameters:

- **`catalog_name`**
  (`str`)
  –

  The name of the catalog that was not found.

Source code in `src/fenic/core/error.py`

```
def __init__(self, catalog_name: str):
    """Initialize a catalog not found error.

    Args:
        catalog_name: The name of the catalog that was not found.
    """
    super().__init__(f"Catalog '{catalog_name}' does not exist")
```

## CloudExecutionError

```
CloudExecutionError(error_message: str)
```

Bases: `ExecutionError`

```
              flowchart TD
              fenic.core.error.CloudExecutionError[CloudExecutionError]
              fenic.core.error.ExecutionError[ExecutionError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.ExecutionError --> fenic.core.error.CloudExecutionError
                                fenic.core.error.FenicError --> fenic.core.error.ExecutionError

              click fenic.core.error.CloudExecutionError href "" "fenic.core.error.CloudExecutionError"
              click fenic.core.error.ExecutionError href "" "fenic.core.error.ExecutionError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Errors during physical plan execution in a cloud session.

Initialize a cloud execution error.

Parameters:

- **`error_message`**
  (`str`)
  –

  The error message describing what went wrong.

Source code in `src/fenic/core/error.py`

```
def __init__(self, error_message: str):
    """Initialize a cloud execution error.

    Args:
        error_message: The error message describing what went wrong.
    """
    super().__init__(
        f"{error_message}. " "Please file a ticket with Typedef support."
    )
```

## CloudSessionError

```
CloudSessionError(error_message: str)
```

Bases: `SessionError`

```
              flowchart TD
              fenic.core.error.CloudSessionError[CloudSessionError]
              fenic.core.error.SessionError[SessionError]
              fenic.core.error.ConfigurationError[ConfigurationError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.SessionError --> fenic.core.error.CloudSessionError
                                fenic.core.error.ConfigurationError --> fenic.core.error.SessionError
                                fenic.core.error.FenicError --> fenic.core.error.ConfigurationError

              click fenic.core.error.CloudSessionError href "" "fenic.core.error.CloudSessionError"
              click fenic.core.error.SessionError href "" "fenic.core.error.SessionError"
              click fenic.core.error.ConfigurationError href "" "fenic.core.error.ConfigurationError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Cloud session lifecycle errors.

Initialize a cloud session error.

Parameters:

- **`error_message`**
  (`str`)
  –

  The error message describing what went wrong.

Source code in `src/fenic/core/error.py`

```
def __init__(self, error_message: str):
    """Initialize a cloud session error.

    Args:
        error_message: The error message describing what went wrong.
    """
    super().__init__(
        f"{error_message}. " "Please file a ticket with Typedef support."
    )
```

## ColumnNotFoundError

```
ColumnNotFoundError(column_name: str, available_columns: List[str])
```

Bases: `PlanError`

```
              flowchart TD
              fenic.core.error.ColumnNotFoundError[ColumnNotFoundError]
              fenic.core.error.PlanError[PlanError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.PlanError --> fenic.core.error.ColumnNotFoundError
                                fenic.core.error.FenicError --> fenic.core.error.PlanError

              click fenic.core.error.ColumnNotFoundError href "" "fenic.core.error.ColumnNotFoundError"
              click fenic.core.error.PlanError href "" "fenic.core.error.PlanError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Column doesn't exist.

Initialize a column not found error.

Parameters:

- **`column_name`**
  (`str`)
  –

  The name of the column that was not found.
- **`available_columns`**
  (`List[str]`)
  –

  List of column names that are available.

Source code in `src/fenic/core/error.py`

```
def __init__(self, column_name: str, available_columns: List[str]):
    """Initialize a column not found error.

    Args:
        column_name: The name of the column that was not found.
        available_columns: List of column names that are available.
    """
    super().__init__(
        f"Column '{column_name}' not found. "
        f"Available columns: {', '.join(sorted(available_columns))}"
    )
```

## ConfigurationError

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.ConfigurationError[ConfigurationError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.ConfigurationError

              click fenic.core.error.ConfigurationError href "" "fenic.core.error.ConfigurationError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Errors during session configuration or initialization.

## DatabaseAlreadyExistsError

```
DatabaseAlreadyExistsError(database_name: str)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.DatabaseAlreadyExistsError[DatabaseAlreadyExistsError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.DatabaseAlreadyExistsError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.DatabaseAlreadyExistsError href "" "fenic.core.error.DatabaseAlreadyExistsError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Database already exists.

Initialize a database already exists error.

Parameters:

- **`database_name`**
  (`str`)
  –

  The name of the database that already exists.

Source code in `src/fenic/core/error.py`

```
def __init__(self, database_name: str):
    """Initialize a database already exists error.

    Args:
        database_name: The name of the database that already exists.
    """
    super().__init__(f"Database '{database_name}' already exists")
```

## DatabaseNotFoundError

```
DatabaseNotFoundError(database_name: str)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.DatabaseNotFoundError[DatabaseNotFoundError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.DatabaseNotFoundError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.DatabaseNotFoundError href "" "fenic.core.error.DatabaseNotFoundError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Database doesn't exist.

Initialize a database not found error.

Parameters:

- **`database_name`**
  (`str`)
  –

  The name of the database that was not found.

Source code in `src/fenic/core/error.py`

```
def __init__(self, database_name: str):
    """Initialize a database not found error.

    Args:
        database_name: The name of the database that was not found.
    """
    super().__init__(f"Database '{database_name}' does not exist")
```

## ExecutionError

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.ExecutionError[ExecutionError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.ExecutionError

              click fenic.core.error.ExecutionError href "" "fenic.core.error.ExecutionError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Errors during physical plan execution.

## FenicError

Bases: `Exception`

```
              flowchart TD
              fenic.core.error.FenicError[FenicError]

              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Base exception for all fenic errors.

## FileLoaderError

```
FileLoaderError(exception: Exception)
```

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.FileLoaderError[FileLoaderError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.FileLoaderError

              click fenic.core.error.FileLoaderError href "" "fenic.core.error.FileLoaderError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

File loader error.

Initialize a file loader error.

Parameters:

- **`exception`**
  (`Exception`)
  –

  The exception that was raised.

Source code in `src/fenic/core/error.py`

```
def __init__(self, exception: Exception):
    """Initialize a file loader error.

    Args:
        exception: The exception that was raised.
    """
    super().__init__(f"File loader error: {exception}")
```

## InternalError

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.InternalError[InternalError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.InternalError

              click fenic.core.error.InternalError href "" "fenic.core.error.InternalError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Internal invariant violations.

## InvalidExampleCollectionError

Bases: `ValidationError`

```
              flowchart TD
              fenic.core.error.InvalidExampleCollectionError[InvalidExampleCollectionError]
              fenic.core.error.ValidationError[ValidationError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.ValidationError --> fenic.core.error.InvalidExampleCollectionError
                                fenic.core.error.FenicError --> fenic.core.error.ValidationError

              click fenic.core.error.InvalidExampleCollectionError href "" "fenic.core.error.InvalidExampleCollectionError"
              click fenic.core.error.ValidationError href "" "fenic.core.error.ValidationError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Exception raised when a semantic example collection is invalid.

## LineageError

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.LineageError[LineageError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.LineageError

              click fenic.core.error.LineageError href "" "fenic.core.error.LineageError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Errors during lineage traversal.

## PlanError

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.PlanError[PlanError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.PlanError

              click fenic.core.error.PlanError href "" "fenic.core.error.PlanError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Errors during logical plan construction and validation.

## SessionError

Bases: `ConfigurationError`

```
              flowchart TD
              fenic.core.error.SessionError[SessionError]
              fenic.core.error.ConfigurationError[ConfigurationError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.ConfigurationError --> fenic.core.error.SessionError
                                fenic.core.error.FenicError --> fenic.core.error.ConfigurationError

              click fenic.core.error.SessionError href "" "fenic.core.error.SessionError"
              click fenic.core.error.ConfigurationError href "" "fenic.core.error.ConfigurationError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Session lifecycle errors.

## TableAlreadyExistsError

```
TableAlreadyExistsError(table_name: str, database: Optional[str] = None)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.TableAlreadyExistsError[TableAlreadyExistsError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.TableAlreadyExistsError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.TableAlreadyExistsError href "" "fenic.core.error.TableAlreadyExistsError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Table already exists.

Initialize a table already exists error.

Parameters:

- **`table_name`**
  (`str`)
  –

  The name of the table that already exists.
- **`database`**
  (`Optional[str]`, default:
  `None`
  )
  –

  Optional name of the database containing the table.

Source code in `src/fenic/core/error.py`

```
def __init__(self, table_name: str, database: Optional[str] = None):
    """Initialize a table already exists error.

    Args:
        table_name: The name of the table that already exists.
        database: Optional name of the database containing the table.
    """
    if database:
        table_ref = f"{database}.{table_name}"
    else:
        table_ref = table_name
    super().__init__(
        f"Table '{table_ref}' already exists. "
        f"Use mode='overwrite' to replace the existing table."
    )
```

## TableNotFoundError

```
TableNotFoundError(table_name: str, database: str)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.TableNotFoundError[TableNotFoundError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.TableNotFoundError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.TableNotFoundError href "" "fenic.core.error.TableNotFoundError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Table doesn't exist.

Initialize a table not found error.

Parameters:

- **`table_name`**
  (`str`)
  –

  The name of the table that was not found.
- **`database`**
  (`str`)
  –

  The name of the database containing the table.

Source code in `src/fenic/core/error.py`

```
def __init__(self, table_name: str, database: str):
    """Initialize a table not found error.

    Args:
        table_name: The name of the table that was not found.
        database: The name of the database containing the table.
    """
    self.table_name = table_name
    self.database = database
    super().__init__(f"Table '{database}.{table_name}' does not exist")
```

## ToolAlreadyExistsError

```
ToolAlreadyExistsError(tool_name: str)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.ToolAlreadyExistsError[ToolAlreadyExistsError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.ToolAlreadyExistsError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.ToolAlreadyExistsError href "" "fenic.core.error.ToolAlreadyExistsError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Tool already exists.

Initialize a tool already exists error.

Parameters:

- **`tool_name`**
  (`str`)
  –

  The name of the tool that already exists.

Source code in `src/fenic/core/error.py`

```
def __init__(self, tool_name: str):
    """Initialize a tool already exists error.

    Args:
        tool_name: The name of the tool that already exists.
    """
    super().__init__(f"Tool '{tool_name}' already exists")
```

## ToolNotFoundError

```
ToolNotFoundError(tool_name: str)
```

Bases: `CatalogError`

```
              flowchart TD
              fenic.core.error.ToolNotFoundError[ToolNotFoundError]
              fenic.core.error.CatalogError[CatalogError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.CatalogError --> fenic.core.error.ToolNotFoundError
                                fenic.core.error.FenicError --> fenic.core.error.CatalogError

              click fenic.core.error.ToolNotFoundError href "" "fenic.core.error.ToolNotFoundError"
              click fenic.core.error.CatalogError href "" "fenic.core.error.CatalogError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Tool doesn't exist.

Initialize a tool not found error.

Parameters:

- **`tool_name`**
  (`str`)
  –

  The name of the tool that was not found.

Source code in `src/fenic/core/error.py`

```
def __init__(self, tool_name: str):
    """Initialize a tool not found error.

    Args:
        tool_name: The name of the tool that was not found.
    """
    super().__init__(f"Tool '{tool_name}' does not exist")
```

## TypeMismatchError

```
TypeMismatchError(expected: Union[DataType, List[DataType]], actual: DataType, context: str)
```

Bases: `PlanError`

```
              flowchart TD
              fenic.core.error.TypeMismatchError[TypeMismatchError]
              fenic.core.error.PlanError[PlanError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.PlanError --> fenic.core.error.TypeMismatchError
                                fenic.core.error.FenicError --> fenic.core.error.PlanError

              click fenic.core.error.TypeMismatchError href "" "fenic.core.error.TypeMismatchError"
              click fenic.core.error.PlanError href "" "fenic.core.error.PlanError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Type validation errors.

Initialize a type mismatch error.

Parameters:

- **`expected`**
  (`Union[DataType, List[DataType]]`)
  –

  The expected data type.
- **`actual`**
  (`DataType`)
  –

  The actual data type that was found.
- **`context`**
  (`str`)
  –

  Additional context about where the type mismatch occurred.

Methods:

- **`from_message`**
  –

  Create a TypeMismatchError from a message string.

Source code in `src/fenic/core/error.py`

```
def __init__(self, expected: Union[DataType, List[DataType]], actual: DataType, context: str):
    """Initialize a type mismatch error.

    Args:
        expected: The expected data type.
        actual: The actual data type that was found.
        context: Additional context about where the type mismatch occurred.
    """
    super().__init__(f"{context}: expected {expected}, got {actual}")
```

### from_message

```
from_message(msg: str) -> TypeMismatchError
```

Create a TypeMismatchError from a message string.

Parameters:

- **`msg`**
  (`str`)
  –

  The error message.

Returns:

- `TypeMismatchError`
  –

  A new TypeMismatchError instance with the given message.

Source code in `src/fenic/core/error.py`

```
@classmethod
def from_message(cls, msg: str) -> TypeMismatchError:
    """Create a TypeMismatchError from a message string.

    Args:
        msg: The error message.

    Returns:
        A new TypeMismatchError instance with the given message.
    """
    instance = cls.__new__(cls)  # Bypass __init__
    super(TypeMismatchError, instance).__init__(msg)
    return instance
```

## UnsupportedFileTypeError

```
UnsupportedFileTypeError(file_type: DataType)
```

Bases: `FileLoaderError`

```
              flowchart TD
              fenic.core.error.UnsupportedFileTypeError[UnsupportedFileTypeError]
              fenic.core.error.FileLoaderError[FileLoaderError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FileLoaderError --> fenic.core.error.UnsupportedFileTypeError
                                fenic.core.error.FenicError --> fenic.core.error.FileLoaderError

              click fenic.core.error.UnsupportedFileTypeError href "" "fenic.core.error.UnsupportedFileTypeError"
              click fenic.core.error.FileLoaderError href "" "fenic.core.error.FileLoaderError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Unsupported file type error.

Initialize a unsupported file type error.

Parameters:

- **`file_type`**
  (`DataType`)
  –

  The unsupported file type.

Source code in `src/fenic/core/error.py`

```
def __init__(self, file_type: DataType):
    """Initialize a unsupported file type error.

    Args:
        file_type: The unsupported file type.
    """
    super().__init__(f"Unsupported file type for: {file_type}")
```

## ValidationError

Bases: `FenicError`

```
              flowchart TD
              fenic.core.error.ValidationError[ValidationError]
              fenic.core.error.FenicError[FenicError]

                              fenic.core.error.FenicError --> fenic.core.error.ValidationError

              click fenic.core.error.ValidationError href "" "fenic.core.error.ValidationError"
              click fenic.core.error.FenicError href "" "fenic.core.error.FenicError"
```

Invalid usage of public APIs or incorrect arguments.
