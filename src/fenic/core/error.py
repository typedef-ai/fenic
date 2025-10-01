"""Fenic error hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from fenic.core.types.datatypes import DataType


# Base exception
class FenicError(Exception):
    """Base exception for all fenic errors."""

    pass


# 1. Configuration Errors
class ConfigurationError(FenicError):
    """Errors during session configuration or initialization."""

    pass


class SessionError(ConfigurationError):
    """Session lifecycle errors."""

    pass


class CloudSessionError(SessionError):
    """Cloud session lifecycle errors."""

    def __init__(self, error_message: str):
        """Initialize a cloud session error.

        Args:
            error_message: The error message describing what went wrong.
        """
        super().__init__(
            f"{error_message}. " "Please file a ticket with Typedef support."
        )


# 2. Validation Errors
class ValidationError(FenicError):
    """Invalid usage of public APIs or incorrect arguments."""

    pass


class InvalidExampleCollectionError(ValidationError):
    """Exception raised when a semantic example collection is invalid."""

    pass


# 3. Plan Errors
class PlanError(FenicError):
    """Errors during logical plan construction and validation."""

    pass


class ColumnNotFoundError(PlanError):
    """Column doesn't exist."""

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


class TypeMismatchError(PlanError):
    """Type validation errors."""

    def __init__(self, expected: Union[DataType, List[DataType]], actual: DataType, context: str):
        """Initialize a type mismatch error.

        Args:
            expected: The expected data type.
            actual: The actual data type that was found.
            context: Additional context about where the type mismatch occurred.
        """
        super().__init__(f"{context}: expected {expected}, got {actual}")

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


# 4. Catalog Errors
class CatalogError(FenicError):
    """Catalog and table management errors."""

    pass


class CatalogNotFoundError(CatalogError):
    """Catalog doesn't exist."""

    def __init__(self, catalog_name: str):
        """Initialize a catalog not found error.

        Args:
            catalog_name: The name of the catalog that was not found.
        """
        super().__init__(f"Catalog '{catalog_name}' does not exist")


class CatalogAlreadyExistsError(CatalogError):
    """Catalog already exists."""

    def __init__(self, catalog_name: str):
        """Initialize a catalog already exists error.

        Args:
            catalog_name: The name of the catalog that already exists.
        """
        super().__init__(f"Catalog '{catalog_name}' already exists")


class TableNotFoundError(CatalogError):
    """Table doesn't exist."""

    def __init__(self, table_name: str, database: str):
        """Initialize a table not found error.

        Args:
            table_name: The name of the table that was not found.
            database: The name of the database containing the table.
        """
        self.table_name = table_name
        self.database = database
        super().__init__(f"Table '{database}.{table_name}' does not exist")


class TableAlreadyExistsError(CatalogError):
    """Table already exists."""

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

class ToolNotFoundError(CatalogError):
    """Tool doesn't exist."""

    def __init__(self, tool_name: str):
        """Initialize a tool not found error.

        Args:
            tool_name: The name of the tool that was not found.
        """
        super().__init__(f"Tool '{tool_name}' does not exist")

class ToolAlreadyExistsError(CatalogError):
    """Tool already exists."""

    def __init__(self, tool_name: str):
        """Initialize a tool already exists error.

        Args:
            tool_name: The name of the tool that already exists.
        """
        super().__init__(f"Tool '{tool_name}' already exists")

class DatabaseNotFoundError(CatalogError):
    """Database doesn't exist."""

    def __init__(self, database_name: str):
        """Initialize a database not found error.

        Args:
            database_name: The name of the database that was not found.
        """
        super().__init__(f"Database '{database_name}' does not exist")


class DatabaseAlreadyExistsError(CatalogError):
    """Database already exists."""

    def __init__(self, database_name: str):
        """Initialize a database already exists error.

        Args:
            database_name: The name of the database that already exists.
        """
        super().__init__(f"Database '{database_name}' already exists")


# 5. Execution Errors
class ExecutionError(FenicError):
    """Errors during physical plan execution."""

    pass


class CloudExecutionError(ExecutionError):
    """Errors during physical plan execution in a cloud session."""

    def __init__(self, error_message: str):
        """Initialize a cloud execution error.

        Args:
            error_message: The error message describing what went wrong.
        """
        super().__init__(
            f"{error_message}. " "Please file a ticket with Typedef support."
        )


# 6. Lineage Errors
class LineageError(FenicError):
    """Errors during lineage traversal."""

    pass


# 7. Internal Errors
class InternalError(FenicError):
    """Internal invariant violations."""

    pass


# 8. IO Errors
class FileLoaderError(FenicError):
    """File loader error."""

    def __init__(self, exception: Exception):
        """Initialize a file loader error.

        Args:
            exception: The exception that was raised.
        """
        super().__init__(f"File loader error: {exception}")

class UnsupportedFileTypeError(FileLoaderError):
    """Unsupported file type error."""

    def __init__(self, file_type: DataType):
        """Initialize a unsupported file type error.

        Args:
            file_type: The unsupported file type.
        """
        super().__init__(f"Unsupported file type for: {file_type}")
