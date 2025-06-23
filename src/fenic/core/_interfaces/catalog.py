from abc import ABC, abstractmethod
from typing import List, Tuple

from fenic.core.types import Schema


class BaseCatalog(ABC):
    # Catalog operations
    @abstractmethod
    def does_catalog_exist(self, catalog_name: str) -> bool:
        """Checks if a catalog with the specified name exists."""
        pass

    @abstractmethod
    def get_current_catalog(self) -> str:
        """Get the name of the current catalog."""
        pass

    @abstractmethod
    def set_current_catalog(self, catalog_name: str) -> None:
        """Set the current catalog."""
        pass

    @abstractmethod
    def list_catalogs(self) -> List[str]:
        """Get a list of all catalogs."""
        pass

    # Database operations
    @abstractmethod
    def does_database_exist(self, database_name: str) -> bool:
        """Checks if a database with the specified name exists in the current catalog."""
        pass

    @abstractmethod
    def get_current_database(self) -> str:
        """Get the name of the current database in the current catalog."""
        pass

    @abstractmethod
    def set_current_database(self, database_name: str) -> None:
        """Set the current database in the current catalog."""
        pass

    @abstractmethod
    def list_databases(self) -> List[str]:
        """Get a list of all databases in the current catalog."""
        pass

    @abstractmethod
    def create_database(
        self, database_name: str, ignore_if_exists: bool = True
    ) -> bool:
        """Create a new database in the current catalog."""
        pass

    @abstractmethod
    def drop_database(
        self,
        database_name: str,
        cascade: bool = False,
        ignore_if_not_exists: bool = True,
    ) -> bool:
        """Drop a database from the current catalog."""
        pass

    # Table operations
    @abstractmethod
    def does_table_exist(self, table_name: str) -> bool:
        """Checks if a table with the specified name exists in the current database."""
        pass

    @abstractmethod
    def list_tables(self) -> List[str]:
        """Get a list of all tables in the current database."""
        pass

    @abstractmethod
    def describe_table(self, table_name: str) -> Schema:
        """Get the schema of the specified table."""
        pass

    @abstractmethod
    def drop_table(self, table_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a table from the current database."""
        pass

    @abstractmethod
    def create_table(
        self, table_name: str, schema: Schema, ignore_if_exists: bool = True
    ) -> bool:
        """Create a new table in the current database."""
        pass

    @abstractmethod
    def create_view(
        self,
        view_name: str,
        schema_blob: bytes,
        view_blob: bytes,
        ignore_if_exists: bool = True,
    ) -> bool:
        """Create a new view in the current database."""
        pass

    @abstractmethod
    def drop_view(self, view_name: str, ignore_if_not_exists: bool = True) -> bool:
        """Drop a view from the current database."""
        pass

    @abstractmethod
    def describe_view(self, view_name: str) -> Tuple[object, object]:
        """Get the serialized schema and logical plan of the specified view."""
        pass

    @abstractmethod
    def list_views(self) -> List[str]:
        """Get a list of all views in the current database."""
        pass

    @abstractmethod
    def does_view_exist(self, view_name: str) -> bool:
        """Checks if a view with the specified name exists in the current database."""
        pass