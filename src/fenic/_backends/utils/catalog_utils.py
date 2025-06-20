import logging
from dataclasses import dataclass
from typing import List, Optional

from fenic.core.error import ValidationError

logger = logging.getLogger(__name__)


class QualifiedNameParser:
    """Parse strings like:
      - database names:        catalog.database  OR  database
      - table names:           catalog.database.table  OR  database.table  OR  table
    Identifiers may be quoted with double quotes to allow dots, spaces, etc:
      e.g.  "cat.1"."db name"."tbl".
    """

    @staticmethod
    def _split(name: str) -> List[str]:
        """Split on unquoted dots, strip quotes and whitespace.
        Raises ValueError on empty identifiers or unbalanced quotes.
        """
        tokens: List[str] = []
        current: List[str] = []
        in_quote = False

        for ch in name:
            if ch == '"':
                in_quote = not in_quote
                continue
            if ch == "." and not in_quote:
                part = "".join(current).strip()
                if not part:
                    raise ValueError(f"Empty identifier in '{name}'")
                tokens.append(part)
                current.clear()
            else:
                current.append(ch)

        if in_quote:
            raise ValueError(f"Unbalanced quotes in '{name}'")
        part = "".join(current).strip()
        if not part:
            raise ValueError(f"Empty identifier in '{name}'")
        tokens.append(part)

        return tokens


class BaseIdentifier:
    catalog: Optional[str] = None
    db: Optional[str] = None

    def is_catalog_name_equal(self, catalog_name: str) -> bool:
        """Check if the catalog name is equal, ignoring case."""
        if self.catalog and catalog_name:
            return compare_object_names(self.catalog, catalog_name)
        return True

    def is_db_name_equal(self, db_name: str) -> bool:
        """Check if the database name is equal, ignoring case."""
        if self.db and db_name:
            return compare_object_names(self.db, db_name)
        return True


@dataclass(frozen=True)
class TableIdentifier(BaseIdentifier):
    table: str
    db: Optional[str] = None
    catalog: Optional[str] = None

    @classmethod
    def from_string(cls, full_name: str) -> "TableIdentifier":
        """Parse a table-qualified name.
        Returns an identifer with keys: catalog, database, table (None if not provided).
        Raises ValueError if not 1–3 parts.
        """
        parts = QualifiedNameParser._split(full_name)
        if len(parts) == 1:
            return TableIdentifier(table=normalize_object_name(parts[0]))
        if len(parts) == 2:
            return TableIdentifier(
                table=normalize_object_name(parts[1]),
                db=normalize_object_name(parts[0]))
        if len(parts) == 3:
            return TableIdentifier(
                catalog=normalize_object_name(parts[0]),
                db=normalize_object_name(parts[1]),
                table=normalize_object_name(parts[2]))
        raise ValidationError(
            f"Invalid table name '{full_name}': expected 1-3 parts, got {len(parts)}"
        )

    def is_table_name_equal(self, table_name: str) -> bool:
        """Check if the table name is equal, ignoring case."""
        if self.table and table_name:
            return compare_object_names(self.table, table_name)
        return True

    def enrich(self, catalog_name: str, db_name: str) -> "TableIdentifier":
        """Enrich the table identifier with the catalog name and database name."""
        return TableIdentifier(
            catalog=self.catalog if self.catalog else catalog_name,
            db=self.db if self.db else db_name,
            table=self.table,
        )


@dataclass(frozen=True)
class DBIdentifier(BaseIdentifier):
    db: str
    catalog: Optional[str] = None

    @classmethod
    def from_string(cls, full_name: str) -> "DBIdentifier":
        """Parse a database-qualified name.
        Returns dict with keys: catalog (or None) and database.
        Raises ValueError if not 1 or 2 parts.
        """
        parts = QualifiedNameParser._split(full_name)
        if len(parts) == 1:
            return DBIdentifier(db=normalize_object_name(parts[0]))
        if len(parts) == 2:
            return DBIdentifier(
                catalog=normalize_object_name(parts[0]),
                db=normalize_object_name(parts[1]))
        raise ValidationError(
            f"Invalid database name '{full_name}': expected 1 or 2 parts, got {len(parts)}"
        )

    def enrich(self, catalog_name: str) -> "DBIdentifier":
        """Enrich the database identifier with the catalog name."""
        if self.catalog:
            return self
        return DBIdentifier(catalog=catalog_name, db=self.db)


def compare_object_names(object_name_1: str, object_name_2: str) -> bool:
    """Compare two object names, ignoring case."""
    return object_name_1.casefold() == object_name_2.casefold()

def normalize_object_name(name: str) -> str:
    """Normalize an object name, handling DuckDB's naming conventions."""
    return name.casefold()
