"""Schema definitions for DataFrame structures.

This module provides classes for defining and working with DataFrame schemas.
It includes ColumnField for individual column definitions and Schema for complete
DataFrame structure definitions.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic.dataclasses import ConfigDict, dataclass

from fenic._constants import PRETTY_PRINT_INDENT
from fenic.core.types import ArrayType, DataType, StructType


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ColumnField:
    """Represents a typed column in a DataFrame schema.

    A ColumnField defines the structure of a single column by specifying its name
    and data type. This is used as a building block for DataFrame schemas.

    Attributes:
        name: The name of the column.
        data_type: The data type of the column, as a DataType instance.
    """

    name: str
    data_type: DataType

    def __str__(self) -> str:
        """Return a string representation of the ColumnField.

        Returns:
            A string in the format "ColumnField(name='name', data_type=type)".
        """
        return f"ColumnField(name='{self.name}', data_type={self.data_type})"

    def _str_with_indent(self, indent: int = 0) -> str:
        """Return a pretty-printed string representation with indentation.

        Args:
            indent: Number of spaces to indent.

        Returns:
            A formatted string representation of the ColumnField.
        """

        def indent_datatype(data_type: DataType, current_indent: int) -> str:
            """Format a data type with proper indentation for nested structures."""
            if isinstance(data_type, ArrayType):
                spaces = PRETTY_PRINT_INDENT * current_indent
                content_spaces = PRETTY_PRINT_INDENT * (current_indent + 1)
                element_type_str = indent_datatype(data_type.element_type, current_indent + 1)
                return f"ArrayType(\n{content_spaces}element_type={element_type_str}\n{spaces})"

            elif isinstance(data_type, StructType):
                spaces = PRETTY_PRINT_INDENT * current_indent
                content_spaces = PRETTY_PRINT_INDENT * (current_indent + 1)
                field_strs = []
                for field in data_type.struct_fields:
                    field_data_type_str = indent_datatype(field.data_type, current_indent + 1)
                    field_strs.append(f"{content_spaces}StructField(name='{field.name}', data_type={field_data_type_str})")

                fields_content = "\n".join(field_strs)
                return f"StructType(\n{fields_content}\n{spaces})"

            else:
                return str(data_type)

        spaces = PRETTY_PRINT_INDENT * indent
        data_type_str = indent_datatype(self.data_type, indent)
        return f"{spaces}ColumnField(name='{self.name}', data_type={data_type_str})"


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Schema:
    """Represents the schema of a DataFrame.

    A Schema defines the structure of a DataFrame by specifying an ordered collection
    of column fields. Each column field defines the name and data type of a column
    in the DataFrame.

    Attributes:
        column_fields: An ordered list of ColumnField objects that define the
            structure of the DataFrame.
    """

    column_fields: List[ColumnField]

    def __str__(self) -> str:
        """Return a string representation of the Schema.

        Returns:
            A multi-line string with proper indentation showing the schema structure.
        """
        return self._str_with_indent(base_indent=0)

    def _str_with_indent(self, base_indent: int = 0) -> str:
        """Return a pretty-printed string with custom base indentation.

        Args:
            base_indent: Number of spaces to use as base indentation.

        Returns:
            A multi-line string with proper indentation relative to base_indent.
        """
        base_spaces = PRETTY_PRINT_INDENT * base_indent
        field_strs = []
        for field in self.column_fields:
            field_strs.append(field._str_with_indent(indent=base_indent + 1))

        fields_content = "\n".join(field_strs)
        return f"{base_spaces}Schema(\n{fields_content}\n{base_spaces})"

    def column_names(self) -> List[str]:
        """Get a list of all column names in the schema.

        Returns:
            A list of strings containing the names of all columns in the schema.
        """
        return [field.name for field in self.column_fields]


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class DatasetMetadata:
    """Metadata for a dataset (table or view).

    Attributes:
        schema: The schema of the dataset.
        description: The natural language description of the dataset's contents.
    """
    schema: Schema
    description: Optional[str]
