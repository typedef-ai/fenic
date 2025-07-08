"""Schema definitions for DataFrame structures.

This module provides classes for defining and working with DataFrame schemas.
It includes ColumnField for individual column definitions and Schema for complete
DataFrame structure definitions.
"""

from typing import List

from pydantic.dataclasses import ConfigDict, dataclass

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

    def _pretty_str(self, indent: int = 0) -> str:
        """Return a pretty-printed string representation with indentation.

        Args:
            indent: Number of spaces to indent.

        Returns:
            A formatted string representation of the ColumnField.
        """
        spaces = " " * indent
        data_type_str = self._format_data_type(self.data_type, indent)
        return f"{spaces}ColumnField(name='{self.name}', data_type={data_type_str})"

    def _format_data_type(self, data_type: DataType, indent: int) -> str:
        """Format a data type with proper indentation for nested structures.

        Args:
            data_type: The data type to format.
            indent: Current indentation level.

        Returns:
            A formatted string representation of the data type.
        """
        if isinstance(data_type, ArrayType):
            spaces = " " * indent
            content_spaces = " " * (indent + 2)
            element_type_str = self._format_data_type(data_type.element_type, indent + 2)
            return f"ArrayType(\n{content_spaces}element_type={element_type_str}\n{spaces})"

        elif isinstance(data_type, StructType):
            spaces = " " * indent
            content_spaces = " " * (indent + 2)
            field_strs = []
            for field in data_type.struct_fields:
                field_data_type_str = self._format_data_type(field.data_type, indent + 2)
                field_strs.append(f"{content_spaces}StructField(name='{field.name}', data_type={field_data_type_str})")

            fields_content = "\n".join(field_strs)
            return f"StructType(\n{fields_content}\n{spaces})"

        else:
            # For primitive types, just return their string representation
            return str(data_type)


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
            A string containing a comma-separated list of column field representations.
        """
        field_strs = []
        for field in self.column_fields:
            field_strs.append(field._pretty_str(indent=2))

        fields_content = "\n".join(field_strs)
        return f"Schema(\n{fields_content}\n)"

    def _inline_str(self) -> str:
        """Return a single line string representation of the Schema."""
        return f"schema=[{', '.join([str(field) for field in self.column_fields])}]"

    def column_names(self) -> List[str]:
        """Get a list of all column names in the schema.

        Returns:
            A list of strings containing the names of all columns in the schema.
        """
        return [field.name for field in self.column_fields]
