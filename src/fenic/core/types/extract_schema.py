"""Type definitions for semantic extraction schemas."""
from __future__ import annotations

from typing import List, Union

from pydantic.dataclasses import ConfigDict, dataclass

from fenic.core.types.datatypes import DataType, _PrimitiveType


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ExtractSchemaList:
    """Represents a list data type for structured extraction schema definitions.

    A schema list contains elements of a specific data type and is used for defining
    array-like structures in structured extraction schemas.
    """

    element_type: Union[DataType, ExtractSchema]

    def __init__(
        self,
        element_type: Union[DataType, ExtractSchema],
    ):
        """Initialize an ExtractSchemaList.

        Args:
            element_type: The data type of elements in the list. Must be either a primitive
                DataType or another ExtractSchema.

        Raises:
            ValueError: If element_type is a non-primitive DataType.
        """
        if isinstance(element_type, DataType) and not isinstance(
            element_type, _PrimitiveType
        ):
            raise ValueError(
                f"Invalid element type: {element_type}. Only primitive types are supported directly. "
                f"For complex types, please use ExtractSchemaList or ExtractSchema instead."
            )
        self.element_type = element_type

    def __str__(self) -> str:
        """Return a string representation of the ExtractSchemaList.

        Returns:
            A string in the format "ExtractSchemaList(element_type)".
        """
        return f"ExtractSchemaList({self.element_type})"


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ExtractSchemaField:
    """Represents a field within an structured extraction schema.

    An extract schema field has a name, a data type, and a required description that explains
    what information should be extracted into this field.
    """

    name: str
    data_type: Union[DataType, ExtractSchemaList, ExtractSchema]
    description: str

    def __init__(
        self,
        name: str,
        data_type: Union[DataType, ExtractSchemaList, ExtractSchema],
        description: str,
    ):
        """Initialize an ExtractSchemaField.

        Args:
            name: The name of the field.
            data_type: The data type of the field. Must be either a primitive DataType,
                ExtractSchemaList, or ExtractSchema.
            description: A description of what information should be extracted into this field.

        Raises:
            ValueError: If data_type is a non-primitive DataType.
        """
        self.name = name
        if isinstance(data_type, DataType) and not isinstance(
            data_type, _PrimitiveType
        ):
            raise ValueError(
                f"Invalid data type: {data_type}. Only primitive types are supported directly. "
                f"For complex types, please use ExtractSchemaList or ExtractSchema instead."
            )
        self.data_type = data_type
        self.description = description

    def __str__(self) -> str:
        """Return a string representation of the ExtractSchemaField.

        Returns:
            A string in the format "ExtractSchemaField(name, data_type, description)".
        """
        return (
            f"ExtractSchemaField({self.name}, {self.data_type}, {self.description!r})"
        )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ExtractSchema:
    """Represents a structured extraction schema.

    An extract schema contains a collection of named fields with descriptions that define
    what information should be extracted into each field.
    """

    struct_fields: List[ExtractSchemaField]

    def __str__(self) -> str:
        """Return a string representation of the ExtractSchema.

        Returns:
            A string containing a comma-separated list of field representations.
        """
        return (
            f"ExtractSchema({', '.join([str(field) for field in self.struct_fields])})"
        )

    def field_names(self) -> List[str]:
        """Get a list of all field names in the schema.

        Returns:
            A list of strings containing the names of all fields in the schema.
        """
        return [field.name for field in self.struct_fields]
