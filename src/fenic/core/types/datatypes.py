"""Core data type definitions for the DataFrame API.

This module defines the type system used throughout the DataFrame API. It includes:
- Base classes for all data types
- Primitive types (string, integer, float, etc.)
- Composite types (arrays, structs)
- Specialized types (embeddings, markdown, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

# === Base Classes ===


class DataType(ABC):
    """Base class for all data types.

    You won't instantiate this class directly. Instead, use one of the
    concrete types like `StringType`, `ArrayType`, or `StructType`.

    Used for casting, type validation, and schema inference in the DataFrame API.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the data type.

        Returns:
            A string describing the data type.
        """
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Compare this data type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are equal, False otherwise.
        """
        pass

    def __ne__(self, other: object) -> bool:
        """Compare this data type with another object for inequality.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are not equal, False otherwise.
        """
        return not self == other

    @abstractmethod
    def __hash__(self):
        """Return a hash value for this data type.

        Returns:
            An integer hash value.
        """
        return super().__hash__()


class _PrimitiveType(DataType):
    """Marker class for all primitive type."""

    pass


class _LogicalType(DataType):
    """Marker class for all logical types."""

    pass


# === Singleton Primitive Types ===


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class _StringType(_PrimitiveType):
    def __str__(self) -> str:
        """Return a string representation of the string type.

        Returns:
            The string "StringType".
        """
        return "StringType"

    def __eq__(self, other: object) -> bool:
        """Compare this string type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a string type, False otherwise.
        """
        return isinstance(other, _StringType)


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class _IntegerType(_PrimitiveType):
    def __str__(self) -> str:
        """Return a string representation of the integer type.

        Returns:
            The string "IntegerType".
        """
        return "IntegerType"

    def __eq__(self, other: object) -> bool:
        """Compare this integer type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also an integer type, False otherwise.
        """
        return isinstance(other, _IntegerType)


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class _FloatType(_PrimitiveType):
    def __str__(self) -> str:
        """Return a string representation of the float type.

        Returns:
            The string "FloatType".
        """
        return "FloatType"

    def __eq__(self, other: object) -> bool:
        """Compare this float type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a float type, False otherwise.
        """
        return isinstance(other, _FloatType)


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class _DoubleType(_PrimitiveType):
    def __str__(self) -> str:
        """Return a string representation of the double type.

        Returns:
            The string "DoubleType".
        """
        return "DoubleType"

    def __eq__(self, other: object) -> bool:
        """Compare this double type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a double type, False otherwise.
        """
        return isinstance(other, _DoubleType)


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class _BooleanType(_PrimitiveType):
    def __str__(self) -> str:
        """Return a string representation of the boolean type.

        Returns:
            The string "BooleanType".
        """
        return "BooleanType"

    def __eq__(self, other: object) -> bool:
        """Compare this boolean type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a boolean type, False otherwise.
        """
        return isinstance(other, _BooleanType)


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class _DateType(_PrimitiveType):
    def __str__(self) -> str:
        """Return a string representation of the date type.

        Returns:
            The string "DateType".
        """
        return "DateType"

    def __eq__(self, other: object) -> bool:
        """Compare this date type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a date type, False otherwise.
        """
        return isinstance(other, _DateType)


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class _TimestampType(_PrimitiveType):
    def __str__(self) -> str:
        """Return a string representation of the timestamp type.

        Returns:
            The string "TimestampType".
        """
        return "TimestampType"

    def __eq__(self, other: object) -> bool:
        """Compare this timestamp type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a timestamp type, False otherwise.
        """
        return isinstance(other, _TimestampType)


# === Composite and Parameterized Types ===


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ArrayType(DataType):
    """A type representing a homogeneous variable-length array (list) of elements.

    Attributes:
        element_type: The data type of each element in the array.

    Example: Create an array of strings
        ```python
        ArrayType(StringType)
        ArrayType(element_type=StringType)
        ```
    """

    element_type: DataType

    def __str__(self) -> str:
        """Return a string representation of the array type.

        Returns:
            A string describing the array type and its element type.
        """
        return f"ArrayType(element_type={self.element_type})"

    def __eq__(self, other: object) -> bool:
        """Compare this array type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also an array type with the same element type,
            False otherwise.
        """
        return isinstance(other, ArrayType) and self.element_type == other.element_type


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class StructField:
    """A field in a StructType. Fields are nullable.

    Attributes:
        name: The name of the field.
        data_type: The data type of the field.
    """

    name: str
    data_type: DataType

    def __str__(self) -> str:
        """Return a string representation of the struct field.

        Returns:
            A string describing the field name and data type.
        """
        return f"StructField(name={self.name}, data_type={self.data_type})"

    def __eq__(self, other: object) -> bool:
        """Compare this struct field with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a struct field with the same name and data type,
            False otherwise.
        """
        return (
            isinstance(other, StructField)
            and self.name == other.name
            and self.data_type == other.data_type
        )


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class StructType(DataType):
    """A type representing a struct (record) with named fields.

    Attributes:
        fields: List of field definitions.

    Example: Create a struct with name and age fields
        ```python
        StructType([
            StructField("name", StringType),
            StructField("age", IntegerType),
        ])
        ```
    """

    struct_fields: List[StructField]

    def __str__(self) -> str:
        """Return a string representation of the struct type.

        Returns:
            A string describing the struct type and its fields.
        """
        return f"StructType(struct_fields=[{', '.join([str(field) for field in self.struct_fields])}])"

    def __eq__(self, other: object) -> bool:
        """Compare this struct type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a struct type with the same fields,
            False otherwise.
        """
        return (
            isinstance(other, StructType) and self.struct_fields == other.struct_fields
        )


@dataclass(frozen=True)
class EmbeddingType(_LogicalType):
    """A type representing a fixed-length embedding vector.

    Attributes:
        dimensions: The number of dimensions in the embedding vector.
        embedding_model: Name of the model used to generate the embedding.

    Example: Create an embedding type for text-embedding-3-small
        ```python
        EmbeddingType(384, embedding_model="text-embedding-3-small")
        ```
    """

    dimensions: int
    embedding_model: str

    def __str__(self) -> str:
        """Return a string representation of the embedding type.

        Returns:
            A string describing the embedding type, its dimensions, and model.
        """
        return (
            f"EmbeddingType(dimensions={self.dimensions}, model={self.embedding_model})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare this embedding type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also an embedding type with the same dimensions
            and model, False otherwise.
        """
        return (
            isinstance(other, EmbeddingType)
            and self.dimensions == other.dimensions
            and self.embedding_model == other.embedding_model
        )


# === Tagged String Types ===


@dataclass(frozen=True)
class _MarkdownType(_LogicalType):
    """Represents a markdown document."""

    def __str__(self) -> str:
        """Return a string representation of the markdown type.

        Returns:
            The string "MarkdownType".
        """
        return "MarkdownType"

    def __eq__(self, other: object) -> bool:
        """Compare this markdown type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a markdown type, False otherwise.
        """
        return isinstance(other, _MarkdownType)


@dataclass(frozen=True)
class _HtmlType(_LogicalType):
    """Represents a valid HTML document."""

    def __str__(self) -> str:
        """Return a string representation of the HTML type.

        Returns:
            The string "HtmlType".
        """
        return "HtmlType"

    def __eq__(self, other: object) -> bool:
        """Compare this HTML type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also an HTML type, False otherwise.
        """
        return isinstance(other, _HtmlType)


@dataclass(frozen=True)
class _JsonType(_LogicalType):
    """Represents a valid JSON document."""

    def __str__(self) -> str:
        """Return a string representation of the JSON type.

        Returns:
            The string "JsonType".
        """
        return "JsonType"

    def __eq__(self, other: object) -> bool:
        """Compare this JSON type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a JSON type, False otherwise.
        """
        return isinstance(other, _JsonType)


@dataclass(frozen=True)
class TranscriptType(_LogicalType):
    """Represents a string containing a transcript in a specific format."""

    format: Literal["generic", "srt", "webvtt"]

    def __str__(self) -> str:
        """Return a string representation of the transcript type.

        Returns:
            A string describing the transcript type and its format.
        """
        return f"TranscriptType(format='{self.format}')"

    def __eq__(self, other: object) -> bool:
        """Compare this transcript type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a transcript type with the same format,
            False otherwise.
        """
        return isinstance(other, TranscriptType) and self.format == other.format


@dataclass(frozen=True)
class DocumentPathType(_LogicalType):
    """Represents a string containing a a document's local (file system) or remote (URL) path."""

    format: Literal["pdf"] = "pdf"

    def __str__(self) -> str:
        """Return a string representation of the document path type.

        Returns:
            A string describing the document path type and its format.
        """
        return f"DocumentPathType(format='{self.format}')"

    def __eq__(self, other: object) -> bool:
        """Compare this document path type with another object for equality.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is also a document path type with the same format,
            False otherwise.
        """
        return isinstance(other, DocumentPathType) and self.format == other.format


def _is_dtype_numeric(dtype: DataType) -> bool:
    """Check if a data type is a numeric type."""
    return dtype in (IntegerType, FloatType, DoubleType)


def _is_logical_type(type: DataType) -> bool:
    """Check if a type is a logical type.  If type is a struct or array that contains a LogicalType, it is logical type."""
    if isinstance(type, _LogicalType):
        return True
    elif isinstance(type, StructType):
        return any(_is_logical_type(field.data_type) for field in type.struct_fields)
    elif isinstance(type, ArrayType):
        return _is_logical_type(type.element_type)
    return False


# === Instances of Singleton Types ===
StringType = _StringType()
"""Represents a UTF-8 encoded string value."""

IntegerType = _IntegerType()
"""Represents a signed integer value."""

FloatType = _FloatType()
"""Represents a 32-bit floating-point number."""

DoubleType = _DoubleType()
"""Represents a 64-bit floating-point number."""

BooleanType = _BooleanType()
"""Represents a boolean value. (True/False)"""

DateType = _DateType()
"""Represents a date value."""

TimestampType = _TimestampType()
"""Represents a timestamp value."""

MarkdownType = _MarkdownType()
"""Represents a string containing Markdown-formatted text."""

HtmlType = _HtmlType()
"""Represents a string containing raw HTML markup."""

JsonType = _JsonType()
"""Represents a string containing JSON data."""
