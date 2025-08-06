"""Errors for the proto serde module."""

from __future__ import annotations

from typing import Optional, Type


class SerdeError(Exception):
    """Base exception for serialization/deserialization errors.

    All serde-specific exceptions inherit from this class. Provides
    consistent error handling with optional path and type information.
    """

    pass


class DeserializationError(SerdeError):
    """Errors during deserialization from protobuf format."""

    def __init__(
        self,
        message: str,
        object_type: Optional[Type] = None,
        field_path: Optional[str] = None,
    ):
        """Initialize a DeserializationError.

        Args:
            message: The error message.
            object_type: Optional type information for the error.
            field_path: Optional field path where the error occurred.
        """
        self.object_type = object_type
        self.field_path = field_path
        if object_type and field_path:
            super().__init__(f"{message} at {field_path} in {object_type.__name__}")
        else:
            super().__init__(message)


class SerializationError(SerdeError):
    """Errors during serialization to protobuf format."""

    def __init__(
        self,
        message: str,
        object_type: Optional[Type] = None,
        field_path: Optional[str] = None,
    ):
        """Initialize a SerializationError.

        Args:
            message: The error message.
            object_type: Optional type information for the error.
            field_path: Optional field path where the error occurred.
        """
        self.object_type = object_type
        self.field_path = field_path
        if object_type and field_path:
            super().__init__(f"{message} at {field_path} in {object_type.__name__}")
        else:
            super().__init__(message)

class UnsupportedTypeError(SerdeError):
    """Errors for unsupported types."""

    def __init__(
        self,
        reason: str,
        object_type: Optional[Type] = None,
        field_path: Optional[str] = None,
    ):
        """Initialize an UnsupportedType error.

        Args:
            reason: The reason why the type is unsupported.
            object_type: The unsupported type.
            field_path: Optional field path where the error occurred.
        """
        self.object_type = object_type
        self.field_path = field_path
        if object_type and field_path:
            super().__init__(f"{object_type.__name__} at {field_path} is not supported: {reason}")
        else:
            super().__init__(reason)