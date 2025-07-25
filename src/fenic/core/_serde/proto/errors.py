"""Errors for the proto serde module."""
from __future__ import annotations

from typing import Optional, Type


class SerdeError(Exception):
    """Base exception for serialization errors."""
    pass


class DeserializationError(SerdeError):
    """Errors during deserialization."""

    def __init__(self, message: str, object_type: Optional[Type] = None, field_path: Optional[str] = None):
        """Initialize a DeserializationError."""
        self.object_type = object_type
        self.field_path = field_path
        if object_type and field_path:
            super().__init__(f"{message} at {field_path} in {object_type.__name__}")
        else:
            super().__init__(message)


class RegistrationError(SerdeError):
    """Errors during type registration."""
    pass


class SerializationError(SerdeError):
    """Errors during serialization."""

    def __init__(self, message: str, object_type: Optional[Type] = None, field_path: Optional[str] = None):
        """Initialize a SerializationError."""
        self.object_type = object_type
        self.field_path = field_path
        if object_type and field_path:
            super().__init__(f"{message} at {field_path} in {object_type.__name__}")
        else:
            super().__init__(message)
