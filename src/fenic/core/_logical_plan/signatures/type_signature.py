"""Simplified type signature classes for function validation.

This module provides a streamlined TypeSignature hierarchy focused solely on
validating LogicalExpr arguments with standard DataTypes.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from fenic.core.error import InternalError, TypeMismatchError, ValidationError
from fenic.core.types.datatypes import (
    ArrayType,
    DataType,
    StructType,
    _is_dtype_numeric,
    _PrimitiveType,
)


class TypeSignature(ABC):
    """Base class for type signatures."""

    @abstractmethod
    def validate(self, arg_types: List[DataType], func_name: str) -> None:
        """Validate that argument types match this signature."""
        pass

class Exact(TypeSignature):
    """Exact argument types for functions (e.g., length(str) -> int)."""

    def __init__(self, expected_arg_types: List[DataType]):
        self.expected_arg_types = expected_arg_types

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != len(self.expected_arg_types):
            raise InternalError(
                f"{func_name} expects {len(self.expected_arg_types)} arguments, "
                f"got {len(actual_arg_types)}"
            )

        for i, (expected, actual) in enumerate(zip(self.expected_arg_types, actual_arg_types, strict=False)):
            if actual != expected:
                raise TypeMismatchError(
                    expected=expected,
                    actual=actual,
                    context=f"{func_name} Argument {i}",
                )

class Any(TypeSignature):
    """All arguments can be of any type, but an exact number of arguments is required."""

    def __init__(self, expected_num_args: int):
        self.expected_num_args = expected_num_args

    def validate(self, arg_types: List[DataType], func_name: str) -> None:
        if len(arg_types) != self.expected_num_args:
            raise ValidationError(
                f"{func_name} expects {self.expected_num_args} arguments, "
                f"got {len(arg_types)}"
            )

class Uniform(TypeSignature):
    """All arguments must be the same type."""

    def __init__(self, expected_num_args: int, required_type: Optional[DataType] = None):
        self.expected_num_args = expected_num_args
        self.required_type = required_type

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != self.expected_num_args:
            raise InternalError(
                f"{func_name} expects {self.expected_num_args} arguments, "
                f"got {len(actual_arg_types)}"
            )

        if not actual_arg_types:
            return

        first_type = actual_arg_types[0]
        if self.required_type and first_type != self.required_type:
            raise TypeMismatchError(
                expected=self.required_type,
                actual=first_type,
                context=f"{func_name} Argument 0",
            )

        for i, actual_arg_type in enumerate(actual_arg_types[1:], 1):
            if actual_arg_type != first_type:
                raise TypeMismatchError.from_message(
                    f"{func_name} expects all arguments to have the same type. "
                    f"Argument 0 has type {first_type}, but argument {i} has type {actual_arg_type}"
                )


class VariadicUniform(TypeSignature):
    """Variable number of arguments of the same type (e.g., semantic.map with multiple string columns)."""

    def __init__(self, expected_min_args: int = 0, required_type: Optional[DataType] = None):
        self.expected_min_args = expected_min_args
        self.required_type = required_type

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) < self.expected_min_args:
            raise ValidationError(
                f"{func_name} expects at least {self.expected_min_args} arguments, "
                f"got {len(actual_arg_types)}"
            )

        if not actual_arg_types:
            return

        first_type = actual_arg_types[0]
        if self.required_type and first_type != self.required_type:
            raise TypeMismatchError(expected=self.required_type, actual=first_type, context=f"{func_name} Argument 0")

        for i, actual_arg_type in enumerate(actual_arg_types[1:], 1):
            if actual_arg_type != first_type:
                raise TypeMismatchError.from_message(
                    f"{func_name} expects all arguments to have the same type. "
                    f"Argument 0 has type {first_type}, but argument {i} has type {actual_arg_type}"
                )



class VariadicAny(TypeSignature):
    """Variable number of arguments of any types (e.g., struct(T1, T2, ...) -> struct)."""

    def __init__(self, expected_min_args: int = 0):
        self.expected_min_args = expected_min_args

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) < self.expected_min_args:
            raise ValidationError(
                f"{func_name} expects at least {self.expected_min_args} arguments, "
                f"got {len(actual_arg_types)}"
            )


class Numeric(TypeSignature):
    """Arguments must be numeric types (IntegerType, FloatType, or DoubleType)."""

    def __init__(self, expected_num_args: int = 1):
        self.expected_num_args = expected_num_args

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != self.expected_num_args:
            raise InternalError(
                f"{func_name} expects {self.expected_num_args} arguments, "
                f"got {len(actual_arg_types)}"
            )

        for i, actual_arg_type in enumerate(actual_arg_types):
            if not _is_dtype_numeric(actual_arg_type):
                raise TypeMismatchError.from_message(
                    f"{func_name} expects numeric type for argument {i}, "
                    f"got {actual_arg_type}"
                )


class OneOf(TypeSignature):
    """Function supports multiple signatures."""

    def __init__(self, alternative_signatures: List[TypeSignature]):
        self.alternative_signatures = alternative_signatures

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        errors = []
        for signature in self.alternative_signatures:
            try:
                signature.validate(actual_arg_types, func_name)
                return  # Valid signature found
            except Exception as e:
                errors.append(str(e))

        # No valid signature found
        raise TypeMismatchError.from_message(
            f"{func_name} does not match any valid signature:\n" +
            "\n".join(f"  - {error}" for error in errors)
        )


# === Specialized Type Signatures for Arrays and Structs ===

class AnyExcludingStructs(TypeSignature):
    """Matches any type except the excluded types."""
    def __init__(self, expected_num_args: int = 1):
        self.expected_num_args = expected_num_args

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != self.expected_num_args:
            raise ValidationError(f"{func_name} expects {self.expected_num_args} arguments, got {len(actual_arg_types)}")
        for i, actual_arg_type in enumerate(actual_arg_types):
            if isinstance(actual_arg_type, ArrayType):
                if isinstance(actual_arg_type.element_type, StructType):
                    raise TypeMismatchError.from_message(
                        f"{func_name} expects an array of non-structs at argument {i}"
                    )
            elif isinstance(actual_arg_type, StructType) :
                raise TypeMismatchError.from_message(
                    f"{func_name} expects a non-struct type at argument {i}"
                )

class ArrayOfAny(TypeSignature):
    """Matches any ArrayType regardless of element type."""

    def __init__(self, expected_num_args: int = 1):
        self.expected_num_args = expected_num_args

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != self.expected_num_args:
            raise ValidationError(
                f"{func_name} expects {self.expected_num_args} arguments, "
                f"got {len(actual_arg_types)}"
            )

        for i, actual_arg_type in enumerate(actual_arg_types):
            if not isinstance(actual_arg_type, ArrayType):
                raise TypeMismatchError.from_message(
                    f"{func_name} expects argument {i} to be an array type, "
                    f"got {actual_arg_type}"
                )

class ArrayOfPrimitives(TypeSignature):
    """Validates array of primitive types."""

    def __init__(self, expected_num_args: int = 1):
        self.expected_num_args = expected_num_args

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != self.expected_num_args:
            raise ValidationError(
                f"{func_name} expects {self.expected_num_args} arguments, got {len(actual_arg_types)}"
            )

        for i, actual_arg_type in enumerate(actual_arg_types):
            if not isinstance(actual_arg_type, ArrayType):
                raise TypeMismatchError.from_message(
                    f"{func_name} expects argument {i} to be an array type, "
                    f"got {actual_arg_type}"
                )

            if not isinstance(actual_arg_type.element_type, _PrimitiveType):
                raise TypeMismatchError.from_message(
                    f"{func_name} expects argument {i} to be an array of primitive types, "
                    f"got {actual_arg_type.element_type}"
                )


class ArrayWithMatchingElement(TypeSignature):
    """Validates array + element where element type must match array element type."""

    def __init__(self):
        self.arg_names = ["array", "element"]

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != 2:
            arg_names_str = f" ({', '.join(self.arg_names)})"
            raise ValidationError(
                f"{func_name} expects 2 arguments{arg_names_str}, "
                f"got {len(actual_arg_types)}"
            )

        actual_array_type, actual_element_type = actual_arg_types

        # Validate first argument is ArrayType
        if not isinstance(actual_array_type, ArrayType):
            raise TypeMismatchError.from_message(
                f"{func_name} expects argument 0 to be an array type, "
                f"got {actual_array_type}"
            )

        # Validate element type matches array element type
        if actual_array_type.element_type != actual_element_type:
            raise TypeMismatchError(
                expected=actual_array_type.element_type,
                actual=actual_element_type,
                context=f"{func_name} Argument 1",
            )


class EqualTypes(TypeSignature):
    """Validates that two arguments have the same DataType (including metadata equality).

    Useful for non-singleton types like EmbeddingType, ArrayType, and StructType.
    """

    def __init__(self, expected_type: type):
        self.expected_type = expected_type

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != 2:
            raise InternalError(
                f"{func_name} expects 2 arguments, got {len(actual_arg_types)}"
            )

        # Check both arguments are instances of expected_type
        for i, actual_arg_type in enumerate(actual_arg_types):
            if not isinstance(actual_arg_type, self.expected_type):
                raise TypeMismatchError.from_message(
                    f"{func_name} expects argument {i} to be an instance of {self.expected_type.__name__}, "
                    f"got {actual_arg_type}"
                )

        # Check that both arguments are equal (including metadata)
        if actual_arg_types[0] != actual_arg_types[1]:
            raise TypeMismatchError.from_message(
                f"{func_name} expects both arguments to be equal. "
                f"Argument 0 has type {actual_arg_types[0]}, but argument 1 has type {actual_arg_types[1]}"
            )


class InstanceOf(TypeSignature):
    """Validates that arguments are instances of specific types.

    More descriptive than manually checking isinstance for each argument.
    """

    def __init__(self, expected_types: List[type]):
        self.expected_types = expected_types

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != len(self.expected_types):
            raise InternalError(
                f"{func_name} expects {len(self.expected_types)} arguments, "
                f"got {len(actual_arg_types)}"
            )

        for i, (expected_type, actual_arg_type) in enumerate(zip(self.expected_types, actual_arg_types, strict=False)):
            if not isinstance(actual_arg_type, expected_type):
                raise TypeMismatchError.from_message(
                    f"{func_name} expects argument {i} to be an instance of {expected_type.__name__}, "
                    f"got {actual_arg_type}"
                )
