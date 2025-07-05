"""Simplified type signature classes for function validation.

This module provides a streamlined TypeSignature hierarchy focused solely on 
validating LogicalExpr arguments with standard DataTypes.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

from fenic.core.error import TypeMismatchError, ValidationError
from fenic.core.types.datatypes import (
    ArrayType,
    DataType,
    StringType,
    StructType,
    is_dtype_numeric,
)


class TypeSignature(ABC):
    """Base class for type signatures."""

    @abstractmethod
    def validate(self, arg_types: List[DataType], func_name: str) -> None:
        """Validate that argument types match this signature."""
        pass

    def get_expected_types(self, arg_types: List[DataType]) -> List[DataType]:
        """Get the expected types for implicit casting.

        Default implementation returns arg_types (no specific expectation).
        Override in subclasses that have specific type expectations.
        """
        return arg_types


class PositionalSignature(TypeSignature):
    """Position-based signature validation with flexible constraints.

    Validates arguments by position using a list of constraints. Each position can specify
    exact DataType matching, isinstance checking, or use exact_values for singleton types.
    Supports custom validation logic for complex relationships between arguments.
    """

    def __init__(
        self,
        constraints: List[Union[DataType, type]],
        exact_values: Optional[List[DataType]] = None,
        custom_validator: Optional[Callable[[List[DataType], str], None]] = None,
        arg_names: Optional[List[str]] = None
    ):
        """Initialize with position-based type constraints.

        Args:
            constraints: List of DataType instances (for exact matching) or type classes
                        (for isinstance checking) that arguments must satisfy.
            exact_values: Optional list of exact DataType instances to match. Use None
                         for positions that should use isinstance() checking.
            custom_validator: Optional function for additional validation logic.
            arg_names: Optional list of argument names for error messages.
        """
        self.constraints = constraints
        self.exact_values = exact_values or [None] * len(constraints)
        self.custom_validator = custom_validator
        self.arg_names = arg_names

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != len(self.constraints):
            if self.arg_names:
                arg_names_str = f" ({', '.join(self.arg_names)})"
            else:
                arg_names_str = ""
            raise ValidationError(
                f"{func_name} expects {len(self.constraints)} arguments{arg_names_str}, "
                f"got {len(actual_arg_types)}"
            )

        # Validate each argument position
        for i, (constraint, actual_arg_type) in enumerate(zip(self.constraints, actual_arg_types, strict=False)):
            exact_value = self.exact_values[i]

            if exact_value is not None:
                # Check for exact equality (singleton types)
                if actual_arg_type != exact_value:
                    raise TypeMismatchError(
                        expected=exact_value,
                        actual=actual_arg_type,
                        context=f"{func_name} Argument {i}",
                    )
            elif isinstance(constraint, type):
                # Check isinstance (non-singleton types)
                if not isinstance(actual_arg_type, constraint):
                    if constraint == ArrayType:
                        raise TypeMismatchError.from_message(
                            f"{func_name} expects argument {i} to be an array type, "
                            f"got {actual_arg_type}"
                        )
                    elif constraint == StructType:
                        raise TypeMismatchError.from_message(
                            f"{func_name} expects argument {i} to be a struct type, "
                            f"got {actual_arg_type}"
                        )
                    else:
                        raise TypeMismatchError.from_message(
                            f"{func_name} expects argument {i} to be an instance of {constraint.__name__}, "
                            f"got {actual_arg_type}"
                        )
            else:
                # DataType instance - check for exact equality
                if actual_arg_type != constraint:
                    raise TypeMismatchError(
                        expected=constraint,
                        actual=actual_arg_type,
                        context=f"{func_name} Argument {i}",
                    )

        # Apply custom validation if provided
        if self.custom_validator:
            self.custom_validator(actual_arg_types, func_name)

    def get_expected_types(self, arg_types: List[DataType]) -> List[DataType]:
        """Return expected types from constraints for implicit casting."""
        expected_types = []
        for i, constraint in enumerate(self.constraints):
            if i < len(arg_types):
                exact_value = self.exact_values[i] if i < len(self.exact_values) else None
                if exact_value is not None:
                    expected_types.append(exact_value)
                elif isinstance(constraint, DataType):
                    expected_types.append(constraint)
                else:
                    # For type classes, return the actual type (no expectation)
                    expected_types.append(arg_types[i])
            else:
                # No argument provided for this constraint
                if isinstance(constraint, DataType):
                    expected_types.append(constraint)
                else:
                    break
        return expected_types


class Exact(PositionalSignature):
    """Exact argument types for functions (e.g., length(str) -> int).

    Syntactic sugar for PositionalSignature with DataType instances.
    """

    def __init__(self, expected_arg_types: List[DataType]):
        super().__init__(expected_arg_types)



class Uniform(TypeSignature):
    """All arguments must be the same type."""

    def __init__(self, expected_num_args: int, required_type: Optional[DataType] = None):
        self.expected_num_args = expected_num_args
        self.required_type = required_type

    def validate(self, actual_arg_types: List[DataType], func_name: str) -> None:
        if len(actual_arg_types) != self.expected_num_args:
            raise ValidationError(
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

    def get_expected_types(self, arg_types: List[DataType]) -> List[DataType]:
        """Return expected uniform type if required_type is specified."""
        if self.required_type and arg_types:
            return [self.required_type] * len(arg_types)
        return arg_types


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
            raise ValidationError(
                f"{func_name} expects {self.expected_num_args} arguments, "
                f"got {len(actual_arg_types)}"
            )

        for i, actual_arg_type in enumerate(actual_arg_types):
            if not is_dtype_numeric(actual_arg_type):
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


class ArrayOfAny(PositionalSignature):
    """Matches any ArrayType regardless of element type."""

    def __init__(self, expected_num_args: int = 1):
        constraints = [ArrayType] * expected_num_args
        super().__init__(constraints)


class ArrayWithMatchingElement(PositionalSignature):
    """Validates array + element where element type must match array element type."""

    def __init__(self):
        def custom_validator(actual_arg_types: List[DataType], func_name: str) -> None:
            actual_array_type, actual_element_type = actual_arg_types
            if actual_array_type.element_type != actual_element_type:
                raise TypeMismatchError(
                    expected=actual_array_type.element_type,
                    actual=actual_element_type,
                    context=f"{func_name} Argument 1",
                )
        
        super().__init__([ArrayType, DataType], custom_validator=custom_validator, arg_names=["array", "element"])


class StructWithStringKey(PositionalSignature):
    """Validates struct + string key for field access."""

    def __init__(self):
        super().__init__([StructType, StringType], arg_names=["struct", "field_name"])
