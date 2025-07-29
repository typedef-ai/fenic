"""Function signature validation and return type inference.

This module provides the FunctionSignature class that combines type validation
with return type inference for functions.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Union

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.type_signature import (
    Exact,
    Numeric,
    TypeSignature,
    VariadicAny,
)
from fenic.core.error import InternalError
from fenic.core.types.datatypes import DataType, DoubleType, FloatType, IntegerType


class ReturnTypeStrategy(Enum):
    """Enum for special return type inference strategies."""
    SAME_AS_INPUT = auto()   # Return the same type as the first input
    PROMOTED = auto()        # Return promoted numeric type
    DYNAMIC = auto()         # Return type determined by function implementation


class FunctionSignature:
    """Complete signature for a function."""

    def __init__(
        self,
        function_name: str,
        type_signature: TypeSignature,
        return_type: Union[DataType, ReturnTypeStrategy]
    ):
        self.function_name = function_name
        self.type_signature = type_signature
        self.return_type = return_type

        # Validate return type strategy compatibility
        self._validate_return_type_compatibility()

    def validate_and_infer_type(
        self,
        args: List[LogicalExpr],
        plan: LogicalPlan,
        session_state: BaseSessionState,
        dynamic_return_type_func: Optional[Callable[[List[DataType], LogicalPlan, BaseSessionState], DataType]] = None
    ) -> DataType:
        """Validate arguments and infer return type using the plan's schema."""
        # Get types of all arguments using to_column_field
        arg_types = [arg.to_column_field(plan, session_state).data_type for arg in args]

        # Validate against signature (no implicit casting in initial implementation)
        self.type_signature.validate(arg_types, self.function_name)

        # Infer return type
        if self.return_type == ReturnTypeStrategy.DYNAMIC:
            if dynamic_return_type_func is None:
                raise InternalError(f"DYNAMIC return type requires dynamic_return_type_func for {self.function_name}")
            return_type = dynamic_return_type_func(arg_types, plan, session_state)
        else:
            return_type = self.infer_return_type(arg_types)

        return return_type


    def infer_return_type(self, arg_types: List[DataType]) -> DataType:
        """Infer return type from argument types."""
        if isinstance(self.return_type, DataType):
            return self.return_type
        elif self.return_type == ReturnTypeStrategy.SAME_AS_INPUT:
            return arg_types[0]
        elif self.return_type == ReturnTypeStrategy.PROMOTED:
            return self._promote_types(arg_types)
        elif self.return_type == ReturnTypeStrategy.DYNAMIC:
            raise InternalError("DYNAMIC return type requires dynamic_return_type_func")
        else:
            raise InternalError(f"Unknown return type: {self.return_type}")

    def _validate_return_type_compatibility(self) -> None:
        """Validate that return type strategy is compatible with type signature."""
        if self.return_type == ReturnTypeStrategy.SAME_AS_INPUT:
            if isinstance(self.type_signature, VariadicAny):
                raise InternalError(
                    f"{self.function_name}: SAME_AS_INPUT return type strategy not compatible "
                    f"with VariadicAny type signature (multiple different types)"
                )
            elif isinstance(self.type_signature, Exact):
                if len(self.type_signature.expected_arg_types) > 1:
                    # Check if all types are the same
                    first_type = self.type_signature.expected_arg_types[0]
                    if not all(t == first_type for t in self.type_signature.expected_arg_types):
                        raise InternalError(
                            f"{self.function_name}: SAME_AS_INPUT not compatible with "
                            f"Exact signature having different types"
                        )

        if self.return_type == ReturnTypeStrategy.PROMOTED:
            if not isinstance(self.type_signature, Numeric):
                raise InternalError(
                    f"{self.function_name}: PROMOTED return type strategy only compatible "
                    f"with Numeric type signature"
                )

    def _promote_types(self, arg_types: List[DataType]) -> DataType:
        """Promote numeric types to the most general type."""
        if not arg_types:
            raise InternalError("Cannot promote empty type list")

        # Simple promotion rules: Integer -> Float -> Double
        has_double = any(t == DoubleType for t in arg_types)
        has_float = any(t == FloatType for t in arg_types)

        if has_double:
            return DoubleType
        elif has_float:
            return FloatType
        else:
            return IntegerType
