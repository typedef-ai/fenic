"""Function signature validation and return type inference.

This module provides the FunctionSignature class that combines type validation
with return type inference for functions.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan

from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.types import (
    Exact,
    Numeric,
    TypeSignature,
    VariadicAny,
)
from fenic.core._logical_plan.utils import can_cast
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
        return_type: Union[DataType, ReturnTypeStrategy],
        allow_implicit_casting: bool = True
    ):
        self.function_name = function_name
        self.type_signature = type_signature
        self.return_type = return_type
        self.allow_implicit_casting = allow_implicit_casting

        # Validate return type strategy compatibility
        self._validate_return_type_compatibility()

    def validate_and_infer_type(
        self, 
        args: List[LogicalExpr], 
        plan: LogicalPlan, 
        dynamic_return_type_func: Optional[Callable[[List[DataType], LogicalPlan], DataType]] = None
    ) -> Tuple[DataType, List[LogicalExpr]]:
        """Validate arguments and infer return type using the plan's schema.

        Returns:
            Tuple of (return_type, final_args) where final_args may include implicit casts.
        """
        # Get types of all arguments using to_column_field
        arg_types = [arg.to_column_field(plan).data_type for arg in args]

        # Validate and apply implicit casting
        final_args, final_types = self._validate_and_cast(args, arg_types, plan)

        # Infer return type using final types
        if self.return_type == ReturnTypeStrategy.DYNAMIC:
            if dynamic_return_type_func is None:
                raise InternalError(f"DYNAMIC return type requires dynamic_return_type_func for {self.function_name}")
            return_type = dynamic_return_type_func(arg_types, plan)
        else:
            return_type = self.infer_return_type(final_types)

        return return_type, final_args


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

    def _validate_and_cast(self, args: List[LogicalExpr], arg_types: List[DataType], plan: LogicalPlan) -> Tuple[List[LogicalExpr], List[DataType]]:
        """Validate types and apply implicit casts if needed to make validation pass."""
        # If implicit casting is disabled, validate directly without casting
        if not self.allow_implicit_casting:
            self.type_signature.validate(arg_types, self.function_name)
            return args, arg_types

        # Import here to avoid circular imports
        from fenic.core._logical_plan.expressions.basic import CastExpr

        # First, check what types the signature expects
        expected_types = self.type_signature.get_expected_types(arg_types)

        # Apply casts where needed
        final_args = []
        final_types = []

        for _i, (arg, actual_type, expected_type) in enumerate(zip(args, arg_types, expected_types, strict=False)):
            if actual_type != expected_type and can_cast(actual_type, expected_type):
                # Insert cast node
                cast_expr = CastExpr(arg, expected_type)
                final_args.append(cast_expr)
                final_types.append(expected_type)
            else:
                # No cast needed or not possible
                final_args.append(arg)
                final_types.append(actual_type)

        # Now validate with final types
        self.type_signature.validate(final_types, self.function_name)

        return final_args, final_types