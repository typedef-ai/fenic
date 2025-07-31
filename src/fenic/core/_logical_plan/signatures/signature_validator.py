"""SignatureValidator for composition-based type validation.

This module provides the SignatureValidator class that wraps FunctionSignature functionality
for use in composition-based expressions instead of inheritance-based approaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from fenic.core._logical_plan.plans.base import LogicalPlan

from fenic.core._interfaces.session_state import BaseSessionState
from fenic.core._logical_plan.expressions.base import LogicalExpr
from fenic.core._logical_plan.signatures.function_signature import FunctionSignature
from fenic.core._logical_plan.signatures.registry import FunctionRegistry
from fenic.core.types.datatypes import DataType


class SignatureValidator:
    """Handles type signature validation and return type inference through composition.
    
    This class wraps FunctionSignature functionality to enable composition-based
    type validation instead of inheritance from deprecated function base classes.
    """

    def __init__(self, function_name: str):
        """Initialize SignatureValidator with function name for lazy signature loading.
        
        Args:
            function_name: Name of the function to validate, used for registry lookup
        """
        self.function_name = function_name
        self._signature: Optional[FunctionSignature] = None

    def validate_and_infer_type(
        self,
        args: List[LogicalExpr],
        plan: LogicalPlan,
        session_state: BaseSessionState,
        dynamic_return_type_func: Optional[Callable[[List[DataType], LogicalPlan, BaseSessionState], DataType]] = None
    ) -> DataType:
        """Validate arguments and infer return type using the plan's schema.
        
        This method mirrors the exact behavior of FunctionSignature.validate_and_infer_type
        but with lazy loading of the signature from the registry.
        
        Args:
            args: List of LogicalExpr arguments to validate
            plan: LogicalPlan object for schema context
            session_state: The session state to use for the new node
            dynamic_return_type_func: Optional function for dynamic return type inference
            
        Returns:
            DataType: The inferred return type after validation
            
        Raises:
            InternalError: If validation fails or signature is not found
        """
        # Lazy load signature from registry
        if self._signature is None:
            self._signature = FunctionRegistry.get_signature(self.function_name)
        
        # Delegate to the signature's validate_and_infer_type method
        return self._signature.validate_and_infer_type(args, plan, session_state, dynamic_return_type_func)