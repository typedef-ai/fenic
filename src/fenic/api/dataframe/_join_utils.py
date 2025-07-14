"""Utility functions for DataFrame join operations."""

from typing import List, Optional, Tuple, Union, Mapping

from fenic.api.column import Column, ColumnOrName
from fenic.core._logical_plan import LogicalExpr
from fenic.core._logical_plan.expressions import ColumnExpr, AliasExpr
from fenic.core.error import ValidationError
from fenic.core.types.enums import JoinType
from fenic.core._utils import misc as misc_utils


def validate_join_parameters(
    on: Optional[Union[str, List[str]]],
    left_on: Optional[Union[ColumnOrName, List[ColumnOrName]]],
    right_on: Optional[Union[ColumnOrName, List[ColumnOrName]]],
    how: JoinType
) -> None:
    """Validate join parameter combinations."""
    # Check mutual exclusivity of 'on' vs 'left_on'/'right_on'
    if on is not None and (left_on is not None or right_on is not None):
        raise ValidationError(
            "Cannot use 'on' parameter together with 'left_on'/'right_on' parameters. "
            "Use either 'on' for simple joins or both 'left_on' and 'right_on' for complex joins."
        )

    # Check that left_on/right_on are used together
    if (left_on is not None) != (right_on is not None):
        missing = "right_on" if left_on is not None else "left_on"
        provided = "left_on" if left_on is not None else "right_on"
        raise ValidationError(
            f"Both 'left_on' and 'right_on' must be provided together. "
            f"Got {provided} but missing {missing}."
        )

    # Validate cross join constraints
    if how == "cross":
        if _has_join_conditions(on, left_on, right_on):
            raise ValidationError(
                "Cross joins cannot have join conditions. "
                "Remove 'on', 'left_on', and 'right_on' parameters for cross joins."
            )
    else:
        # Non-cross joins require conditions
        if not _has_join_conditions(on, left_on, right_on):
            raise ValidationError(
                f"Join type '{how}' requires join conditions. "
                f"Provide either 'on' parameter or both 'left_on' and 'right_on' parameters."
            )

    # Validate matching lengths for left_on/right_on
    if left_on is not None and right_on is not None:
        _validate_join_condition_lengths(left_on, right_on)

def build_join_conditions(
    on: Optional[Union[str, List[str]]],
    left_on: Optional[Union[ColumnOrName, List[ColumnOrName]]],
    right_on: Optional[Union[ColumnOrName, List[ColumnOrName]]]
) -> Tuple[List, List]:
    """Build left and right join condition lists."""
    if left_on is not None and right_on is not None:
        # Convert to lists if needed
        left_cols = left_on if isinstance(left_on, list) else [left_on]
        right_cols = right_on if isinstance(right_on, list) else [right_on]

        # Build condition expressions
        left_conditions = [Column._from_col_or_name(col)._logical_expr for col in left_cols]
        right_conditions = [Column._from_col_or_name(col)._logical_expr for col in right_cols]

    elif on is not None:
        # Convert to list if needed
        on_cols = on if isinstance(on, list) else [on]

        # For 'on' parameter, same conditions apply to both sides
        conditions = [Column._from_col_or_name(col)._logical_expr for col in on_cols]
        left_conditions = conditions
        right_conditions = conditions

    else:
        # Cross joins have no conditions
        left_conditions = []
        right_conditions = []

    return left_conditions, right_conditions

def _has_join_conditions(
    on: Optional[Union[str, List[str]]],
    left_on: Optional[Union[ColumnOrName, List[ColumnOrName]]],
    right_on: Optional[Union[ColumnOrName, List[ColumnOrName]]]
) -> bool:
    """Check if any join conditions are specified."""
    return (
        (on is not None and (not isinstance(on, list) or len(on) > 0)) or
        left_on is not None or
        right_on is not None
    )

def _validate_join_condition_lengths(
    left_on: Union[ColumnOrName, List[ColumnOrName]],
    right_on: Union[ColumnOrName, List[ColumnOrName]]
) -> None:
    """Validate that left_on and right_on have matching lengths."""
    left_cols = left_on if isinstance(left_on, list) else [left_on]
    right_cols = right_on if isinstance(right_on, list) else [right_on]

    if len(left_cols) != len(right_cols):
        raise ValidationError(
            f"Length mismatch: 'left_on' has {len(left_cols)} columns, "
            f"'right_on' has {len(right_cols)} columns. Both must have the same length."
        )


def _resolve_join_binding(
    placeholder_key: str,
    bindings: Optional[Mapping[str, Column]] = None
) -> Union[AliasExpr, ColumnExpr]:
    if placeholder_key in bindings:
        column = bindings[placeholder_key]
        logical_expr = column._logical_expr

        # Check if the expression is already aliased
        if isinstance(logical_expr, AliasExpr):
            # Validate that the alias matches the key
            if logical_expr.name != placeholder_key:
                raise ValidationError(
                    f"Alias name must match the key. Expected '{placeholder_key}', got '{logical_expr.name}'")
            return logical_expr
        else:
            aliased_expr = AliasExpr(logical_expr, placeholder_key)
            return aliased_expr
    else:
        # Use column reference for missing keys
        return ColumnExpr(placeholder_key)


def resolve_join_bindings(
    instruction: str,
    bindings: Optional[Mapping[str, Column]]
) -> tuple[Union[AliasExpr, ColumnExpr], Union[AliasExpr, ColumnExpr]]:
    """Resolve placeholder bindings to logical expressions with proper aliasing and validation.

    Args:
        instruction: The instruction string with placeholders
        bindings: Mapping of placeholder names to column expressions (can be None)

    Returns:
        List of logical expressions for each unique placeholder in the instruction

    Raises:
        ValidationError: If alias names don't match keys or expression names don't match keys
    """
    if bindings is None:
        bindings = {}
    placeholder_keys = misc_utils.parse_instruction(instruction)
    if len(placeholder_keys) != 2:
        raise ValidationError(
            f"Join instructions must have exactly two placeholders, 'placeholder:left' and 'placeholder:right'."
            f" Got {len(placeholder_keys)} placeholders: {placeholder_keys}"
        )
    left_on: Optional[str] = None
    right_on: Optional[str] = None

    for placeholder_key in placeholder_keys:
        if placeholder_key.endswith(":left"):
            left_on = placeholder_key.split(":")[0]
        elif placeholder_key.endswith(":right"):
            right_on = placeholder_key.split(":")[0]
    if left_on is None or right_on is None:
        raise ValidationError(
            f"Join instructions must have 2 placeholders named 'placeholder:left' and 'placeholder:right'. "
            f"Got placeholders: {placeholder_keys}"
        )

    left_expr = _resolve_join_binding(left_on, bindings)
    right_expr = _resolve_join_binding(right_on, bindings)
    return left_expr, right_expr
