"""Utility functions for DataFrame join operations."""

from typing import List, Optional, Tuple, Union

from fenic.api.column import Column, ColumnOrName
from fenic.core.error import ValidationError
from fenic.core.types.enums import JoinType


def validate_join_parameters(
    self,
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
