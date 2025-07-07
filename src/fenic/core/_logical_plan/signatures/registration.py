"""Register all function signatures after expressions are loaded.

This module is imported after the expression classes are fully loaded
to avoid circular import issues.
"""

from fenic.core._logical_plan.signatures import (
    arithmetic,
    basic,
    case,
    comparison,
    text,
)

# Register all signatures
arithmetic.register_arithmetic_signatures()
comparison.register_comparison_signatures()
case.register_case_signatures()
basic.register_basic_signatures()
text.register_text_signatures()
