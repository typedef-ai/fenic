"""Register all function signatures after expressions are loaded.

This module is imported after the expression classes are fully loaded
to avoid circular import issues.
"""

from fenic.core._logical_plan.signatures import basic

# Register all signatures
basic.register_basic_signatures()
