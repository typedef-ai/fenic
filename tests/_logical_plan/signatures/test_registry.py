"""Test FunctionRegistry behavior."""

import pytest

from fenic.core._logical_plan.signatures import FunctionRegistry
from fenic.core.error import InternalError


def test_registry_unknown_function_error():
    """Test that unknown functions raise proper errors."""
    with pytest.raises(InternalError, match="Unknown function: nonexistent"):
        FunctionRegistry.get_signature("nonexistent")
