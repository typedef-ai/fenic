import pytest

from fenic.api.functions.core import col
from fenic.api.functions.semantic import _resolve_bindings
from fenic.api.functions.text import concat
from fenic.core.error import ValidationError


def test_resolve_bindings_accepts_unaliased():
    exprs = _resolve_bindings(
        instruction="Hello {full_name}",
        bindings={"full_name": concat(col("first"), col("last"))}
    )
    # Should alias the expression as 'full_name'
    assert len(exprs) == 1
    assert getattr(exprs[0], 'name', None) == "full_name"


def test_resolve_bindings_accepts_correct_alias():
    exprs = _resolve_bindings(
        instruction="Hello {full_name}",
        bindings={"full_name": concat(col("first"), col("last")).alias("full_name")}
    )
    assert len(exprs) == 1
    assert getattr(exprs[0], 'name', None) == "full_name"


def test_resolve_bindings_raises_on_alias_mismatch():
    with pytest.raises(ValidationError, match="Alias name must match the key"):
        _resolve_bindings(
            instruction="Hello {full_name}",
            bindings={"full_name": concat(col("first"), col("last")).alias("wrong_name")}
        )

def test_resolve_bindings_deduplicates_multiple_placeholders():
    """Test deduplication with multiple different placeholders."""
    exprs = _resolve_bindings(
        instruction="Hello {name}, you live in {city}. How do you like {city}, {name}?",
        bindings={"name": col("name"), "city": col("city")}
    )
    # Should have 2 expressions (name and city) despite multiple occurrences
    assert len(exprs) == 2
    names = [getattr(expr, 'name', None) for expr in exprs]
    assert sorted(names) == sorted(["name", "city"]) 