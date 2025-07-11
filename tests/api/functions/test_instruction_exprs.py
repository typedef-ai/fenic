import pytest

from fenic.api.functions.core import col
from fenic.api.functions.semantic import _build_instruction_exprs
from fenic.api.functions.text import concat
from fenic.core.error import ValidationError


def test_build_instruction_exprs_accepts_unaliased():
    exprs = _build_instruction_exprs(
        instruction="Hello {full_name}",
        bindings={"full_name": concat(col("first"), col("last"))}
    )
    # Should alias the expression as 'full_name'
    assert len(exprs) == 1
    assert getattr(exprs[0], 'name', None) == "full_name"


def test_build_instruction_exprs_accepts_correct_alias():
    exprs = _build_instruction_exprs(
        instruction="Hello {full_name}",
        bindings={"full_name": concat(col("first"), col("last")).alias("full_name")}
    )
    assert len(exprs) == 1
    assert getattr(exprs[0], 'name', None) == "full_name"


def test_build_instruction_exprs_raises_on_alias_mismatch():
    with pytest.raises(ValidationError, match="Alias name must match the key"):
        _build_instruction_exprs(
            instruction="Hello {full_name}",
            bindings={"full_name": concat(col("first"), col("last")).alias("wrong_name")}
        )


def test_build_instruction_exprs_raises_on_name_mismatch():
    class DummyExpr:
        def __init__(self, name):
            self.name = name
        # Simulate not being an AliasExpr
        _logical_expr = None  # Add the required attribute
    
    class MockColumn:
        def __init__(self, name):
            self._logical_expr = DummyExpr(name)
    
    with pytest.raises(ValidationError, match="Expression name must match the key"):
        _build_instruction_exprs(
            instruction="Hello {full_name}",
            bindings={"full_name": MockColumn("wrong_name")}
        )


def test_build_instruction_exprs_deduplicates_placeholders():
    """Test that duplicate placeholders in instruction only create one expression."""
    exprs = _build_instruction_exprs(
        instruction="Hello {name}, how are you {name}? Is {name} satisfied?",
        bindings={"name": col("name")}
    )
    # Should only have one expression despite {name} appearing 3 times
    assert len(exprs) == 1
    assert getattr(exprs[0], 'name', None) == "name"


def test_build_instruction_exprs_deduplicates_multiple_placeholders():
    """Test deduplication with multiple different placeholders."""
    exprs = _build_instruction_exprs(
        instruction="Hello {name}, you live in {city}. How do you like {city}, {name}?",
        bindings={"name": col("name"), "city": col("city")}
    )
    # Should have 2 expressions (name and city) despite multiple occurrences
    assert len(exprs) == 2
    names = [getattr(expr, 'name', None) for expr in exprs]
    assert sorted(names) == sorted(["name", "city"]) 