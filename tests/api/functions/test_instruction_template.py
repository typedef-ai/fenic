import pytest

from fenic.api.functions.core import col
from fenic.api.functions.text import concat
from fenic.api.types import InstructionTemplate
from fenic.core.error import ValidationError


def test_instruction_template_accepts_unaliased():
    template = InstructionTemplate(
        "Hello {full_name}",
        full_name=concat(col("first"), col("last"))
    )
    exprs = template.to_resolved_template().exprs
    # Should alias the expression as 'full_name'
    assert len(exprs) == 1
    assert exprs[0].name == "full_name"


def test_instruction_template_accepts_correct_alias():
    template = InstructionTemplate(
        "Hello {full_name}",
        full_name=concat(col("first"), col("last")).alias("full_name")
    )
    exprs = template.to_resolved_template().exprs
    assert len(exprs) == 1
    assert exprs[0].name == "full_name"


def test_instruction_template_raises_on_alias_mismatch():
    template = InstructionTemplate(
        "Hello {full_name}",
        full_name=concat(col("first"), col("last")).alias("wrong_name")
    )
    with pytest.raises(ValidationError, match="Alias name must match the key"):
        template.to_resolved_template()

def test_instruction_template_deduplicates_multiple_placeholders():
    """Test deduplication with multiple different placeholders."""
    template = InstructionTemplate(
        "Hello {name}, you live in {city}. How do you like {city}, {name}?",
        name=col("name"),
        city=col("city")
    )
    exprs = template.to_resolved_template().exprs
    # Should have 2 expressions (name and city) despite multiple occurrences
    assert len(exprs) == 2
    names = [expr.name for expr in exprs]
    assert sorted(names) == sorted(["name", "city"]) 