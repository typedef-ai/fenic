import pytest

from fenic.core.error import ValidationError
from fenic.core.mcp._validators import (
    RegexValidator,
    get_param_validator,
    register_param_validator,
)


def test_regex_validator_accepts_simple_pattern():
    v = RegexValidator()
    v.validate("foo|bar")


def test_regex_validator_supports_slash_delimiters_and_flags():
    v = RegexValidator()
    v.validate("/foo.*/i")


@pytest.mark.parametrize(
    "pattern",
    [
        "   ",  # whitespace-only
        "(",  # unbalanced paren
        "[a-",  # unbalanced bracket
        "{1,2,3}",  # invalid quantifier syntax
    ],
)
def test_regex_validator_rejects_basic_invalid_patterns(pattern):
    v = RegexValidator()
    with pytest.raises(ValidationError):
        v.validate(pattern)


@pytest.mark.parametrize(
    "pattern",
    [
        r"(.+)+",  # nested quantifier
        r"(.*)+",  # nested quantifier
        r"(?:.+){1001}",  # excessive bounded quantifier
        r"(a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|aa|ab|ac)",  # too many alternations
        r"\1",  # backreference
        r"(?<=a)b",  # lookbehind
        r"(\.\*){2,1}",  # m > n for matching between m and n repeats of a character
    ],
)
def test_regex_validator_rejects_redos_like_and_unsupported_constructs(pattern):
    v = RegexValidator()
    with pytest.raises(ValidationError):
        v.validate(pattern)


def test_registry_has_default_regex_validator():
    v = get_param_validator("regex")
    assert isinstance(v, RegexValidator)


def test_registry_register_and_lookup_custom():
    import uuid

    unique_name = f"custom_regex_{uuid.uuid4().hex}"
    register_param_validator(unique_name, RegexValidator())
    v = get_param_validator(unique_name)
    assert isinstance(v, RegexValidator)
