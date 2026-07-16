import pytest
from fastmcp.exceptions import ValidationError
from fenic_mcp.server.utils.validation import validate_and_sanitize_regex


def test_basic_valid_pattern_passes():
    assert validate_and_sanitize_regex("alpha|beta") == "alpha|beta"


def test_slash_delimited_with_supported_flags_strips_and_passes():
    # Supported flags are i, m, s, x; they are ignored because we always apply case-insensitive upstream
    assert validate_and_sanitize_regex("/hello/i") == "hello"
    assert validate_and_sanitize_regex("/a.+b/ms") == "a.+b"


@pytest.mark.parametrize(
    "pattern",
    [
        None,
        "",
    ],
)
def test_empty_or_none_rejected(pattern):
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex(pattern)  # type: ignore[arg-type]


def test_unsupported_flags_rejected():
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("/foo/u")


def test_inline_flags_at_start_are_stripped():
    assert validate_and_sanitize_regex("(?i)Foo") == "Foo"
    assert validate_and_sanitize_regex("(?ms)Foo") == "Foo"


@pytest.mark.parametrize(
    "pattern",
    [
        "(unbalanced",
        "[abc",
        "{1,",
    ],
)
def test_unbalanced_constructs_rejected(pattern: str):
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex(pattern)


def test_quantifier_bounds_validated():
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("a{1001}")
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("a{5,2}")


def test_excessive_alternations_rejected():
    many_alts = "|".join(["a"] * 22)
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex(many_alts)


def test_backreferences_rejected():
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex(r"(a)\1")


def test_unsupported_inline_constructs_rejected():
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("(?P<name>foo)")  # named groups not allowed
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("(?<=a)b")  # lookbehind not allowed


def test_nested_quantifiers_redos_guard_rejected():
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("(.+)+")
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("(.*)+")
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("(?:.+){2,}")


def test_invalid_python_regex_rejected():
    with pytest.raises(ValidationError):
        validate_and_sanitize_regex("a{1,2,3}")
